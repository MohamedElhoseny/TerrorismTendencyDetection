package Models.Sentiment_Analysis.Methodology;

import Models.Sentiment_Analysis.Preprocessing.TextPreprocessor;
import Models.Sentiment_Analysis.Preprocessing.TweetPreprocessor;
import Models.Sentiment_Analysis.Utils.DBInstanceIterator;
import Models.Sentiment_Analysis.Utils.SVMClassifier;
import Models.Sentiment_Analysis.Utils.SVMClassifierTrainer;
import Models.Sentiment_Analysis.Utils.TokenSequence2PorterStems;
import ca.uwo.csd.ai.nlp.kernel.LinearKernel;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.NaiveBayes;
import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.pipe.*;
import cc.mallet.types.*;
import org.apache.commons.collections4.BidiMap;
import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

public class Trainer implements Serializable
{
	//BidiMap<String, Integer> tba; //for text attribute
	//BidiMap<String, Integer> fba; //for feature attribute
	//BidiMap<String, Integer> cba; //for complex attribute

	private DBInstanceIterator dbiterator;
	protected static final String EN_stopwords = "Resources/stoplists/en.txt";
	public static final String Dataset = "Resources/datasets/sentimentanalysis/mallet_domain/out/";
	public static final String Train = "Resources/datasets/sentimentanalysis/mallet_train/";  //for saving preprocessed Training set
	public static final String Test = "Resources/datasets/sentimentanalysis/mallet_test/";  //for saving test dataset models
	public static final String Attributes = "Resources/datasets/sentimentanalysis/mallet_attributes/"; //for saving attribute of training set
	//Those for saving Trained Model
	public static final String Lexicon_Train = "Resources/datasets/sentimentanalysis/mallet_models/lexicon.bin";
	public static final String Text_Train = "Resources/datasets/sentimentanalysis/mallet_models/text.bin";
	public static final String Feature_Train = "Resources/datasets/sentimentanalysis/mallet_models/feature.bin";
	public static final String Complex_Train = "Resources/datasets/sentimentanalysis/mallet_models/complex.bin";


	/**
	 *  Initialize new Object from BidiMap to text,featue,complex attribute
	 *  @param f directory name where the "training" set saved
	 */
	public Trainer()
	{
		//tba = new DualHashBidiMap<>();
		//fba = new DualHashBidiMap<>();
		//cba = new DualHashBidiMap<>();
	}

	/**  Train Text, Lexicon, Feature, Complex datasets */
	public void train()
	{
		dbiterator = DBInstanceIterator.getInstance();
        System.out.println("Start Training ..");

		try {
			System.out.println("Training Text %..");
			trainText();

			System.out.println("Training Complex %..");
			trainCombined();

			System.out.println("Training Feature %..");
			trainFeatures();

			System.out.println("Training Lexicon %..");
			//trainLexicon();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	/** Get Test dataset from db and Pass them to Text Pipe to preprocess them, Train NB and save Test Model
	 * This Method must be called after preparing Test dataset*/
	public void trainTest() throws Exception
	{
		System.out.println("Preprocessing Testset ..");
		//String[] dir = new String[]{ Test+"negative" , Test+"positive"};
		//FileIterator iterator = new FileIterator(dir,FileIterator.LAST_DIRECTORY);
		InstanceList instances = new InstanceList(getTextPipe());
		//instances.addThruPipe(iterator);

		CSVParser parser = new CSVParser(new FileReader(
				"resources/datasets/sentimentanalysis/mallet_test/Sentiment140/sentiment140.csv"),
				CSVFormat.EXCEL.withFirstRecordAsHeader());

		TextPreprocessor preprocessor = new TextPreprocessor("resources/datasets/sentimentanalysis/");
		int count = 0;
		for (CSVRecord record: parser.getRecords())
		{
			String target;
			if (record.get("target").equals("0"))
				target = "negative";
			else
				target = "positive";
			instances.addThruPipe(new Instance(preprocessor.getProcessed(record.get("tweet")),target,
					"Instance"+count++,null));

			System.out.println(count);
		}

		System.out.println(instances.targetLabelDistribution());
		System.out.println("Start Training Testset ..");
		Classifier nb = new NaiveBayesTrainer().train(instances);
		System.out.println("Saving Test Model ..");
		saveModel(nb,Test+"Model.bin");
		saveinstances(instances,Test+"Model-Instances.bin");
		instances.getDataAlphabet().dump(new PrintWriter(new File(Test+"Model-alphabet.dat")));
	}

	/**Responsible for Preparing and saving datasets read and preprocessed, that will be used for training
	 * Must be called after reading dataset file and divide it into 4 datasets (text,complex,lexicon,feature) */
	public void PrepareDataset(TweetPreprocessor preprocessor, String dataset)
	{
		dbiterator = DBInstanceIterator.getInstance();
		try
        {
            System.out.println("Preparing Datasets ...");
            //PrepareAll(preprocessor,dataset);

			System.out.println("Saving work to Database ..");
			saveTextOutput(Dataset + "data_text.csv", "d_text");
			saveTextOutput(Dataset + "data_feature.csv", "d_feature");
			saveTextOutput(Dataset + "data_complex.csv", "d_complex");
			//saveLexiconOuput(Dataset + "data_lexicon.csv",true);
			System.out.println("Datasets ready for training .%.");

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**Responsible for Preparing and saving labeled or unlabeled datasets*/
	public void PrepareTestDataset(String dataset, boolean islabeled)
	{
		dbiterator = DBInstanceIterator.getInstance();
		System.out.println("Preparing Test Datasets ...");
		try {
			PrepareAll(dataset,islabeled);
			System.out.println("Test Dataset is saved & ready for Training ..");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	//<editor-fold desc="Preparing & Saving Dataset">

	/**representing data in files with corresponding preprocess feature (unigram,bigram,POS,lexicon based)*/
	private void PrepareAll(TweetPreprocessor tweetPreprocessor, String Datasetpath) throws IOException
	{
		CSVParser parser = new CSVParser(new FileReader(Datasetpath),CSVFormat.EXCEL.withFirstRecordAsHeader());

		ExecutorService threadpool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		final ConcurrentHashMap<String, String> txtHashMap = new ConcurrentHashMap<>();
		final ConcurrentHashMap<String, String> featHashMap = new ConcurrentHashMap<>();
		final ConcurrentHashMap<String, String> compHashMap = new ConcurrentHashMap<>();
		final ConcurrentHashMap<String, String> lexHashMap = new ConcurrentHashMap<>();

		int count = 0;
		for (CSVRecord record : parser.getRecords())
		{
			System.out.println("Adding Task = " + count);
			int finalCount = count;

			if (count == 17000)
				break;
			threadpool.execute(() ->
			{
				System.out.println("Current Executing task : " + finalCount);
				Instance tweet = new Instance(record.get("event"),record.get("target"),null,null);
				Instance[] preparedinstances = tweetPreprocessor.startProc(tweet);

				txtHashMap.put(preparedinstances[0].getData().toString(), record.get("target"));
				featHashMap.put(preparedinstances[1].getData().toString(), record.get("target"));
				compHashMap.put(preparedinstances[2].getData().toString(), record.get("target"));
				lexHashMap.put(preparedinstances[3].getData().toString(), record.get("target"));
			});
			count++;
		}
		//Initiates an orderly shutdown in which previously submitted tasks are executed,
        //but no new tasks will be accepted
		threadpool.shutdown();

		while (!threadpool.isTerminated());
		System.out.println("ThreadPool is shutdown ..");

		try (BufferedWriter writer1 = new BufferedWriter(new FileWriter(Dataset + "data_text.csv"));
			 BufferedWriter writer2 = new BufferedWriter(new FileWriter(Dataset + "data_feature.csv"));
			 BufferedWriter writer3 = new BufferedWriter(new FileWriter(Dataset + "data_complex.csv"));
			 BufferedWriter writer4 = new BufferedWriter(new FileWriter(Dataset + "data_lexicon.csv")))
		{
			for (String tweet : txtHashMap.keySet())
				writer1.write(txtHashMap.get(tweet) + "," + tweet + "\n");
			for (String tweet : compHashMap.keySet())
				writer3.write(compHashMap.get(tweet) + "," + tweet + "\n");
			for (String tweet : lexHashMap.keySet())
				writer4.write(lexHashMap.get(tweet) + "," + tweet + "\n");
			for (String tweet : featHashMap.keySet())
				writer2.write(featHashMap.get(tweet) + "," + tweet + "\n");

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void PrepareAll(String Datasetpath,boolean islabeled) throws IOException
	{
		CSVParser parser = new CSVParser(new FileReader(Datasetpath),
				CSVFormat.EXCEL.withFirstRecordAsHeader());
		BufferedWriter writer = new BufferedWriter(new FileWriter(Dataset + "data_test.txt"));

		StringBuilder sql = new StringBuilder("INSERT INTO d_test (data,target,name,source) Values ");

		for (CSVRecord record : parser.getRecords())
		{
			String tweet = record.get("tweet");
			String target;
			if (islabeled) target = record.get("target");
			else target = "?";

			writer.write(target + "," + tweet + "\n");

			sql.append("('")
					.append(tweet.replaceAll("'", "").replaceAll("\\\\", ""))
					.append("','").append(target).append("','TestInstance','TestDataset'),");
		}

		sql.deleteCharAt(sql.length() - 1);
		sql.append(";");
		saveOutput(sql);
		writer.close();

	}

	public void reformatLexiconFile(String filepath) throws IOException
	{
	    /*
            Input

            negative,noun(1)=0.19520873828322935
			adj(2)=0.12741157076130974
			wordnet(4)=0.3226203090445391
			polarity(5)=-2.0

			negative,verb(0)=0.1425619834710744
			noun(1)=0.046532846715328466
			wordnet(4)=0.18909483018640288
			polarity(5)=-1.0

            Output
            negative,verb=-0.019185326611942655,noun=-0.23857991238766202,wordnet=-0.2577652389996047

        */
		File preparedfile = new File(Dataset+"data_lexicon_temp.csv");
		BufferedReader reader = new BufferedReader(new FileReader(filepath));
		BufferedWriter writer = new BufferedWriter(new FileWriter(preparedfile));

		String str;
		while ((str = reader.readLine()) != null)
		{
			if (!str.isEmpty()) {
				if (str.startsWith("negative,") || str.startsWith("positive,"))
					writer.write(str.replaceAll("[(][0-9][)]", ""));
				else
					writer.write(","+str.replaceAll("[(][0-9][)]", ""));
			}else
				writer.newLine();
		}
		writer.close();
		reader.close();

		Files.copy(preparedfile.toPath(),
				new File(Dataset+"data_lexicon.csv").toPath(), StandardCopyOption.REPLACE_EXISTING);

		preparedfile.delete();
	}

	private void saveLexiconOuput(String preparedfile, boolean isreformat) throws IOException
	{
		if (!isreformat)
		{
			reformatLexiconFile(preparedfile);
			repaircsv(preparedfile);
		}

		ArrayList<String> features = new ArrayList<>(Arrays.asList("verb", "noun","adj","adv","wordnet","polarity"));
		BufferedReader reader = new BufferedReader(new FileReader(preparedfile));
        dbiterator = DBInstanceIterator.getInstance();

		StringBuilder sql = new StringBuilder("INSERT INTO d_lexicon (verb,noun,adj,adv,wordnet,polarity,target) Values  ");
		String str;
		while ((str = reader.readLine()) != null)
		{
			String[] tokens = str.split(",");
		    String target = tokens[0];

			sql.append("(");
			int i = 1;
			for (String s: features)
			{
				if (i < tokens.length && tokens[i].startsWith(s)){
					sql.append(tokens[i].split("=")[1]);
					i++;
				}else
					sql.append("0");

				sql.append(",");
			}
			sql.append("'").append(target).append("'),");
		}

		sql.deleteCharAt(sql.length()-1).append(";");
		saveOutput(sql);
		System.out.println("Lexicon data saved.");
	}

    private void saveTextOutput(String preparedfile, String dbtable) throws IOException
	{
		repaircsv(preparedfile);
		//0, usermentionsymbol Sorry to hear about your loss  There have been many this year so far
		BufferedReader reader = new BufferedReader(new FileReader(preparedfile));
		String str;
		StringBuilder sql = new StringBuilder("INSERT INTO "+dbtable+" (data,target,name,source) Values ");

		int count = 0;
		while ((str = reader.readLine()) != null)
		{
			String[] sp = str.split(",",2);

			//int target = Integer.valueOf(sp[0]);
			String target = sp[0];
			if (target.equals("negative"))  //target == 0
				sp[0] = "negative";
			else
				sp[0] = "positive";

			count++;
			sql.append("('")
					.append(sp[1].replaceAll("'","").replaceAll("\\\\",""))
					.append("'").append(",'").append(sp[0]).append("',")
					.append("'Instance").append(count).append("','Sentiment140'),\n");
		}
		System.out.println("Saving "+count+" Instances ..");
		if (count != 0)
		{
			sql.deleteCharAt(sql.length()-2).append(";");
			saveOutput(sql);
		}

		System.out.println(dbtable.substring(2)+" data saved.");
	}

    public void saveOutput(StringBuilder sql) throws IOException
	{
		try {
			dbiterator.INSERT(sql);
		} catch (SQLException e) {
			System.out.println("Error while inserting SQL check file : "+e.getMessage());
			FileWriter w = new FileWriter("sql_error.txt");
			w.write(sql.toString());
			w.close();
			System.exit(1);
		}
	}

	public static void repaircsv(String datasetpath) throws IOException
	{
		BufferedReader reader = new BufferedReader(new FileReader(datasetpath));
		StringBuilder data = new StringBuilder();
		String str;
		while ((str = reader.readLine()) != null)
		{
			data.append(str.replaceAll("\"",""));
			data.append("\n");
		}
		reader.close();
		FileWriter f = new FileWriter(datasetpath);
		f.write(data.toString());
		f.close();
	}

    //</editor-fold>

	//<editor-fold desc="Training & Saving Models">
	/**Training on the lexicon-based representation. Saves the model in order to use it on the provided test sets.
	 * The rest of the model representation forms will be created on-the-fly because of the
	 * minimum term frequency threshold that takes both train and test sets into consideration.*/
	private void trainLexicon() throws Exception
	{
		System.out.println("Starting reading Lexicon Training set ..");
		InstanceList lexicon_instance = dbiterator.getInstancesFromAlphabets(DBInstanceIterator.DBSource.LEXICON);
		System.out.println("Lexicon Training set : " + lexicon_instance.size());
		/*svm_parameter param = new svm_parameter();
		param.gamma = 0.0;
		param.C = 100.0;*/
		SVMClassifierTrainer trainer = new SVMClassifierTrainer(new LinearKernel(), true);
		System.out.println("SVM Starts Training on lexicon training set ..");
		SVMClassifier classifier = trainer.train(lexicon_instance);
		System.out.println("Saving SVM Classifier ..");
		System.out.println(lexicon_instance.targetLabelDistribution());
		saveModel(classifier, Lexicon_Train);
		System.out.println("Lexicon classifier saved to : " + Lexicon_Train);
	}

	/**Builds and saves the text-based model built on the training set.*/
	private void trainText() throws Exception
	{
		dbiterator.setDataSource(DBInstanceIterator.DBSource.TEXT);
		System.out.println("Starting reading Text Trainingset ..");

		Pipe serialpipe = getTextPipe();
		InstanceList text_instances = new InstanceList(serialpipe);

		System.out.println("Preprocessing on Text Trainingset ..");
		text_instances.addThruPipe(dbiterator);

		System.out.println("Text Training set : " + text_instances.size());
		System.out.println("Saving preprocessed data ..");
		saveinstances(text_instances, Train + "T.bin");

		System.out.println("Saving Features data ..");
		text_instances.getDataAlphabet().dump(new PrintWriter(new File(Attributes + "text.dat")));

		NaiveBayesTrainer trainer = new NaiveBayesTrainer();

		System.out.println("NaiveBayes Starts Training on Text training set ..");
		NaiveBayes classifier = trainer.train(text_instances);

		System.out.println("Targets labels : \t"+classifier.getLabelAlphabet());
		System.out.println("Saving Naive Bayes Classifier ..");
		saveModel(classifier, Text_Train);
		System.out.println("Text classifier saved to : " + Text_Train);
	}
	
	/**Builds and saves the feature-based model built on the training set.*/
	private void trainFeatures() throws Exception
	{
		dbiterator.setDataSource(DBInstanceIterator.DBSource.FEATURE);
		System.out.println("Starting reading Feature Trainingset ..");
		Pipe serialpipe = getFeaturePipe();
		InstanceList feature_instances = new InstanceList(serialpipe);
		System.out.println("Preprocessing on Feature Trainingset ..");
		feature_instances.addThruPipe(dbiterator);
		System.out.println("Feature Training set : " + feature_instances.size());
		System.out.println("Saving preprocessed data ..");
		saveinstances(feature_instances, Train + "F.bin");
		System.out.println("Saving Features data ..");
		feature_instances.getDataAlphabet().dump(new PrintWriter(new File(Attributes + "feature.dat")));
		ClassifierTrainer trainer = new NaiveBayesTrainer();
		System.out.println("NaiveBayes Starts Training on Feature training set ..");
		Classifier classifier = trainer.train(feature_instances);
		System.out.println("Targets labels : \t"+classifier.getLabelAlphabet());
		System.out.println("Saving Naive Bayes Classifier ..");
		saveModel(classifier, Feature_Train);
		System.out.println("Feature classifier saved to : " + Feature_Train);
	}
	
	/**Builds and saves the combined model built on the training set.*/
	private void trainCombined() throws Exception
	{
		dbiterator.setDataSource(DBInstanceIterator.DBSource.COMPLEX);
		System.out.println("Starting reading Complex Trainingset ..");
		Pipe serialpipe = getComplexPipe(); //as reference unigram best than unigram in POS
		InstanceList complex_instances = new InstanceList(serialpipe);
		System.out.println("Preprocessing on Complex Trainingset ..");
		complex_instances.addThruPipe(dbiterator);
		System.out.println("Complex Training set : " + complex_instances.size());
		System.out.println("Saving preprocessed data ..");
		saveinstances(complex_instances, Train + "C.bin");
		System.out.println("Saving Complex data ..");
		complex_instances.getDataAlphabet().dump(new PrintWriter(new File(Attributes + "complex.dat")));

		ClassifierTrainer trainer = new NaiveBayesTrainer();
		System.out.println("NaiveBayes Starts Training on Complex training set ..");
		Classifier classifier = trainer.train(complex_instances);
		System.out.println("Targets labels : \t"+classifier.getLabelAlphabet());
		System.out.println("Saving Naive Bayes Classifier ..");
		saveModel(classifier, Complex_Train);
		System.out.println("Complex classifier saved to : " + Complex_Train);
	}

   //</editor-fold>

	//<editor-fold desc="Preprocessing Pipe of Models">

	public static Pipe getTextPipe()
	{
		ArrayList<cc.mallet.pipe.Pipe> textfilter = new ArrayList<>();
		// Read data from File objects
		textfilter.add(new Input2CharSequence("UTF-8"));
		// Regular expression for what constitutes a token.
		// This pattern includes Unicode letters, Unicode numbers, and the underscore character. Alternatives:
		// "\\S+" (anything not whitespace)
		// "\\w+" ( A-Z, a-z, 0-9, _ )
		// "[\\p{L}\\p{N}_]+|[\\p{P}]+" (a group of only letters and numbers OR a group of only punctuation marks)
		Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");
		// Tokenize raw strings
		textfilter.add(new CharSequence2TokenSequence(tokenPattern));
		// Normalize all tokens to all lowercase
		textfilter.add(new TokenSequenceLowercase());
		// Remove stopwords from a standard English stoplist.
		// options: [case sensitive] [mark deletions]
		textfilter.add(new TokenSequenceRemoveStopwords(new File(EN_stopwords),"UTF-8",
                false,false,false));
		// This pipe removes all the non Alphabet words from the Token Sequence.
		// For eg if the Token Sequence contains tokens like abc123,then these words are removed from the Token Sequence.
		textfilter.add(new TokenSequenceRemoveNonAlpha());
		//applying stemming
		textfilter.add(new TokenSequence2PorterStems());
		// converts the token sequence in the data fields to the token sequence of N grams.
		// Note: We also used TokenSequenceRemoveStopwords and TokenSequenceRemoveNonAlpha before using this pipe.
		// Therefore tokens don’t contain stopwords and non alphabetic words .
		textfilter.add(new TokenSequenceNGrams(new int[]{2}));
		// Rather than storing tokens as strings, convert them to integers by looking them up in an alphabet.
		textfilter.add(new TokenSequence2FeatureSequence());
		// Now convert the sequence of features to a sparse vector, mapping feature IDs to counts.
		textfilter.add(new FeatureSequence2FeatureVector());
		// Do the same thing for the "target" field:
		// convert a class label string to a Label object, which has an index in a Label alphabet.
		textfilter.add(new Target2Label());
		// return pipe of type : serial
		return new SerialPipes(textfilter);
	}
	public static Pipe getFeaturePipe()
	{
		ArrayList<cc.mallet.pipe.Pipe> textfilter = new ArrayList<>();
		// Read data from File objects
		textfilter.add(new Input2CharSequence("UTF-8"));
		// Regular expression for what constitutes a token.
		// This pattern includes Unicode letters, Unicode numbers, and the underscore character. Alternatives:
		// "\\S+" (anything not whitespace)
		// "\\w+" ( A-Z, a-z, 0-9, _ )
		// "[\\p{L}\\p{N}_]+|[\\p{P}]+" (a group of only letters and numbers OR a group of only punctuation marks)
		Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");
		// Tokenize raw strings
		textfilter.add(new CharSequence2TokenSequence(tokenPattern));
		// Normalize all tokens to all lowercase
		textfilter.add(new TokenSequenceLowercase());
		// Remove stopwords from a standard English stoplist.
		// options: [case sensitive] [mark deletions]
		textfilter.add(new TokenSequenceRemoveStopwords(new File(EN_stopwords),"UTF-8",false,false,false));
		//This pipe removes all the non Alphabet words from the Token Sequence.
		// For eg if the Token Sequence contains tokens like abc123,then these words are removed from the Token Sequence.
		textfilter.add(new TokenSequenceRemoveNonAlpha());
		//applying stemming
		textfilter.add(new TokenSequence2PorterStems());
		// converts the token sequence in the data fields to the token sequence of N grams.
		// Note: We also used TokenSequenceRemoveStopwords and TokenSequenceRemoveNonAlpha before using this pipe.
		// Therefore tokens don’t contain stopwords and non alphabetic words .
		//textfilter.add(new TokenSequenceNGrams(new int[]{1}));
		// Rather than storing tokens as strings, convert
		// them to integers by looking them up in an alphabet.
		textfilter.add(new TokenSequence2FeatureSequence());
		// Now convert the sequence of features to a sparse vector,
		// mapping feature IDs to counts.
		textfilter.add(new FeatureSequence2FeatureVector());
		// Do the same thing for the "target" field:
		// convert a class label string to a Label object,
		// which has an index in a Label alphabet.
		textfilter.add(new Target2Label());

		return new SerialPipes(textfilter);
	}
	public static Pipe getComplexPipe()
	{
		ArrayList<cc.mallet.pipe.Pipe> textfilter = new ArrayList<>();
		// Read data from File objects
		textfilter.add(new Input2CharSequence("UTF-8"));
		// Regular expression for what constitutes a token.
		// This pattern includes Unicode letters, Unicode numbers, and the underscore character. Alternatives:
		// "\\S+" (anything not whitespace)
		// "\\w+" ( A-Z, a-z, 0-9, _ )
		// "[\\p{L}\\p{N}_]+|[\\p{P}]+" (a group of only letters and numbers OR a group of only punctuation marks)
		Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");
		// Tokenize raw strings
		textfilter.add(new CharSequence2TokenSequence(tokenPattern));
		// Normalize all tokens to all lowercase
		textfilter.add(new TokenSequenceLowercase());
		//applying stemming
		textfilter.add(new TokenSequence2PorterStems());
		// converts the token sequence in the data fields to the token sequence of N grams.
		// Note: We also used TokenSequenceRemoveStopwords and TokenSequenceRemoveNonAlpha before using this pipe.
		// Therefore tokens don’t contain stopwords and non alphabetic words .
		textfilter.add(new TokenSequenceNGrams(new int[]{2}));
		// Rather than storing tokens as strings, convert
		// them to integers by looking them up in an alphabet.
		textfilter.add(new TokenSequence2FeatureSequence());
		// Now convert the sequence of features to a sparse vector,
		// mapping feature IDs to counts.
		textfilter.add(new FeatureSequence2FeatureVector());
		// Do the same thing for the "target" field:
		// convert a class label string to a Label object,
		// which has an index in a Label alphabet.
		textfilter.add(new Target2Label());

		return new SerialPipes(textfilter);
	}
    //</editor-fold>

    //<editor-fold desc="Read Attributes files and saved to BidiMaps used by PolarityClassifier beside saved Models">
/*	public BidiMap<String, Integer> getTextAttributes()
	{
		try {
			tba = read_attribute(Attributes+"text.dat");
		} catch (IOException e) {
			e.printStackTrace();
		}
		return tba;
	}
	public BidiMap<String, Integer> getFeatureAttributes()
	{
		try {
			fba = read_attribute(Attributes+"feature.dat");
		} catch (IOException e) {
			e.printStackTrace();
		}
		return fba;
	}
	public BidiMap<String, Integer> getComplexAttributes()
	{
		try {
			cba = read_attribute(Attributes+"complex.dat");
		} catch (IOException e) {
			e.printStackTrace();
		}
		return cba;
	}*/
	//</editor-fold>

    //<editor-fold desc="Save & Load Models,Instances,Attributes ">
	private void saveinstances(InstanceList instances, String filename)
	{
		instances.save(new File(filename));
	}
	public static InstanceList loadinstances(String filename)
	{
		return InstanceList.load(new File(filename));
	}

	public static void saveModel(Classifier model,String filename) throws IOException
	{
		ObjectOutputStream obj = new ObjectOutputStream(new FileOutputStream(new File(filename)));
		obj.writeObject(model);
		obj.close();
	}
	public static Classifier loadModel(String filename) throws IOException, ClassNotFoundException
	{
		Classifier classifier;
		ObjectInputStream obj = new ObjectInputStream(new FileInputStream(new File(filename)));
		classifier = (Classifier) obj.readObject();
		obj.close();

		return classifier;
	}

	public static BidiMap<String,Integer> read_attribute(String filename) throws IOException
	{
		BidiMap<String,Integer> attributes = new DualHashBidiMap<>();
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		String str;
		while ((str = reader.readLine()) != null)
		{
			String[] indices = str.split("=>");
			if (indices.length > 1){
				int index = Integer.valueOf(indices[0].trim());
				attributes.put(indices[1].trim(),index);
			}
		}
		reader.close();

		return attributes;
	}
	public static BidiMap<String,Integer> read_attribute(Alphabet dataalphabet)
	{
		BidiMap<String,Integer> attributes = new DualHashBidiMap<>();
        for (int i=0; i<dataalphabet.size(); i++)
			attributes.put(dataalphabet.lookupObject(i).toString(),i);

		return attributes;
	}

	/**Re-order the attributes of the given Instances according to the training file */
	public static Instance reformatfeaturevector(Instance tweet, Classifier classifier)
	{
		Instance compatible_test_instance;

		//old
		FeatureVector fv = (FeatureVector) tweet.getData();
		int[] oldindices = fv.getIndices();
		double[] oldvalues = fv.getValues();

        //new
		ArrayList<Integer> newindices = new ArrayList<>();
		ArrayList<Double> newvalues = new ArrayList<>();

		//Compatible data alphabet
		for (int i=0; i<oldindices.length; i++)
		{
			String attribute = fv.getAlphabet().lookupObject(oldindices[i]).toString();

			if (!classifier.getAlphabet().contains(attribute))
				System.out.println("Incompatible Attribute : "+attribute);
			else
			{
				//read corresponding index in this classifier dataalphabet
				int index = classifier.getAlphabet().lookupIndex(attribute,false);
				System.out.println("Compatible Attribute : "+attribute+" at index : "+index);
				newindices.add(index);   //correct index of this attribute according to test trainingset
				newvalues.add(oldvalues[i]);     //value
			}
		}

		//Compatible target alphabet
        Label target = classifier.getLabelAlphabet().lookupLabel(tweet.getTarget().toString(),false);

		//Creating Compatible Instance
		fv = new FeatureVector(classifier.getAlphabet(),
				newindices.stream().mapToInt(i -> i).toArray(), newvalues.stream().mapToDouble(i -> i).toArray());

		compatible_test_instance = new Instance(fv,target,tweet.getName(),tweet.getSource());

		return compatible_test_instance;
	}
	//</editor-fold>
}