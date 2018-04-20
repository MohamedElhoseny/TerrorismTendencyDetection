package Models.Word_Weighting;

import Models.Sentiment_Analysis.Methodology.SentimentAnalyser;
import Models.Sentiment_Analysis.Preprocessing.TextPreprocessor;
import Models.Sentiment_Analysis.Preprocessing.TweetPreprocessor;
import Models.Sentiment_Analysis.Utils.TokenSequence2PorterStems;
import cc.mallet.pipe.*;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import cc.mallet.util.FileUtils;
import jsc.util.Logarithm;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.text.DecimalFormat;
import java.util.*;

public class WordWeightingModel
{
	static final String outputdirectory = "Resources/datasets/wordweighting/out/";

	HashMap<Integer, LinkedHashMap<String, Integer>> hash_tf = new LinkedHashMap<>();
	HashMap<Integer, LinkedHashMap<String, Double>> hash_tfidf = new LinkedHashMap<>();
	HashMap<String, Integer> hash_df = new LinkedHashMap<>();
    HashMap<String, Double> highest_tfidf = new HashMap<>();
	HashMap<String, Double> word_score = new HashMap<>();
	HashMap<String, Double> normalized_score = new HashMap<>();

    String _corpus; //path to orignal corpus (for overwritting upon needed)
	String _outputcorpus;  //path to model corpus which applying operation on it (safe original)
    boolean isready;  //if data in corpus is preprocessed, so its ready
    boolean isevaluate; //used to check when inserting word as insertion need model to evaluate first
	double highestscore = 0.0;

	/** Construct Model with given parameters
	 * @param corpus is corpus file path
	 * @param isready set to true if corpus file data is preprocessed before
	 */
	public WordWeightingModel(String corpus, boolean isready) throws IOException
	{
		this.isready = isready;
		this.isevaluate = false;
		this._corpus = corpus;
		this._outputcorpus = outputdirectory+"corpus.txt";

		Files.copy(new File(_corpus).toPath(),new FileOutputStream(_outputcorpus)); //copy
	}

	//<editor-fold desc="Preprocessing" default-state="Capsulated">

	/** Just Preprocess corpus : remove stopwords, nonAlphabetic and applying Porter stemming
	 * @throws Exception if corpus data is already preprocessed
	 */
	public void PreprocessCorpus() throws Exception
	{
		if (!isready)
		{
			System.out.println("Preprocessing WordWeighting Corpus ..");
			TextPreprocessor preprocessor = new TextPreprocessor("resources/datasets/sentimentanalysis/");
			InstanceList instances = new InstanceList(getTextRepresentationPipe());

			String[] lines = FileUtils.readFile(new File(_outputcorpus));
			for (String line : lines)
				instances.addThruPipe(new Instance(preprocessor.getProcessed(line), "", null, null));


			BufferedWriter writer =
					new BufferedWriter(new FileWriter(outputdirectory + "trainingcorpus.txt"));
			TokenSequence tokens;
			StringBuilder builder;

			//write preprocessed instances to file (for saving and fast training)
			for (Instance in : instances)
			{
				tokens = (TokenSequence) in.getData();
				builder = new StringBuilder();

				for (Token token : tokens)
					builder.append(token.getText()).append(" ");

				if (!builder.toString().isEmpty()) {
					writer.write(builder.toString().trim());
					writer.newLine();
				}
			}

			writer.close();
			//update state
			isready = true;

		}else
			throw new Exception("Corpus is already preprocessed !");
	}

    public static Pipe getTextRepresentationPipe()
	{
		ArrayList<Pipe> pipes = new ArrayList<>();
		pipes.add(new Input2CharSequence("UTF-8"));
		pipes.add(new CharSequence2TokenSequence());
		pipes.add(new TokenSequenceLowercase());
		pipes.add(new TokenSequenceRemoveStopwords(new File("Resources/stoplists/en.txt"),
				"UTF-8",false,false,false));
		pipes.add(new TokenSequenceRemoveNonAlpha());
		pipes.add(new TokenSequence2PorterStems());
		return new SerialPipes(pipes);
	}
	//</editor-fold>

	//<editor-fold desc="TF-IDF" default-state="Capsulated">
	/**
	 * Calculating tf , df, idf then tf-idf for each word in document, Score for each word and Normalize Scores
	 */
	public void Evaluate() throws IOException
	{
		if (isready)
		{
			System.out.println("WordWeighting training .%");
			int numLine = 0;
			Logarithm logarithm = new Logarithm(10);
			String[] lines = FileUtils.readFile(new File(outputdirectory+"trainingcorpus.txt"));
			for (String line: lines)
			{
				String[] tokens = line.split(" ");
				LinkedHashMap<String, Integer> newTFHash = new LinkedHashMap<>(); //Hold term frequency for each word
				LinkedHashMap<String, Integer> newDFHash = new LinkedHashMap<>(); //Hold document frequency for each word

				for (String term : tokens) {
					// TF (counting word for each document)
					if (newTFHash.containsKey(term)) //if given word is found in TF collection increment its value
						newTFHash.replace(term, newTFHash.get(term) + 1);
					else newTFHash.put(term, 1);  //init TF for new word

					// DF
					if (!newDFHash.containsKey(term)) //why check ? as if word repeated in document
						newDFHash.put(term, 1);
				}
				// DF (Counting doc for each word
				for (String key : newDFHash.keySet()) {
					if (hash_df.containsKey(key)) //if hash_df contain word increment its document frequency
						hash_df.replace(key, hash_df.get(key) + 1);
					else
						hash_df.put(key, 1); //init DF for new word
				}
				hash_tf.put(numLine, newTFHash); //finally add doc num w hashmap for each word and its doc freq
				newDFHash.clear();

				numLine++;  //increment doc number
				if (numLine % 499 == 0)
					System.out.print(".");
			}
			/** calculate tf-idf for each word in document*/
			System.out.println("Calculating tf-idf ..");
			for (Integer doc : hash_tf.keySet()) //loop for each document in corpus
			{
				LinkedHashMap<String, Integer> hashmap = hash_tf.get(doc); //read its terms with their tf
				//calculate tf-idf for each term in this document
				LinkedHashMap<String, Double> tf_idf = new LinkedHashMap<>();
				for (String term : hashmap.keySet()) {
					double tf = 1 + logarithm.log(hashmap.get(term));    //tf = 1 + log tf
					double idf = logarithm.log(numLine / hash_df.get(term)); //idf = log(N/df(t))
					tf_idf.put(term, tf * idf);
				}
				hash_tfidf.put(doc, tf_idf);
			}
             System.out.println("Getting TF-IDF Score ..");
			_GetTFIDFScore();
			System.out.println("Getting Word Score ..");
			_GetWordScore();
			isevaluate = true;

		}else
			System.out.println("Failed to Evaluate, corpus must be preprocessing first !");
	}

	private void _GetTFIDFScore()
	{
		for (String word: hash_df.keySet()) //for each word found in corpus
		{
			double highest_tfidff = 0.0;

			for (Integer doc: hash_tfidf.keySet()) //for each doc in corpus search about word and check its tf-idf
			{
				try {
					double word_tfidf = hash_tfidf.get(doc).get(word);
					if (highest_tfidff < word_tfidf)
						highest_tfidff = word_tfidf;

				}catch (Exception ignored){
					//if word not present in this doc Exception will fired
				}
			}

			highest_tfidf.put(word,highest_tfidff);
		}
	}

	/**
	 * Getting word score based on the word that appear in the most of corpus topics
	 */
	private void _GetWordScore()
	{
		double df;
		double wordscore;
		Logarithm logarithm = new Logarithm(10);

		//Get df for all words
		for (String word: hash_df.keySet())
		{
			df = hash_df.get(word);
			wordscore = 1 + logarithm.log(df);

			word_score.put(word,wordscore);
			if (wordscore > highestscore)
				highestscore = wordscore;
		}

		//Normalize Score
		for (String word: word_score.keySet())
			normalized_score.put(word, word_score.get(word) / highestscore);
	}
    //</editor-fold>

	//<editor-fold desc="Extend & Load dictionary" default-state="Capsulated">
	public static Map<String,Double> getDictionary()
	{
		Map<String , Double> dictionary = new HashMap<>();
		try {
			CSVParser parser = new CSVParser(new FileReader(outputdirectory+"Words_scores.csv"),
                    CSVFormat.EXCEL.withFirstRecordAsHeader());

			for (CSVRecord record: parser.getRecords())
			{
				dictionary.put(record.get(Words.name), Double.valueOf(record.get(Words.normalize_weight)));
			}


		} catch (IOException e) {
			e.printStackTrace();
		}
		return dictionary;
	}

	public static void extendCorpus(String corpus,String newCorpuspath)
	{
		BufferedWriter writer = null;
		BufferedReader reader = null;

		try
		{
			writer = new BufferedWriter(new FileWriter(corpus,true));
			reader = new BufferedReader(new FileReader(newCorpuspath));
            String line;

			while ((line = reader.readLine()) != null)
			{
				if (!line.isEmpty())
				{
					writer.write(line);
					writer.newLine();
				}
			}
			writer.close();
			reader.close();
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	/** Used to update dictionary according given vocabulary file, only the words in the vocabulary file will remain
	 * @param vocabfile vocabulary file containing words required only to be in our model
	 * @throws IOException
	 */
	public static void updateDictionary(String vocabfile) throws IOException
	{
        Map<String,Double> old_dict = getDictionary();
        String[] lines = FileUtils.readFile(new File(vocabfile));

        Map<String,Double> new_dict = new HashMap<>();

		for (int i = 1; i < lines.length; i++)
		{
			if (old_dict.containsKey(lines[i]))
				new_dict.put(lines[i],old_dict.get(lines[i]));
		}

		saveScores(sortByValue(new_dict));
	}
	//</editor-fold>

	//<editor-fold desc="Insert/Modify Score for specific word">
	public double getAppropriateDF(Freq freqlevel)
    {
        // 1 + log10 (df) = word_weight
        Logarithm logarithm = new Logarithm(10);
        double invWeight = 0.0;

        switch (freqlevel)
        {
            case High: //word_weight = 0.9
                invWeight = 0.9 * highestscore;
                break;
			case Medium:
				invWeight = 0.6 * highestscore;
				break;
			case Low:
				invWeight = 0.3 * highestscore;
				break;
        }

        return Math.round(logarithm.antilog(invWeight - 1));
    }

	/**
	 *  1) invWeight / highest_weight = 0.9     .:. invWeight is the word weight that give us 0.9 for word score
	 *  using 1 + log10(df) = word weight
	 *  2) df = invLog ( invWeight - 1 )
	 */
    public void insertWordtoCorpus(String word, Freq freqlevel, boolean re_evaluate, boolean overwrite) throws Exception
	{
		if (isevaluate)
		{
			int df = (int) getAppropriateDF(freqlevel); //this is df needed for this word to be at appropiate frequency

            if (hash_df.containsKey(word))
            	injectModel(word,hash_df.get(word),df,re_evaluate,overwrite);
            else
            	injectModel(word,0,df,re_evaluate,overwrite);

		}else
			throw new Exception("Model must be evaluated first before modifying scores of words");
	}

	private void injectModel(String word, int old_df, int new_df, boolean reevaluate, boolean overwrite) throws Exception
	{
        System.out.println("word = "+word+" old = "+old_df+" new = "+new_df);
        String[] lines = FileUtils.readFile(new File(_outputcorpus));
        BufferedWriter writer = new BufferedWriter(new FileWriter(_outputcorpus,false));
        int n;

        if (new_df > old_df)
		{
			n = new_df - old_df;   //number of topic that needed to insert in this corpus contining this word

			for (String line: lines)
				writer.write(line+"\n");

			for (int i = 0; i < n; i++)
				writer.write(word+"\n");
		}
		else
		{
        	n = old_df - new_df;
			for (String line : lines)
			{
				if (n == 0) break;

				if (line.contains(word)) {
					line.replace(word, "");
					n--;
				}
			}

			for (String line: lines)
				writer.write(line+"\n");
		}
		writer.close();
        isready = false;

        if (overwrite)
        	Files.copy(new File(_outputcorpus).toPath(),new File(_corpus).toPath(), StandardCopyOption.REPLACE_EXISTING);

		if (reevaluate)
		{
			PreprocessCorpus();
			Evaluate();
		}

	}

    //</editor-fold>

	//<editor-fold desc="remove noise word">

	/** Removing any noise words that not need it
	 * @param word  word that wants to remove
	 * @param overwrite if true orignal corpus file will updated, otherwise another temp file for corpus is created
	 * @throws IOException
	 */
	public void removeWordFromCorpus(String word,boolean reevaluate, boolean overwrite) throws Exception
	{
		String[] lines = FileUtils.readFile(new File(_outputcorpus));
		BufferedWriter writer = new BufferedWriter(new FileWriter(_outputcorpus,false));

		//reformat output corpus after update
		for (String line: lines){
			writer.write(line.replaceAll(word,""));
			writer.newLine();
		}
		writer.close();
		isready = false;

		if (overwrite)
			Files.copy(new File(_outputcorpus).toPath(),new File(_corpus).toPath(), StandardCopyOption.REPLACE_EXISTING);

		if (reevaluate){
			PreprocessCorpus();
			Evaluate();
		}
	}

	//</editor-fold>

	//<editor-fold desc="Saving Outputs of models">
    /**
     *  Export/Save data file
     */
    public void SaveOutputDetails()
    {
        try
        {
            BufferedWriter bWriter = new BufferedWriter(new FileWriter(outputdirectory+"tf_idf.txt"));

            for (Integer doc : hash_tfidf.keySet()) //loop for each Documents (specified with tf or tfidf)
            {
                HashMap<String, Integer> Docs = hash_tf.get(doc); //read each Document[Terms,Tf-idf] in corpus
                bWriter.write("doc "+doc+": ");
                //for each term in document print word:Weight
                for (String term : Docs.keySet())
                {
                    //iphone:2.557
                    bWriter.write(term + ": [Tf = "+ new DecimalFormat("##.###").format(Docs.get(term))
                            +", dF = " + new DecimalFormat("##.###").format(hash_df.get(term))
                            +", Tf-idf = " + new DecimalFormat("##.###").format(hash_tfidf.get(doc).get(term)) + "] ");
                }
                bWriter.write("\n");
            }
            bWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void SaveWordScores()
	{
    	System.out.println("Saving WordWeighting work ..");
        Map<String,Double> sortednormalizedscore = sortByValue(normalized_score);
        //Map<String, Double> sortedscore = new TreeMap<>(highest_tfidf); sort by alphabet

		saveScores(sortednormalizedscore);
        saveVocabulary(sortednormalizedscore);
    }

	private void saveVocabulary(Map<String,Double> sortedasc)
	{
		//Map<String,Double> sortedasc = sortByValue(normalized_score);
		try
		{
			FileWriter writer = new FileWriter(WordWeightingModel.outputdirectory+"vocabulary.txt");

			writer.write("Corpus : "+sortedasc.size()+" Vocabulary - sorted by : highest score\n");
			for (String s: sortedasc.keySet()) {
				writer.write(s+"\n");
			}

			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	private static void saveScores(Map<String,Double> dic)
	{
		try {
			FileWriter out = new FileWriter(outputdirectory + "Words_scores.csv");
			CSVPrinter csvwriter = new CSVPrinter(out, CSVFormat.EXCEL.withHeader(Words.class));
			for (String word : dic.keySet())
				csvwriter.printRecord(word, dic.get(word));
			csvwriter.close();
		}catch (IOException e)
		{
			e.printStackTrace();
		}
	}
    /** Sorting words by highest weight */
    public static Map<String, Double> sortByValue(Map<String, Double> unsortedMap)
    {
        //Map ---> List<Map> ---> Collections.sort() --> List<Map> (Sorted) ---> LinkedHashMap

        // 1. Convert Map to List of Map
        List<Map.Entry<String, Double>> list = new LinkedList<>(unsortedMap.entrySet());

        // 2. Sort list with Collections.sort(), provide a custom Comparator
        //    Try switch the o1 o2 position for a different order
        list.sort((o1, o2) -> (o2.getValue()).compareTo(o1.getValue()));

        // 3. Loop the sorted list and put it into a new insertion order Map LinkedHashMap
        Map<String, Double> sortedMap = new LinkedHashMap<>();
        for (Map.Entry<String, Double> entry : list) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        return sortedMap;
    }
    //</editor-fold>

	public enum Words
	{
    	name,normalize_weight
	}
	public enum Freq
	{
		High,Medium,Low
	}
}
