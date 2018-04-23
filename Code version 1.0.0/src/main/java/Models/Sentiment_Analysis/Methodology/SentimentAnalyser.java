package Models.Sentiment_Analysis.Methodology;
import Models.Evaluation.Factory;
import Models.Evaluation.Visulatizer;
import Models.Sentiment_Analysis.Preprocessing.TweetPreprocessor;
import cc.mallet.classify.NaiveBayes;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.Noop;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelVector;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class SentimentAnalyser implements Factory,Serializable
{
	//the path to the "resources" folder
	public static String directory = "Resources/datasets/sentimentanalysis/";
	//if set to "true", if Model fail to classify tweet, test dataset Model will try to clarify
	public static boolean usetestModel = true;


	PolarityClassifier pc;
	TweetPreprocessor tp;

	boolean useTestModel;
	NaiveBayes multiNB;
	InstanceList test;
	//BidiMap<String, Integer> train_attributes;
	//InstanceList training_text;


	public SentimentAnalyser() {
		this(directory,usetestModel);
	}

	/** Construct PolarityClassifier with saved Attribute and Models, TweetPreprocessor for input tweets
	 *  and load test Model if useSM is True
	 * @param main_folder directory of datasets files
	 * @param useSM True if you want to clarify on saved Test Models if Algorithms disagree to classify
	 * @throws Exception
	 */
	public SentimentAnalyser(String main_folder, boolean useSM)
	{
		long now = System.currentTimeMillis();
		Trainer tr = new Trainer();

		//pc = new PolarityClassifier(main_folder, tr.getTextAttributes(), tr.getFeatureAttributes(), tr.getComplexAttributes());
		System.out.println("Loading saved Classifiers by Trainer ...");
		pc = new PolarityClassifier(main_folder);
		System.out.println("Initialize datasets for Tweet Preprocessing ...");
		tp = new TweetPreprocessor(main_folder);

		useTestModel = useSM;
		if (useTestModel)
		    initClarifyModel();

        long end = (System.currentTimeMillis() - now);
		System.out.println("Sentiment analysis constructed in "+end+" ms.");
	}

	/** 1) preprocesses the given tweet, creates different representations of it (stored in "all[]" Instances)
	 *  2) tests it in the PolarityClassifier class.
	 *  3) determine if result want to clarify based on saved test Model or not
	 *  @param tweet input tweet to classify
	 *  @return object[2] where object[0] is polarity string , object[1] is polarity score
	 */
	public Object[] getPolarity(Instance tweet)
	{
		System.out.println("Preprocessing tweet : "+tweet.getData());
		Instance[] all = tp.startProc(tweet); //Create text,feature,lexicon,complex instances of this tweet

		System.out.println("Applying Methodology ..");
		Object[] out = pc.apply(all);

		System.out.println("Methodology's result agreed on :- \n Polarity = "+out[0].toString() +"\n Polarity Score = "+out[1]);
		System.out.println("_______________________________________________________________");

		if (useTestModel)
		{
			if (!out[0].equals("nan"))
				return out;
			else
				out = clarifyOnModel(tweet);  //if HC and LC disagree to predict

		}
		return out;
	}

	/** Used to loading Clarify Model to use it for clarifying or evaluating */
	private void initClarifyModel()
	{
		//ExecutorService threadpool = Executors.newFixedThreadPool(8);
		System.out.println("Starting Loading test dataset Model ...");
		//threadpool.execute(() -> {
		try {
			multiNB = (NaiveBayes) Trainer.loadModel(directory+"mallet_test/Model.bin");
			multiNB.getAlphabet().stopGrowth();
			multiNB.getLabelAlphabet().stopGrowth();
			//train_attributes = Trainer.read_attribute(multiNB.getAlphabet());
			//training_text = InstanceList.load(new File(directory + "mallet_test/Model-Instances.bin"));
			//test.setPipe(training_text.getPipe());
			test = new InstanceList(Trainer.getTextPipe());

		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}
		System.out.println("Model loaded .");
		//});
		//for Memory Improvement
      	  /*threadpool.execute(() -> {
				//read attributes file of this test data [save to train_attributs bidimap]
				try {
					train_attributes = Trainer.read_attribute(multiNB.getAlphabet().);
				} catch (IOException e) {
					e.printStackTrace();
				}
				System.out.println("Attributes loaded .");
			});*/
          /*  threadpool.execute(() -> {
				//read training text and save as instances to training_text
				training_text = InstanceList.load(new File(main_folder+"mallet_test/Model-Instances.bin"));
				System.out.println("trainingInstances loaded .");
			});*/

		//threadpool.shutdown();
		//while (!threadpool.isTerminated());

		System.out.println("Clarify Model loaded 100%");
	}

	/** Decides upon a "disagreed" document by applying the learned model based on the previously build model.*/
	private Object[] clarifyOnModel(Instance tweet)
	{
		System.out.println("Starting Clarifying On Previous Model of Test dataset ..");

		//synchronized (this)
		//{
			Instance instance = new Instance(tweet.getData(),tweet.getTarget(),tweet.getName(),tweet.getSource());  //preprocess
			test.addThruPipe(instance);
			// re-order attributes so that they are compatible with the training set's ones
			instance = Trainer.reformatfeaturevector(test.get(0), multiNB);    //reformat
		    test.remove(0);
		//}
		System.out.println("Starting Classifying tweet using Previous Model ..");
		LabelVector vector = multiNB.classify(instance).getLabelVector();
        double pos = vector.value(multiNB.getLabelAlphabet().lookupLabel("positive"));
		double neg = vector.value(multiNB.getLabelAlphabet().lookupLabel("negative"));
		System.out.println("_______________________________________________________________");
		System.out.println("Results from previous trained Model : \n " +
				"Positive = "+pos+"\n" +
				" Negative = "+neg);

		Object[] out = new Object[2];
		if (pos > 0.5)
			out[0] = "light positive";
		else
			out[0] = "light negative";

		out[1] = neg;  //return negative not best


		return out;
	}

	//<editor-fold desc="Evaluation & Visualization Functions" default-state="Capsulated">
	@Override
	public double evaluate(Instance tweet)
	{
        Object[] polarity = getPolarity(tweet);
        System.out.println("Polarity = "+polarity[0]+"\nNegative score : "+polarity[1]);
		return (double) polarity[1];
	}

	public Trial[] evaluateClassifiers(InstanceList testinstances,boolean withClarifymodel)
	{
		//first preprocess and pass for Trail after adding to InstanceList with text pipe
		Instance[] textInstances = new Instance[testinstances.size()];
		Instance[] featureInstances = new Instance[testinstances.size()];
		Instance[] complexInstances = new Instance[testinstances.size()];
		Instance[] lexiconInstances = new Instance[testinstances.size()];

		System.out.println("applying TweetPreprocessing on Test instances ..");
		int c = 0;
		for (Instance tweet: testinstances)
		{
			Instance[] preparetweets = tp.startProc(tweet); //apply only tweet preprocessing on Instance Data
			textInstances[c]    = preparetweets[0];
			featureInstances[c] = preparetweets[1];
			complexInstances[c] = preparetweets[2];
			lexiconInstances[c] = preparetweets[3];
			if (++c%100==0)
				System.out.print(".");
		}
		//Create 5 trials for classifiers
		Trial[] classifierResults = new Trial[5];

		if (withClarifymodel)
		{
			if (!useTestModel)
				initClarifyModel();
			Trial cnb = this.evaluateClarifiedModel(textInstances);

			if (cnb !=null) {
				PolarityClassifier.saveClassifierTrial("ClarifyNB", cnb);
				classifierResults[4] = cnb;
			}
		}

		System.out.println("Starting Evaluating Classifiers .%");
		Trial[] hc_lc = pc.EvaluateClassifiers(textInstances,featureInstances,complexInstances,lexiconInstances);
		for (int i = 0; i < hc_lc.length; i++)
			classifierResults[i] = hc_lc[i];

        return classifierResults;
	}

	/** This method used to predict the testset and calculate accuracy to all parts of methodology
	 * @param testinstances the test dataset
	 * @return Hashmap<String,Double> continue percentage for methodology accuracy
	 */
	public Map<String,Double> evaluateMethodology(InstanceList testinstances)
	{
		pc.resetHC_LC();

		double n_testtweets = testinstances.size(); //hold number of tested tweets
		double n_agreed = 0;           //hold number of tweet that our Model agreed on their results
		double n_disagreed = 0;		  //hold number of tweet that our Model disagreed on their results
		double n_TPmodel = 0;        //hold number of true predicted tweets by Model [True positive, True negative]
		double n_TPclarifymodel = 0;//hold number of true predicted tweets by Clarify Model

		System.out.println("Starting Evaluating Methodology .%");
		for (Instance tweet: testinstances)
		{
			Object[] results = getPolarity(tweet);

			String result = results[0].toString();

			if (result.contains("light"))
			{
                 n_disagreed++;
                 if(result.contains(tweet.getTarget().toString()))
                 	n_TPclarifymodel++;
			}else
			{
				n_agreed++;
				if (result.equals(tweet.getTarget().toString()))
					n_TPmodel++;
			}
		}

		int[] t_hclc = pc.getTrueHC_LC();
		System.out.println("hc = "+t_hclc[0]);
		System.out.println("lc = "+t_hclc[1]);

		Map<String,Double> accuracy = new HashMap<>();
		accuracy.put("hc",t_hclc[0]/n_testtweets);
		accuracy.put("lc",t_hclc[1]/n_testtweets);
		accuracy.put("agree", n_agreed/n_testtweets);
		accuracy.put("disagree", n_disagreed/n_testtweets);
		accuracy.put("hc_lc", n_TPmodel/n_agreed);
		accuracy.put("clarifymodel", n_TPclarifymodel/n_disagreed);
		accuracy.put("methodology", (n_TPmodel+n_TPclarifymodel)/n_testtweets);

		saveAccuracy(accuracy);

		return accuracy;
	}

	public void saveAccuracy(Map<String,Double> results)
	{
		try {
			CSVPrinter writer = new CSVPrinter(new FileWriter(directory+"Results/result.csv"),
					CSVFormat.EXCEL.withHeader("hc","lc","agree","disagree","hc_lc","clarifymodel","methodology"));

			writer.printRecord(results.get("hc"),results.get("lc"),results.get("agree"),results.get("disagree"),
					results.get("hc_lc"),results.get("clarifymodel"),results.get("methodology"));


			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private Trial evaluateClarifiedModel(Instance[] text)
	{
		if (multiNB != null)
		{
			//preprocess the prepared tweets through the same pipe of clarify model [TextPipe]
			for (Instance i: text)
				test.addThruPipe(new Instance(i.getData(),i.getTarget(),i.getName(),i.getSource()));

			// re-order attributes so that they are compatible with the training set's ones
			InstanceList trace = new InstanceList(new Noop());
			for (Instance i: test)
				trace.add(Trainer.reformatfeaturevector(i, multiNB));


			System.out.println("Evaluating ClarifyModel Classifier on Testset .. ");
			return new Trial(multiNB,trace);
		}
		return null;
	}
	//</editor-fold>
}