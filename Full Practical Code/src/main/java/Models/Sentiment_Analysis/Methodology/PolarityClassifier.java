package Models.Sentiment_Analysis.Methodology;

import Models.Evaluation.Visulatizer;
import Models.Evaluation.visualization.malllet.ConfusionMatrix;
import Models.Sentiment_Analysis.Preprocessing.LexiconPreprocessor;
import Models.Sentiment_Analysis.Preprocessing.TweetPreprocessor;
import Models.Sentiment_Analysis.Utils.SVMClassifier;
import cc.mallet.classify.Classification;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.Noop;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelVector;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.*;
import java.util.Arrays;

public class PolarityClassifier implements Serializable
{
	static String folder;

	//produced by Trainer
	/*BidiMap<String, Integer> tba;
	BidiMap<String, Integer> fba;
	BidiMap<String, Integer> cba;*/

	//Create nb, svm classifiers to load saved Model created by Trainer
	Classifier[] mnb_classifiers;
	SVMClassifier lexicon_classifier;

	//used for the processed tweet
	/*Instance text = null;
	Instance feature = null;
	Instance complex = null;
	Instance lexicon_instances = null;*/
	Instance[] preprocessed_tweet;

	//T,F,C after preprocessed while training [saved instancelist]  (hst5dmhom 3shan a5le tweet t3de bnfs pipe bt3thom)
	InstanceList training_text;
	InstanceList training_feature;
	InstanceList training_complex;

    int t_hc = 0;
    int t_lc = 0;


	/** Initialize PolarityClassifier with text,feature,complex Attributes Bidimaps & Init Classifiers
	 * @param mainfolder path to dataset
	 * @param tb Training Text Instances Attributes
	 * @param fb Training Feature Instances Attributes
	 * @param cb Training Complex Instances Attributes
	 */
	//public PolarityClassifier(String mainfolder, BidiMap<String, Integer> tb, BidiMap<String, Integer> fb, BidiMap<String, Integer> cb)
	public PolarityClassifier(String mainfolder)
	{
		folder = mainfolder;
		//initializeAttributes(tb, fb, cb); //create bidimaps and init with parametes [text.tsv data , ....]
		System.out.println("PolarityClassifier Initialized with saved Attributes ..");

		//text[0] => preprocessed tweet of user[applied by same pipe of model], text[1] => tweet after reformat functions
		/*text = new Instance[2];
		feature = new Instance[2];
		complex = new Instance[2];*/

		preprocessed_tweet = new Instance[4];
		training_text = new InstanceList(Trainer.getTextPipe());
		training_feature = new InstanceList(Trainer.getFeaturePipe());
		training_complex = new InstanceList(Trainer.getComplexPipe());
		initializeClassifiers(); //initialize classifier by loading saved Models and init train data to instances
	}
	/**Initializes the MNB and SVM classifiers, by loading the previously generated models.*/
	private void initializeClassifiers()
	{
		//Reading all files Saved by Trainer to start Classifying Inputs
		mnb_classifiers = new Classifier[3];
		try
		{
			System.out.println("Loading saved Models ...");
			mnb_classifiers[0] = Trainer.loadModel(Trainer.Text_Train);
			mnb_classifiers[1] = Trainer.loadModel(Trainer.Feature_Train);
			mnb_classifiers[2] = Trainer.loadModel(Trainer.Complex_Train);
			lexicon_classifier = (SVMClassifier) Trainer.loadModel(Trainer.Lexicon_Train);

			mnb_classifiers[0].getAlphabet().stopGrowth();
			mnb_classifiers[1].getAlphabet().stopGrowth();
			mnb_classifiers[2].getAlphabet().stopGrowth();
			lexicon_classifier.getAlphabet().stopGrowth();
			/*System.out.println("Loading saved preprocessed data ...");
			training_text = Trainer.loadinstances(Trainer.Train+"T.bin");
			training_feature = Trainer.loadinstances(Trainer.Train+"F.bin");
			training_complex = Trainer.loadinstances(Trainer.Train+"C.bin");*/

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	/**Initializes the BidiMaps(tba,fba,cba) with the given bidimaps .*/
	/*private void initializeAttributes(BidiMap<String, Integer> tb,BidiMap<String, Integer> fb,BidiMap<String, Integer> cb)
	{
		tba = tb;
		fba = fb;
		cba = cb;
	}*/


	//<editor-fold desc="Applying Methodology">

	private Instance[] PreprocessTweet(Instance[] all)
	{
		//used for the processed tweet
		Instance[] tweet_representation = new Instance[4];  //text,feature,complex,lexicon
		try
		{
            //read instance of tweet text after being passed to TweetPreprocessor [Next pass to pipe of corresponding saved instancelist]
			tweet_representation[0] = getText(all[0]);
            //read instance of tweet feature after being passed to TweetPreprocessor [Next pass to pipe of corresponding saved instancelist]
			tweet_representation[1] = getFeature(all[1]);
            //read instance of tweet complex after being passed to TweetPreprocessor [Next pass to pipe of corresponding saved instancelist]
			tweet_representation[2] = getComplex(all[2]);
            //read instance of tweet text after being passed to TweetPreprocessor [NO Need for passing to Pipe as it ready to classify]
			tweet_representation[3] = all[3];
		} catch (Exception e) {
			e.printStackTrace();
		}

		//remove the attributes from the test set that are not used in the train set
		System.out.println("Applying reformat filter to Text,Feature,Complex ..");
		tweet_representation[0] = reformatText(tweet_representation[0]);
		tweet_representation[1] = reformatFeature(tweet_representation[1]);
		tweet_representation[2] = reformatComplex(tweet_representation[2]);

		System.out.println("Applying Classifiers on reformatted Instances ...");
		return tweet_representation;
	}

	/**The main method that sets up all the processes of the Ensemble classifier.
	 * Returns the decision made by the two classifiers.*/
	public Object[] apply(Instance[] preparedtweet)
	{
		preprocessed_tweet = PreprocessTweet(preparedtweet); //pass tweet representation to corresponding Pipe for their classifier

		double[] hc = applyHC(Arrays.copyOfRange(preprocessed_tweet,0,3));// Applies the learned MNB models and returns the results
		double lc   = applyLC(preprocessed_tweet[3]);   // Applies Lexicon based Classifier [swn]

		//Adjust depend on Distribution
		double content_pos_vals = (hc[0]+hc[2]+hc[4]) / 73.97;
		double content_neg_vals = (hc[1]+hc[3]+hc[5]) / 73.97;
		double hc_val = (1 + content_neg_vals - content_pos_vals) / 2;

		System.out.println("_______ Classifiers prediction ______");
		System.out.println("Tweet T Class = "+preparedtweet[0].getTarget());
		System.out.println("Content_Pos_vals = "+ content_pos_vals);
		System.out.println("Content_neg_vals = "+content_neg_vals);
		System.out.println("Hc_val = "+hc_val);
		System.out.println("Lc_val = "+lc);


		if (hc_val > 0.5 && preparedtweet[0].getTarget().toString().equals("negative") ||
				hc_val < 0.5 && preparedtweet[0].getTarget().toString().equals("positive"))
		{
			System.out.println("HC agreed on Right !");
			t_hc++;
		}else
			System.out.println("HC agreed on Wrong ^_^ !");


		if (lc > 0.5 && preparedtweet[0].getTarget().toString().equals("negative") ||
				lc < 0.5 && preparedtweet[0].getTarget().toString().equals("positive"))
		{
			System.out.println("LC agreed on Right !");
			t_lc++;
		}else
			System.out.println("LC agreed on Wrong ^_^ !");



		Object[] output = new Object[2];   //obj[0] => polarity string , obj[1] => polarity score

		if ((hc_val < 0.5) && (lc < 0.5))   //lc = 1.0  negative
			output[0] = "positive";

		else if ( ((hc_val > 0.5) && (lc > 0.5)) || hc_val > 0.7 )   //no need to take advantage of lc if hc_val agreed on neg > 0.7
		//else if (hc_val > 0.5 && lc > 0.5)
			output[0] = "negative";
		else
			output[0] = "nan";

		output[1] = hc_val;

		System.out.println("____________________________________ ");
		return output;
	}
	
	/**Applies the learned MNB models and returns the output of HC.*/
	private double[] applyHC(Instance[] hc_TweetInstances)
	{
		double[] scores = new double[6];  //index = 6  [two double values for each classifier of 3 NB]

		for (int i=0; i<mnb_classifiers.length; i++)  //loop for each classifier (3 nb) i=0 dh hsht8l 3la text i=1 hsht8l 3la feature ..
		{
			Instance test;

			if (i==0)
				test = hc_TweetInstances[0];
			else if (i==1)
				test = hc_TweetInstances[1];
			else
				test = hc_TweetInstances[2];


            LabelVector results = mnb_classifiers[i].classify(test).getLabelVector();
            double[] preds = new double[results.numLocations()];
            preds[0] = results.value("positive");
            preds[1] = results.value("negative");
            //positive => 0,2,4   [0 = value of positive label in first Classifier , ...]
            //negative => 1,3,5   [1 = value of negative label in first Classifier , ...]

			//Adjust for Priority
            if (i==0){
				scores[0] = preds[0]*31.07;  //postive score for text instance
				scores[1] = preds[1]*31.07;
			} else if (i==1){
				scores[2] = preds[0]*11.95;  //postive score for feature instance
				scores[3] = preds[1]*11.95;
			} else if (i==2){
				scores[4] = preds[0]*30.95;  //postive score for complex instance
				scores[5] = preds[1]*30.95;
			}
		}
		return scores;
	}

	/**Applies the LC classifier (LBRepresentation)*/
	private double applyLC(Instance lc_TweetInstance)
	{
		LabelVector result = lexicon_classifier.classify(lc_TweetInstance).getLabelVector();
		return result.value("negative"); //return negative not best
	}

	public void resetHC_LC()
	{
		this.t_lc = 0;
		this.t_hc = 0;
	}
	public int[] getTrueHC_LC()
	{
		return new int[]{t_hc,t_lc};
	}

    //</editor-fold>

	//<editor-fold desc="Reformats For removing features from test_set that not found in training set">
	private Instance reformatText(Instance instance)
	{
		 //
		return Trainer.reformatfeaturevector(instance, mnb_classifiers[0]);
	}
	private Instance reformatFeature(Instance instance)
	{
		//
		return Trainer.reformatfeaturevector(instance, mnb_classifiers[1]);
	}
	private Instance reformatComplex(Instance instance)
	{
		//
		return Trainer.reformatfeaturevector(instance, mnb_classifiers[2]);
	}

	//</editor-fold>

    //<editor-fold desc="Preprocess Tweets on Classifiers Pipe">
	/**Returns the Text-based Representations [pass Instance [i love my country] to the same pipe of corresponding instances.pipe ].*/
	private Instance getText(Instance data)
	{
		//System.out.println("[TextInstance][Before - Pipe] : "+data.getData().toString());
		//InstanceList newData = new InstanceList(mnb_classifiers[0].getInstancePipe());
		assert data!=null;
		training_text.addThruPipe(data);
		System.out.println("[TextInstance] : "+data.getData().toString());
		if (training_text.size() > 3000)
			training_text = new InstanceList(Trainer.getTextPipe());

		return data;
	}
	/**Returns the Feature-based Representations.*/
	private Instance getFeature(Instance data)
	{
		//System.out.println("[FeatureInstance][Before - Pipe] : "+data.getData().toString());
		assert data!=null;
        training_feature.addThruPipe(data);
		System.out.println("[FeatureInstance] : "+data.getData().toString());
		if (training_text.size() > 3000)
			training_text = new InstanceList(Trainer.getFeaturePipe());
        return data;
	}
	/**Returns the Complex (text+POS) Representations.*/
	private Instance getComplex(Instance data)
	{
		//System.out.println("[ComplexInstance][Before - Pipe] : "+data.getData().toString());
		assert data!=null;
        training_complex.addThruPipe(data);
		System.out.println("[ComplexInstance] : "+data.getData().toString());
		if (training_text.size() > 3000)
			training_text = new InstanceList(Trainer.getComplexPipe());
        return data;
	}
	//</editor-fold>

	//<editor-fold desc="Classifiers Evaluation">
	public Trial[] EvaluateClassifiers(Instance[] text, Instance[] feature, Instance[] complex, Instance[] lexicon)
	{
		InstanceList textinstances    = new InstanceList(new Noop());
		InstanceList featureinstances = new InstanceList(new Noop());
		InstanceList complexinstances = new InstanceList(new Noop());
		InstanceList lexiconinstances = new InstanceList(new Noop());

		System.out.println("Preprocessing & reformatting test dataset ..");

		for (int i = 0; i < text.length; i++)
		{
			try
			{      //Processing Instance to their Corresponding Classifier Pipe
				text[i]    = getText(text[i]);
				feature[i] = getFeature(feature[i]);
				complex[i] = getComplex(complex[i]);
			} catch (Exception e) {
				e.printStackTrace();
				break;
			}
			if (i%100==0)
				System.out.print(".");
		}
		System.out.println("Reformatting .. ");
		//Reformatting Instances Feature Vector to add them in their Instancelist to evaluate
		for (int i = 0; i < text.length; i++)
		{
			text[i]    = reformatText(text[i]);
			feature[i] = reformatFeature(feature[i]);
			complex[i] = reformatComplex(complex[i]);
		}

		System.out.println("Evaluating ..");
		textinstances.addAll(Arrays.asList(text));
		featureinstances.addAll(Arrays.asList(feature));
		complexinstances.addAll(Arrays.asList(complex));
		lexiconinstances.addAll(Arrays.asList(lexicon));

		return trialClassifiers(textinstances,featureinstances,complexinstances,lexiconinstances);
	}

	public Trial[] trialClassifiers(InstanceList text,InstanceList feature,InstanceList complex,InstanceList lexicon)
	{
		Trial[] trial = new Trial[4];

		trial[0] = new Trial(mnb_classifiers[0],text);
		saveClassifierTrial("TextNB",trial[0]);

		trial[1] = new Trial(mnb_classifiers[1],feature);
		saveClassifierTrial("FeatureNB",trial[1]);

		trial[2] = new Trial(mnb_classifiers[2],complex);
		saveClassifierTrial("ComplexNB",trial[2]);

		trial[3] = new Trial(lexicon_classifier,lexicon);
		saveClassifierTrial("LexiconSVM",trial[3]);

		System.out.println("Saving Instances Distributions ...");
		InstanceList training = Trainer.loadinstances(Trainer.Train+"T.bin");
		saveDatasetsDistribution("TextNB",training);
		training = Trainer.loadinstances(Trainer.Train+"F.bin");
		saveDatasetsDistribution("FeatureNB",training);
		training = Trainer.loadinstances(Trainer.Train+"C.bin");
		saveDatasetsDistribution("ComplexNB",training);

		return trial;
	}

	//<editor-fold desc="Saving work">
	private void saveDatasetsDistribution(String classifiername, InstanceList trainingset)
	{
		try
		{
			CSVPrinter writer = new CSVPrinter(new FileWriter(folder+"Results/Datadistribution/" +
					classifiername+".csv"),CSVFormat.EXCEL.withHeader("Classifier,Positive,Negative"));

			double pos =  trainingset.targetLabelDistribution().value("positive");
			double neg =  trainingset.targetLabelDistribution().value("negative");
			writer.printRecord(classifiername,pos,neg);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static void saveClassifierTrial(String filepath,Trial result)
	{
		try
		{
			CSVPrinter csvwriter = new CSVPrinter(new FileWriter(folder+"Results/"+filepath+".csv"),
					CSVFormat.EXCEL.withHeader("Tweet","True","Predicted","Negative","Positive"));

			csvwriter.printRecord(new ConfusionMatrix(result).toString());
			for (Classification r : result)
			{
				String d = r.getInstance().getName().toString();
				String t = r.getInstance().getTarget().toString();
				String p = r.getLabelVector().getBestLabel().toString();
				csvwriter.printRecord(d, t, p, r.getLabelVector().value("negative"),
						r.getLabelVector().value("positive"));
			}

			csvwriter.close();
		} catch (Exception e) {
			System.out.println(filepath);
			e.printStackTrace();
		}
	}

	//</editor-fold>

	//</editor-fold>
}