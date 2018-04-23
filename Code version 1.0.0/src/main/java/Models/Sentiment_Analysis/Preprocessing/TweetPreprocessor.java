package Models.Sentiment_Analysis.Preprocessing;

import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.Label;
import cc.mallet.types.LabelAlphabet;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

import java.io.Serializable;
import java.util.Arrays;


public class TweetPreprocessor implements Serializable
{
	String main_folder;

	TextPreprocessor tp;
	ComplexPreprocessor cp;
	FeaturePreprocessor fp;
	LexiconPreprocessor lp;

	public static LabelAlphabet TargetAlphabets;
	MaxentTagger tagger;


	public TweetPreprocessor(String t)
	{
		main_folder = t;

		tp = new TextPreprocessor(main_folder);
		cp = new ComplexPreprocessor();
		fp = new FeaturePreprocessor(main_folder);
		lp = new LexiconPreprocessor(main_folder);

		TargetAlphabets = new LabelAlphabet();
		TargetAlphabets.lookupLabel("negative"); //0.0
		TargetAlphabets.lookupLabel("positive"); //1.0
		tagger = new MaxentTagger(main_folder+"datasets/gate-EN-twitter.model");
	}


	/**
	 * @return text,feature,lexicon,complex instances[4] for tweet set
	 */
	public Instance[] getAllInstances(Instance tweet)
    {
		/*calculate each of them for each tweet that will be used by Polarity Classifier */
		//text_instances feature_instances complex_instances lexicon_instances
		Instance[] preparedInstances = new Instance[4];
		preparedInstances[0] = getTextInstances(tweet);
		preparedInstances[1] = getFeatureInstances(tweet);
		preparedInstances[2] = getComplexInstances(preparedInstances[0]);
		preparedInstances[3] = getLexiconInstances(tweet);

		return preparedInstances;
	}

	/** setter function used to set tweet to do Tweetpreprocessor
	 * @param t  tweet
	 **/

	/**
	 *  Starting tweetpreprocessor on the tweet set, Initializing text,complex,lexicon,feature Instances for give tweet
	 */
	public Instance[] startProc(Instance tweet)
	{
		return getAllInstances(tweet);
	}

	/**Instantiates the text-based Instances*/
	private Instance getTextInstances(Instance tweet)
	{
		//System.out.println("[TextInstance][Before] Tweet Instance : "+tweet);
		String tmp_txt = tp.getProcessed(tweet.getData().toString());
		//System.out.println("[TextInstance][After] Tweet Instance : "+tmp_txt);
		return new Instance(tmp_txt,tweet.getTarget(),tweet.getName(),tweet.getSource());
	}

	/**Instantiates the complex-based Instances*/
	private Instance getComplexInstances(Instance processed_text)
	{
		//System.out.println("[ComplexInstance][Before] Tweet Instance : "+processed_text);
		String tmp_cmplx = cp.getProcessed(processed_text.getData().toString(), tagger);
		//System.out.println("[ComplexInstance][After] Tweet Instance : "+tmp_cmplx);
		return new Instance(tmp_cmplx,processed_text.getTarget(),processed_text.getName(),processed_text.getSource());
	}

	/**Initializes the feature-based Instances*/
	private Instance getFeatureInstances(Instance tweet)
	{
		//System.out.println("[FeatureInstance][Before] Tweet Instance : "+tweet);
		String tem = fp.getProcessed(tweet.getData().toString());
		//System.out.println("[FeatureInstance][After] Tweet Instance : "+tem);
		return new Instance(tem,tweet.getTarget(),tweet.getName(),tweet.getSource());
	}

	private Instance getLexiconInstances(Instance tweet)
	{
		//System.out.println("[LexiconInstance][Before] Tweet Instance : "+tweet);

		double[] vals = lp.getProcessed(tweet.getData().toString(), tagger);
		int indices[] = new int[vals.length];

		int count = 0;
		for (int j = 0; j < vals.length; j++)
		{
			if (vals[j] != 0.0)
			{
				vals[count] = vals[j];
				indices[count] = j;
				count++;
			}
		}

		indices = Arrays.copyOf(indices, count);
		vals = Arrays.copyOf(vals, count);

		FeatureVector fv = new FeatureVector(LexiconPreprocessor.getDataAlphabet(),indices,vals);
		//for neutral problem
		Label classLabel;
		if (TargetAlphabets.contains(tweet.getTarget().toString()))
			classLabel = TweetPreprocessor.TargetAlphabets.lookupLabel(tweet.getTarget().toString());
		else
			classLabel = TweetPreprocessor.TargetAlphabets.lookupLabel("negative");

	    return new Instance(fv,classLabel,tweet.getName(),tweet.getSource());
	}
}