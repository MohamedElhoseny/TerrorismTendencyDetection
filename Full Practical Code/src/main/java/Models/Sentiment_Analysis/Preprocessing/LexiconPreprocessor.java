package Models.Sentiment_Analysis.Preprocessing;
import Models.Sentiment_Analysis.SentiWordnet.SWN3;
import cc.mallet.types.Alphabet;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

import java.io.IOException;
import java.io.Serializable;
import java.util.StringTokenizer;

public class LexiconPreprocessor extends Preprocessor implements Serializable
{
	public static SWN3 swn;
	public static Alphabet data;

	/** Construct LexiconPreprocessor and Initialize list of abbreviations,HappyEmotions,SadEmotions,Pos,Neg Words, SentiwordNet from their datasets
	 * @param t main directory for sentimentanalysis path
	 */
	public LexiconPreprocessor(String t)
	{
		this.main_folder = t;

		//specifying datasets need to work
		super.setDataSource(DataSource.Abbreviation);
		super.setDataSource(DataSource.HappyEmoj);
		super.setDataSource(DataSource.SadEmoj);
		super.setDataSource(DataSource.PosWord);
		super.setDataSource(DataSource.NegWord);

		if (swn == null)
        {
            System.out.println("Loading SentiwordNet ..");
            swn = new SWN3(main_folder+"datasets/sentiwordnet.txt");
            System.out.println("SentiwordNet Constructed ..");
        }

		data = new Alphabet();
		data.lookupIndex("verb",true);
		data.lookupIndex("noun",true);
		data.lookupIndex("adj",true);
		data.lookupIndex("adv",true);
		data.lookupIndex("wordnet",true);
		data.lookupIndex("polarity",true);
	}
	
	/**Some common pre-processing stuff*/
	public double[] getProcessed(String str, MaxentTagger tagger)
	{
		StringTokenizer st = new StringTokenizer(str);
		String current;
		String toreturn = "";

		while (st.hasMoreTokens())
		{
			current = st.nextToken();						
			current = replaceEmoticons(current);			// current is altered to "happy"/"sad"
			current = replaceTwitterFeatures(current);		// i.e. links, mentions, hash-tags
			current = replaceConsecutiveLetters(current);	// replaces more than 2 repetitive letters with 2
			current = replaceNegation(current);				// if current is a negation word, then current = "not"
			current = replaceAbbreviations(current);		// if current is an abbreviation, then replace it
			current = current.replaceAll("[^A-Za-z]", " ");
			toreturn = toreturn.concat(" "+current);
		}
		return getPOS(toreturn, tagger);
	}

	/**The only extra method compared to the text-based approach.*/
	private double[] getPOS(String sample, MaxentTagger tagger)
	{
		String tagged = tagger.tagString(sample.trim().replaceAll(" +", " "));
		StringTokenizer stk = new StringTokenizer(tagged);
		StringBuilder output = new StringBuilder();

		double noun=0.0;
		double adj=0.0;
		double verb=0.0;
		double adv=0.0;
		double polarity = 0.0;
		boolean foundNegation = false;
		
		while (stk.hasMoreTokens())
		{
			String token = stk.nextToken();

			String tmp = token.substring(0, token.lastIndexOf("_")).toLowerCase();  //Word
			int idx = token.lastIndexOf("_");
			String pos = token.substring(idx+1);  //POS

			//Check for negation
			if (tmp.equals("not"))
				foundNegation = true;

			else if (pos.equals("NN") || pos.equals("NNS") || pos.equals("NNP") || pos.equals("NNPS"))
			{
				output.append("n#").append(tmp).append(" ");
				if (foundNegation){
					foundNegation = false;
					noun = noun - swn.extract(tmp, "n");  //subtract if negation sentiment
				}else
					noun = noun + swn.extract(tmp, "n");  //add if not negation sentimen

			}else if (pos.equals("RB") || pos.equals("RBR") || pos.equals("RBS") || pos.equals("RP")){
				output.append("r#").append(tmp).append(" ");
				if (foundNegation){
					foundNegation = false;
					adv = adv - swn.extract(tmp, "r");
				}else
					adv = adv + swn.extract(tmp, "r");

			}else if (pos.equals("JJ") || pos.equals("JJR") || pos.equals("JJS")){
				output.append("a#").append(tmp).append(" ");
				if (foundNegation){
					foundNegation = false;
					adj = adj - swn.extract(tmp, "a");
				}else
					adj = adj + swn.extract(tmp, "a");

			}else if (pos.equals("VB") || pos.equals("VBD") || pos.equals("VBG") || pos.equals("VBN") || pos.equals("VBP") || pos.equals("VBZ")){
				output.append("v#").append(tmp).append(" ");
				if (foundNegation){
					foundNegation = false;
					verb = verb - swn.extract(tmp, "v");
				}else
					verb = verb + swn.extract(tmp, "v");
			}

			// The polarity value  [score of this word according to Negative,Positive words]
			if (tmp.equals("not"))
				foundNegation = true;

			//Check if this word is positive or negative
			else if (posWords.contains(tmp)){
				if (foundNegation){
					polarity = polarity - 1.0;
					foundNegation = false;
				}else{
					polarity = polarity + 1.0;
				}
			}else if (negWords.contains(tmp)){
				if (foundNegation){
					polarity = polarity + 1.0;
					foundNegation=false;
				}else{
					polarity = polarity - 1.0;
				}
			}
		}

		//Preparing Output of this tweet
		double[] ret = new double[6];
		ret[0] = verb;
		ret[1] = noun;
		ret[2] = adj;
		ret[3] = adv;
		ret[4] = adv+verb+noun+adj;  //score of wordnet at all
		ret[5] = polarity; //score of founding pos/neg words
		return ret;
	}


	public static Alphabet getDataAlphabet()
	{ return data;}

}