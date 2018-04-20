package Models.Sentiment_Analysis.Preprocessing;

import java.io.Serializable;
import java.util.StringTokenizer;

public class TextPreprocessor extends Preprocessor implements Serializable
{
	/** Construct TextPreprocessor and Initialize list of abbreviations,HappyEmotions,SadEmotions from their datasets
	 * @param t main directory for sentimentanalysis path
	 */
	public TextPreprocessor(String t)
	{
		this.main_folder = t;
		//specifying datasets need to work
		super.setDataSource(DataSource.Abbreviation);
		super.setDataSource(DataSource.HappyEmoj);
		super.setDataSource(DataSource.SadEmoj);
	}

	/** Master function of TextPreprocessor which perform the following function to a give text :
	 *  replaceEmotions
	 *  replaceTwitterFeatures
	 *  replaceConsecutiveLetters
	 *  replaceNegation
	 *  replaceAbbreviations
	 *  replace other undefined symbols
	 *
	 *  @param str give text to apply TextProcessor work on it
	 */
	public String getProcessed(String str)
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
			current = current.replaceAll("[^A-Za-z0-9]", " "); //replace any characters that not recognize
			toreturn = toreturn.concat(" "+current);
		}
		return toreturn;
	}
	

}