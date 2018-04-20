package Models.Sentiment_Analysis.Preprocessing;

import org.apache.commons.lang3.StringUtils;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.StringTokenizer;


public class FeaturePreprocessor extends Preprocessor implements Serializable
{
	LinkedList<String> dotsymbol = new LinkedList<>();
	LinkedList<String> exclamationsymbol = new LinkedList<>();

	/**
	 * initialize abbreviations,happ sad data aand init dotsymbol,exclamations
	 */
	public FeaturePreprocessor(String t)
	{
		this.main_folder = t;
		//specifying datasets need to work
		super.setDataSource(DataSource.Abbreviation);
		super.setDataSource(DataSource.HappyEmoj);
		super.setDataSource(DataSource.SadEmoj);
		System.out.println("----------------------------");
	}
	
	/**Some common pre-processing stuff*/
	public String getProcessed(String str)
	{
		StringTokenizer st = new StringTokenizer(str);
		String current;
		String toreturn = "";
		boolean isSpecial = false;

		while (st.hasMoreTokens())
		{
			current = st.nextToken();
			String backup = current;

			current = replaceEmoticons(current);			// current is altered to "happy"/"sad"
			current = replaceTwitterFeatures(current);		// i.e. links, mentions, hash-tags
			current = replaceNegation(current);				// if current is a negation word, then current = "negation"

			//Checking is this token is special to keeptrack as feature or not
			String filtered = backup.replaceAll("[^A-Za-z]", ""); //remove anything from token backuped

			//if this token contains more than 3 uppercases so its special
			if (StringUtils.isAllUpperCase(filtered) && filtered.length() > 3) {
				isSpecial = true;
				current = filtered.toLowerCase();  //current now replaced
			}
			//if this token start with contains # so word hashed is special
			if (backup.contains("#")){
				isSpecial = true;
				current = backup.substring(backup.indexOf("#")+1);
			}
			//if this token contains repetitions as likeeeeeeeee so its special
			if (filtered.length() > 0 && containsRepetitions(filtered)){
				isSpecial = true;
				current = replaceConsecutiveLetters(backup);  //apply ConsecutiveLetters
			}
	/*		if ((current.contains("hashtagsymbol")
					|| current.contains("urlinksymbol")
					|| current.contains("negation")
					|| current.contains("usermentionsymbol")
					|| current.contains("consecutivesymbol")
					|| current.contains("alcapital")) )
				isSpecial = true;*/

			//Check if this token is special add it to the tweet string
			if (isSpecial)
				toreturn = toreturn.concat(current+" ");
			isSpecial = false;
		}
		return toreturn;
	}

}