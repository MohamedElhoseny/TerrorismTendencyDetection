package Models.Sentiment_Analysis.Preprocessing;

import edu.stanford.nlp.tagger.maxent.MaxentTagger;

import java.io.Serializable;
import java.util.StringTokenizer;

public class ComplexPreprocessor implements Serializable
{
	
	/** responsible for applying Part of speech on a give text
     * @param str    tweet that will be used to apply tagger on it
     * @param tagger side of stanford corenlp used to apply part of speech on a given text
     * @return tagged twwet
     * */
	public String getProcessed(String str, MaxentTagger tagger)
	{
		return getPOS(str, tagger);
	}

	/** applying part of speech according to the parameters .*/
	private String getPOS(String sample, MaxentTagger tagger)
	{
		String tagged = tagger.tagString(sample.trim().replaceAll(" +", " ")); //tagging

		StringTokenizer stk = new StringTokenizer(tagged);
		StringBuilder output = new StringBuilder();

		while (stk.hasMoreTokens())
		{
			String tmp = stk.nextToken();

			String tmp2 = tmp.replaceAll("[^A-Za-z_0-9]", "");
			output.append(tmp2).append(" ");

			if (tmp.contains("."))
				output = new StringBuilder(output.toString().concat("."));
			if (tmp.contains("!"))
				output = new StringBuilder(output.toString().concat("!"));
			if (tmp.contains(","))
				output = new StringBuilder(output.toString().concat(","));
			if (tmp.contains("?"))
				output = new StringBuilder(output.toString().concat("?"));
		}
		return output.toString();
	}
}