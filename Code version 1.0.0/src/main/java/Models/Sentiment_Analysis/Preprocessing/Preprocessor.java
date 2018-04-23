package Models.Sentiment_Analysis.Preprocessing;

import java.io.*;
import java.util.Hashtable;
import java.util.LinkedList;

public class Preprocessor
{
    public String main_folder;
    protected static Hashtable<String, String> abbreviations;
    protected static LinkedList<String> happyEmo;
    protected static LinkedList<String> sadEmo;
    protected static LinkedList<String> posWords;
    protected static LinkedList<String> negWords;
    public enum DataSource {
        Abbreviation,
        HappyEmoj,
        SadEmoj,
        PosWord,
        NegWord
    }

    /** Method to specify which datasource this preprocess need to initialize it
     * @param source Enum specify the source
     */
    public void setDataSource(DataSource source)
    {
        switch (source)
        {
            case Abbreviation:
                if (abbreviations == null)
                    getAbbreviations();
                break;
            case HappyEmoj:
                if (happyEmo == null)
                    getHappyEmoticons();
                break;
            case SadEmoj:
                if (sadEmo == null)
                    getSadEmoticons();
                break;
            case PosWord:
                if (posWords == null)
                    getPosWords();
                break;
            case NegWord:
                if (negWords == null)
                    getNegWords();
                break;
            default:
                break;
        }
    }

    /**Replaces consecutive letters as follow "something likeeeeeeeeeeeee" to "something like"
     * @param current a token to check for consecutiveLetters
     * @return token doesn't continue any repeatation letter
     * */
    public String replaceConsecutiveLetters(String current)
    {
        String tmp = current.replaceAll("[^A-Za-z]", ""); //match all strings that contain a non-letter
        if (tmp.length()>0 && containsRepetitions(tmp))
        {
            tmp = replaceRepetitions(tmp);
            return tmp;
        }
        return current;
    }

    /**Check whether the given String contains consecutive letters*/
    public boolean containsRepetitions(String str)
    {
        StringBuilder toreturn = new StringBuilder(str.substring(0, 1));
        char prev = str.charAt(0);
        int cnt = 0;

        for (int i=1; i<str.length(); i++)
        {
            char current = str.charAt(i);
            toreturn.append(current);
            if (current==prev){
                cnt++;
                if (cnt>=2)
                    return true;
            }else
                cnt = 0;
            prev = str.charAt(i);
        }
        return false;
    }

    /** replace repetitions of letters, must called by replaceConsecutiveLetters function */
    public String replaceRepetitions(String str)
    {
        StringBuilder toreturn= new StringBuilder(str.substring(0, 1));
        char prev = str.charAt(0);
        boolean found = false;
        for (int i=1; i<str.length(); i++){
            char current = str.charAt(i);
            toreturn.append(current);
            if (current==prev){
                if (found)
                    toreturn = new StringBuilder(toreturn.substring(0, toreturn.length() - 1));
                else
                    found = true;
            }else if (found)
                found = false;
            prev = str.charAt(i);
        }
        return toreturn.toString();
    }

    /**Replaces emoticons with their value: "feeling sad" vs "feeling happy"
     * @param current a token to check if it's a happy or sad emotion
     **/
    public String replaceEmoticons(String current)
    {
        //System.out.println("Checking is "+current+" is Emotion ?");
        if (happyEmo.contains(current))
            current = "feeling happy";
        else if (sadEmo.contains(current))
            current = "feeling sad";

        //System.out.println("Checking returned : "+current);
        return current;
    }

    /** Replaces UserMentions, Hashtags, UrlLinks
     * @param current a token to check
     * @return hastags token without '#', replace userMentions to 'usermentionsymbol', replace urlLinks to 'urlinksymbol'
     **/
    public String replaceTwitterFeatures(String current)
    {
        if (current.contains("#"))
            current = current.replaceAll("#", " ");
        if (current.contains("@"))
            current = "usermentionsymbol";
        if (current.contains("http:") || current.contains("https:"))
            current = "urlinksymbol";
        return current;
    }

    /**Finds whether a negation occurs in a word and returns "not".*/
    public String replaceNegation(String current)
    {
        String tmp1 = current.toLowerCase();
        if (tmp1.endsWith("n\'t"))   //don't ..
        {
            tmp1 = tmp1.substring(0, tmp1.lastIndexOf("n\'t"));
            tmp1 = tmp1.concat(" not");
            if (tmp1.contains("wo "))
                tmp1 = "will not";
            else if (tmp1.contains("ca "))
                tmp1 = "can not";
            else if (tmp1.contains("ai "))
                tmp1 = "is not";

            return tmp1;
        }
        String tmp = current.replaceAll("[^A-Za-z0-9]", "").toLowerCase();

        if (tmp.equals("cannot") || tmp.equals("cant"))
            return "can not";

        if ((tmp.equals("not"))
                || (tmp.equals("no"))
                || (tmp.equals("none"))
                || (tmp.equals("noone"))
                || tmp.equals("nobody")
                || tmp.equals("nothing")
                || tmp.equals("neither")
                || tmp.equals("nor")
                || tmp.equals("nowhere")
                || tmp.equals("never")
                || tmp.equals("nver")
                || tmp.equals("hardly")
                || tmp.equals("scarcely")
                || tmp.equals("barely")
                || tmp.equals("no1"))
            return "not";

        return current;
    }

    /**Replaces abbreviations*/
    public String replaceAbbreviations(String current)
    {
        String tmp = current.replaceAll("[^A-Za-z0-9\']", "").toLowerCase();
        if (abbreviations.get(tmp)!=null){
            tmp = abbreviations.get(tmp);
            return tmp;
        }
        return current;
    }


    /**Fetch the list of abbreviations and return its contents in a hashtable.*/
    private void getAbbreviations()
    {
        abbreviations = new Hashtable<>();
        try
        {
            BufferedReader rdr = new BufferedReader(new FileReader(new File(main_folder+"datasets/abbreviations.txt")));
            String inline;
            while ((inline=rdr.readLine())!=null)
                abbreviations.put(inline.substring(0, inline.indexOf("=")), inline.substring(inline.indexOf("=")+1));
            rdr.close();
            System.out.println("Abbreviations dataset loaded ..");

        }catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    /**Get the list of the happy emoticons*/
    private void getHappyEmoticons()
    {
        happyEmo = new LinkedList<>();
        File happy = new File(main_folder+"datasets/happyEmoticons");
        BufferedReader brdr2;
        try
        {
            brdr2 = new BufferedReader(new FileReader(happy));
            String line;
            while ((line=brdr2.readLine()) != null)
                happyEmo.add(line);
            brdr2.close();
            System.out.println("HappyEmotions dataset loaded ..");

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**Get the list of the sad emoticons*/
    private void getSadEmoticons()
    {
        sadEmo = new LinkedList<>();
        File happy = new File(main_folder+"datasets/sadEmoticons");
        try {
            BufferedReader brdr2 = new BufferedReader(new FileReader(happy));
            String line;
            while ((line=brdr2.readLine()) != null)
                sadEmo.add(line);
            brdr2.close();
            System.out.println("SadEmotions dataset loaded ..");

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**Get the list with the positive keywords*/
    private LinkedList<String> getPosWords()
    {
        File pos = new File(main_folder+"datasets/positive-words.txt");
        posWords = new LinkedList<>();
        try{
            BufferedReader brdr2 = new BufferedReader(new FileReader(pos));
            int k = -1;
            String line;
            while ((line=brdr2.readLine()) != null){
                k++;
                if (k>34)
                    posWords.add(line);
            }
            brdr2.close();
            System.out.println("Positive dataset loaded ..");

        }catch (FileNotFoundException fnf){
            System.out.println("Negative File not found");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return posWords;
    }

    /**Get the list with the negative keywords*/
    private LinkedList<String> getNegWords()
    {
        File neg = new File(main_folder+"datasets/negative-words.txt");
        negWords = new LinkedList<>();
        try{
            BufferedReader brdr = new BufferedReader(new FileReader(neg));
            String line;
            int k = -1;
            while ((line=brdr.readLine()) != null){
                k++;
                if (k>34)
                    negWords.add(line.toLowerCase());
            }
            brdr.close();
            System.out.println("Negative dataset loaded ..");
        }catch (FileNotFoundException fnf){
            System.out.println("Negative File not found");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return negWords;
    }
}
