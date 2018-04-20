package Models.Word_Weighting;

import Models.Evaluation.Factory;
import Models.Sentiment_Analysis.Preprocessing.TextPreprocessor;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import cc.mallet.util.FileUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Map;
import java.util.StringTokenizer;

public class WordweightingEvaluator implements Factory,Serializable
{
    private String corpus;
    private Map<String,Double> dictionary;
    private TextPreprocessor preprocessor;
    private InstanceList testing;

    public WordweightingEvaluator()
    {
        //
        this(true);
    }

    public WordweightingEvaluator(boolean isTrained)
    {
        if (isTrained)
        {
            loadDictionary();
            preprocessor = new TextPreprocessor("resources/datasets/sentimentanalysis/");
            testing = new InstanceList(WordWeightingModel.getTextRepresentationPipe());
        }
    }

	public void Train(String corpus) throws Exception
    {
        this.corpus = corpus;

        WordWeightingModel model = new WordWeightingModel(corpus,false);

        //removing stopwords, nonAlpa and applying stemming
        model.PreprocessCorpus();

        //start wordweighting on given corpus
        model.Evaluate();

        /** Modifying corpus */
        //model.insertWordtoCorpus("command",Freq.High, true,false);
        //model.removeWordFromCorpus("Tuesday",true,false);

        //saving outputs
        model.SaveOutputDetails();
        model.SaveWordScores();
    }

    @Override
    public double evaluate(Instance tweet)
    {
        //tweet not preprocessed !
        return getWeight(tweet.getData().toString());
    }

    /** Get weight of Tweet by getting word score for each token in tweet and normalize it
     * @param tweet given tweet
     * @return sum or average of all words weight
     */
    public double getWeight(String tweet)
    {
        tweet = preprocessor.getProcessed(tweet);
        Instance i = new Instance(tweet, "terrorism", null, null);
        testing.addThruPipe(i);

        //tweet words
        TokenSequence tokens = (TokenSequence) i.getData();

        //corresponding weights
        ArrayList<Double> words_weight = new ArrayList<>();

        for (Token token : tokens)
        {
            double w = 0.0;
            String word = token.getText();
            if (dictionary.containsKey(word))
            {
                w = dictionary.get(word);
                words_weight.add(w);
            }
            System.out.println("Word : " + word + " Weight = " + w);
        }

        //Tweet Score
        double result = 0.0;
        for (Double w: words_weight)
            result += w;

        double numweighted = (double) words_weight.size();
        double numtokens = (double) tokens.size();
        Double tweet_tendency = result / numtokens;
        Double words_scores =  result / numweighted;
        //Double tweetscore = tweet_tendency * words_scores;
        Double tweetscore = (tweet_tendency + words_scores)/2;
        System.out.println("--------------------------\nToken size = "+numtokens+"\nToken found = "+numweighted);
        System.out.println("(+) Score = "+result);
        System.out.println("tweet_tendency  = "+tweet_tendency);
        System.out.println("words_scores = "+words_scores);
        System.out.println("Tweet score = "+tweetscore);


        testing.remove(0);
        if (tweetscore.isNaN())
            return 0.0;
        else
            return tweetscore;
    }

    /** Responsible for removing any noisy word from dataset according to the vocabulary topic dictionary
     *  @param datasetpath Dataset required to clean [IT MUST BE PREPARED AND STEMMED !]
     *  @param vocabpath Dictionary continue all topic 'terrorism' vocabulary
     */
    public static void cleanNoisyVocabulary(String datasetpath,String vocabpath)
    {
        try {

            String[] lines = FileUtils.readFile(new File(datasetpath));
            String[] vocabs = FileUtils.readFile(new File(vocabpath));

            FileWriter writer = new FileWriter("resources/datasets/wordweighting/out/r.txt");
            StringTokenizer tokenizer;

            boolean isempty = true;
            for (String line : lines) {
                tokenizer = new StringTokenizer(line);
                while (tokenizer.hasMoreTokens()) {
                    String token = tokenizer.nextToken();
                    for (String v : vocabs) {
                        if (v.equals(token)) {
                            writer.write(token + " ");
                            isempty = false;
                            break;
                        }
                    }
                }
                if (!isempty) {
                    writer.write("\n");
                    isempty = true;
                }

            }
            writer.close();
        }catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    /** Lizzy function using for deleteing lines from vocab dictionary
     * @param vocabpath dictionary
     * @param numline Number line at which any vocabulary occur after will be deleted
     */
    public static void deletelines(String vocabpath, int numline)
    {
        try {

            String[] lines = FileUtils.readFile(new File(vocabpath));

            FileWriter writer = new FileWriter("Resources/datasets/wordweighting/vocabulary.txt");
            for (int i = 0; i < numline; i++) {
                writer.write(lines[i] + "\n");
            }

            writer.close();
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    //<editor-fold desc="Corpus [delegating]" default-state="Capsulated">
    /** Used to load .csv word scores for evaluating tweets*/
    private void loadDictionary() {
        dictionary = WordWeightingModel.getDictionary();
        System.out.println("Word Weighting Dictionary Loaded : "+dictionary.size());
    }
    /** Used to update .csv word file after removing all vocab passed
     * @param vocabfile vocabulary file containing words required only to be in .csv word scores
     */
    public void updateDictionary(String vocabfile) {
        try {
            WordWeightingModel.updateDictionary(vocabfile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    /** Used to extend the corpus .txt file [the corpus file is a txt file not a prepared or preprocessed]
     * @param datasetpath the dataset file path to read and add to the existing corpus
     */
    public void extendCorpus(String datasetpath) {
        //
        WordWeightingModel.extendCorpus(corpus, datasetpath);
    }
    //</editor-fold>
}


