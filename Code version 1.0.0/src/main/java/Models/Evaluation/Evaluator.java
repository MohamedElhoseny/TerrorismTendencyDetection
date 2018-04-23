package Models.Evaluation;
import Models.Sentiment_Analysis.Methodology.SentimentAnalyser;
import Models.Topic_Modeling.LabeledLDA_work.LabeledTopicModel;
import Models.Word_Weighting.WordweightingEvaluator;
import cc.mallet.types.Instance;
import java.io.Serializable;
import java.util.concurrent.*;

public class Evaluator implements Serializable
{

    //Features models
    private static SentimentAnalyser sentimentAnalyser;
    private static LabeledTopicModel topicModel;
    private static WordweightingEvaluator wordWeightingModel;

    private static ExecutorService threadpool;
    private static Evaluator evaluator;
    private Evaluator(){}

    /** Constructing Evaluator include constructing sentiment, wordweight, topicmodel models
     *  also constructing a thread pool
     * @return a singleton evaluator object
     */
    public static Evaluator getEvaluator()
    {
        if (evaluator == null)
        {
            sentimentAnalyser = new SentimentAnalyser();
            topicModel = new LabeledTopicModel();
            wordWeightingModel = new WordweightingEvaluator();
            evaluator = new Evaluator();
            threadpool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        }

        return evaluator;
    }

    /** Used to apply 3 features on a give tweet Instance
     * @param tweet the tweet that wants to catch its feature values
     * @return array of double include scores for this tweet across features
     */
    public double[] getFeatureValues(Instance tweet)
    {
        Callable<Double> S = () -> {
            //Polarity
            return sentimentAnalyser.evaluate(tweet);
        };

        Callable<Double> T = () -> {
            //topic modelling
            return topicModel.evaluate(tweet);
        };

        Callable<Double> W = () -> {
            //word weighting
            return wordWeightingModel.evaluate(tweet);
        };

        if (threadpool == null)
            Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        Future<Double> s_future = threadpool.submit(S);
        Future<Double> t_future = threadpool.submit(T);
        Future<Double> w_future = threadpool.submit(W);

        while (!s_future.isDone() && !w_future.isDone() && !t_future.isDone());

        double[] scores = new double[3]; //s_score, t_score, w_score
        try
        {
            scores[0] = w_future.get();   //wordweightingscore
            scores[1] = s_future.get();   //Sentimentanalysis
            scores[2] = t_future.get();   //TopicModelling

        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
            System.exit(0);
        }

        return scores;
    }

    /** responsible for applying Evaluation equation specified for labelling featuresScores
     * @param values array of double refering to tweet scores
     * @return a String for the the classifing specified on the Evaluator
     */
    public String classifyFeatureValues(double[] values)
    {
        double result = applyEvaluationEquation(values);

        if (result <  1.0)
            return "neutral";
        else if (result > 1.5 )
            return "negative";
        else
            return "positive";
    }

    /** applying evaluation equation to feature values for labelling a tweet according its score
     * @param values featurevector values for a tweet [wordweighting,sentiment,topicmodelling]
     * @return score of given tweet values
     */
    public static double applyEvaluationEquation(double[] values)
    {
        //Evaluator param
        //public static double s_perc = 0.3;
        //public static double t_perc = 0.35;
        //public static double w_perc = 0.35;
        //return (values[0] * w_perc) + (values[1] * s_perc) + (values[2] * t_perc);

        double tweet_polarity = values[1];   // S
        double tweet_tendency = (values[0] + values[2]) / 2;        //average between W & T

        if (tweet_tendency > 0.25)
            tweet_polarity += 1.0;

        return tweet_polarity;
    }

}
