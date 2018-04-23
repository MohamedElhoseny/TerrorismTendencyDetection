import Models.Evaluation.Evaluator;
import Models.Sentiment_Analysis.Methodology.SentimentAnalyser;
import Models.Sentiment_Analysis.Preprocessing.TweetPreprocessor;
import Models.Sentiment_Analysis.Utils.IOUtils;
import Models.Topic_Modeling.LabeledLDA_work.LabeledTopicModel;
import Models.Word_Weighting.WordweightingEvaluator;
import cc.mallet.pipe.Noop;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import org.junit.Before;
import org.junit.Test;

import java.io.FileWriter;
import java.io.IOException;

public class Model_Features
{
    InstanceList testingInstances2 = IOUtils.readTestdataset(
            "resources/datasets/model/test/labeledtweets.csv", true);


    @Test
    public void evaluator() throws IOException
    {
        InstanceList testingInstances2 = IOUtils.readTestdataset("resources/datasets/model/test/labeledtweets.csv", true);
        //InstanceList testingInstances2 = IOUtils.readTestdataset("resources/datasets/model/train/dataset.csv",true);

        Evaluator evaluator = Evaluator.getEvaluator();

        FileWriter writer = new FileWriter("test_resources/test_evaluator.txt");
        for (Instance i : testingInstances2) {
            double[] v = evaluator.getFeatureValues(i);
            writer.write("Tweet : " + i.getData() + "\n");
            writer.write("Word weight = " + v[0] + "\n");
            writer.write("Sentiment = " + v[1] + "\n");
            writer.write("Topic model = " + v[2] + "\n");
            writer.write("Evaluation EQ = " + Evaluator.applyEvaluationEquation(v) + "\n");
            writer.write("Evaluated as : " + evaluator.classifyFeatureValues(v) + "\n");
            writer.write("===================================\n");
        }
        writer.close();
    }

    @Test
    public void TopicModelling()
    {
        LabeledTopicModel tm = new LabeledTopicModel();
        tm.EvaluateTestingSet(testingInstances2, false);
        System.out.println(testingInstances2.get(7).getData());
        tm.evaluate(testingInstances2.get(7));
    }

    @Test
    public void Sentiment()
    {
        SentimentAnalyser analyser = new SentimentAnalyser();
        analyser.evaluateMethodology(testingInstances2);
    }

    @Test
    public void WordWeighting()
    {
        WordweightingEvaluator evaluator = new WordweightingEvaluator(true);
        for (Instance i : testingInstances2)
            evaluator.evaluate(i);
    }
}
