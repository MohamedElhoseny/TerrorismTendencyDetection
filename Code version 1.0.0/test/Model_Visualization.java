import Models.Sentiment_Analysis.Methodology.SentimentAnalyser;
import Models.Sentiment_Analysis.Utils.IOUtils;
import cc.mallet.types.InstanceList;
import org.junit.Test;

public class Model_Visualization
{
    InstanceList testingInstances2 = IOUtils.readTestdataset(
            "resources/datasets/model/test/labeledtweets.csv",true);

    @Test
    public void VisualizeSentiment()
    {
        SentimentAnalyser analyser = new SentimentAnalyser();
        analyser.evaluateClassifiers(testingInstances2,true);
    }
}
