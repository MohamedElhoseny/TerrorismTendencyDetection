import Models.Evaluation.Profile;
import Models.Evaluation.TerrorismTendencyClassifier;
import Models.Sentiment_Analysis.Methodology.SentimentAnalyser;
import Models.Sentiment_Analysis.Utils.IOUtils;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import javafx.application.Application;
import javafx.stage.Stage;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class MainClass extends Application
{
	
	public static void main(String[] args){ launch(args);}
	
    @Override
    public void start(Stage primaryStage)
    {
        /* Training & Testing datasets */
        //InstanceList trainingInstances = IOUtils.readTestdataset("resources/datasets/model/train/training.csv",true);
        //InstanceList testingInstances = IOUtils.readTestdataset("resources/datasets/model/test/unlabeledtweets.txt",false);
        //InstanceList testingInstances2 = IOUtils.readTestdataset("resources/datasets/model/test/labeledtweets.csv",true);
        //InstanceList evaluateTestInstances = IOUtils.readSparseDataset("resources/datasets/model/out/train.csv");


        TerrorismTendencyClassifier tendencyClassifier = new TerrorismTendencyClassifier();

        /* Training */
        //tendencyClassifier.train(trainingInstances,false);
        //tendencyClassifier.train(evaluateTestInstances,true);
        //tendencyClassifier+.saveModel();


        //editfeaturevalue();
        //InstanceList training = tendencyClassifier.getEvaluatedSparse(); //instances after applying new Evaluator updates
        //tendencyClassifier.train(training,true);

        tendencyClassifier.loadPreviousClassifier("resources/datasets/model/out/Classifier.bin");
        //InstanceList testing = tendencyClassifier.PipeInstances(testingInstances2);

        //tendencyClassifier.evaluateModel(testingInstances2,false);
        //tendencyClassifier.visualizeClassifier();

        /* Prediction */
        //tendencyClassifier.loadPreviousClassifier("resources/datasets/model/out/Classifier.bin");
        //trainingInstances = tendencyClassifier.PipeInstances(trainingInstances); //as i dont have a dataset with pos,neg,neu
        //tendencyClassifier.evaluateModel(trainingInstances,true);

        //tendencyClassifier.evaluateModel(trainingInstances,false);//if dataset labeled pos,neg,neu and u want to trace
        //tendencyClassifier.predict(testingInstances2.get(0));
        //tendencyClassifier.predictAll(testingInstances2);

        /*  User test */
        InstanceList usertweet = IOUtils.readTestdataset("test_resources/test_complex.txt",true);
        List<Instance> list = new LinkedList<>();
        list.addAll(usertweet);
        Profile user1 = new Profile(1001, list);
        double r = user1.detect_behaviour(tendencyClassifier);
        System.out.println("user tendency = "+r+" %");

    }

   
    public void editfeaturevalue()
    {
        InstanceList trainingInstances = IOUtils.readTestdataset("resources/datasets/model/train/dataset.csv"
                ,true);

        try {

            CSVParser parser = new CSVParser(new FileReader("resources/datasets/model/out/sparse2.csv"),
                    CSVFormat.EXCEL.withFirstRecordAsHeader());
            CSVPrinter writer = new CSVPrinter(new FileWriter("resources/datasets/model/out/sparse3.csv"),
                    CSVFormat.EXCEL.withHeader("wordweight", "sentiment", "topicmodelling"));

            //WordweightingEvaluator evaluator = new WordweightingEvaluator(true);
            //LabeledTopicModel tm = new LabeledTopicModel();
            SentimentAnalyser analyser = new SentimentAnalyser();

            int i = 0;
            for (CSVRecord record : parser.getRecords())
            {
                //double weight = evaluator.getWeight(trainingInstances.get(i).getData().toString());
                //double topic = tm.evaluate(trainingInstances.get(i));
                double sentiment = analyser.evaluate(trainingInstances.get(i));
                writer.printRecord(record.get("wordweight"), sentiment , record.get("topicmodelling"));
                i++;

                System.out.println("I = "+i);
                if (i % 10000 == 0)
                    break;
            }

            writer.close();

        }catch (IOException e){
            e.printStackTrace();
        }
    }
}
