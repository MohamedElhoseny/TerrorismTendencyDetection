package Models.Sentiment_Analysis;

import Models.Evaluation.Visulatizer;
import Models.Sentiment_Analysis.Methodology.SentimentAnalyser;
import Models.Sentiment_Analysis.Utils.DBInstanceIterator;
import Models.Sentiment_Analysis.Utils.IOUtils;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.Noop;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import javafx.application.Application;
import javafx.stage.Stage;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;
import java.util.Random;


public class Processor extends Application
{
	//the path to the "resources" folder
	public static String directory = "resources/datasets/sentimentanalysis/";
	//if set to "true", if Model fail to classify tweet, test dataset Model will try to clarify
	public static boolean usetestModel = true;
	

	@Override
	public void start(Stage primaryStage)
	{
		InstanceList testinstances = IOUtils.readTestdataset("test_resources/test_complex.txt",true);

        SentimentAnalyser analyser = new SentimentAnalyser(directory, usetestModel);
		Visulatizer visulatizer = new Visulatizer();

        Trial[] trials = analyser.evaluateClassifiers(testinstances,true);
		Map<String,Double> scores = analyser.evaluateMethodology(testinstances);
		visulatizer.VisualizeSentiment(trials,scores);
	}


	public static void main(String[] args)
	{
		launch(args);
	}
}