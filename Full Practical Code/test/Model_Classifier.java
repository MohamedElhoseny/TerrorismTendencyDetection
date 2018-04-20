import Models.Evaluation.Evaluator;
import Models.Evaluation.visualization.malllet.ConfusionMatrix;
import Models.Sentiment_Analysis.Methodology.Trainer;
import Models.Sentiment_Analysis.Preprocessing.*;
import Models.Sentiment_Analysis.Utils.IOUtils;
import Models.Sentiment_Analysis.Utils.SVMClassifier;
import Models.Topic_Modeling.LabeledLDA_work.LabeledLDA;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.Noop;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelVector;
import cc.mallet.util.FileUtils;
import cc.mallet.util.MalletLogger;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.text.DecimalFormat;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.logging.Logger;

public class Model_Classifier
{
    Logger logger = MalletLogger.getLogger(Model_Classifier.class.getName());

    //InstanceList testingInstances2 = IOUtils.readTestdataset("resources/datasets/model/test/labeledtweets.csv", true);
    InstanceList testingInstances2 = IOUtils.readTestdataset(
            "resources/datasets/sentimentanalysis/mallet_domain/out/dataset.csv",true);


    @Test
    public void featureNB() throws IOException, ClassNotFoundException
    {
        //classifier
        Classifier classifier = Trainer.loadModel(
                "resources/datasets/sentimentanalysis/mallet_models/feature.bin");

        //prepare & preprocessor
        FeaturePreprocessor preprocessor = new FeaturePreprocessor("resources/datasets/sentimentanalysis/");
        InstanceList testing = new InstanceList(classifier.getInstancePipe());

        //read testset
        for (int i = 0; i < testingInstances2.size(); i++)
        {
            testing.addThruPipe(new Instance(preprocessor.getProcessed(testingInstances2.get(i).getData().toString())
                    ,testingInstances2.get(i).getTarget(),null,null));
            if (i == 700)
                break;
        }

        //classifying
        int i = 0;
        for (Instance tweet : testing)
        {
            LabelVector resultvector = classifier.classify(tweet).getLabelVector();
            System.out.println("_______________________________________");
            System.out.println("Tweet : "+testing.get(i++).getData());
            System.out.println("Classification Ranks : ");
            System.out.println("Class : "+resultvector.getLabelAtRank(0)+" Score : "+resultvector.getValueAtRank(0));
            System.out.println("Class : "+resultvector.getLabelAtRank(1)+" Score : "+resultvector.getValueAtRank(1));
        }

        logger.info(new ConfusionMatrix(new Trial(classifier,testing)).toString());

    }

    @Test
    public void complexNB() throws IOException, ClassNotFoundException
    {
        //classifier
        Classifier classifier = Trainer.loadModel(
                "resources/datasets/sentimentanalysis/mallet_models/complex.bin");

        //prepare & preprocessor
        TextPreprocessor preprocessor1 = new TextPreprocessor("resources/datasets/sentimentanalysis/");
        ComplexPreprocessor preprocessor2 = new ComplexPreprocessor();
        MaxentTagger tagger = new MaxentTagger("resources/datasets/sentimentanalysis/datasets/gate-EN-twitter.model");

        //read testset
        InstanceList testing = new InstanceList(classifier.getInstancePipe());
        CSVParser parser = new CSVParser(new FileReader("test_resources/test_complex.txt"),
                CSVFormat.EXCEL.withFirstRecordAsHeader());
        for (CSVRecord r : parser.getRecords())
            testing.addThruPipe(new Instance(preprocessor2.getProcessed(preprocessor1.getProcessed(r.get("tweet")),tagger),
                    r.get("target"),null,null));

        //classifying
        int i = 0;
        for (Instance tweet : testing)
        {
            LabelVector resultvector = classifier.classify(tweet).getLabelVector();
            System.out.println("_______________________________________");
            System.out.println("Tweet : "+tweet.getData());
            System.out.println("Classification Ranks : ");
            System.out.println("Class : "+resultvector.getLabelAtRank(0)+" Score : "+resultvector.getValueAtRank(0));
            System.out.println("Class : "+resultvector.getLabelAtRank(1)+" Score : "+resultvector.getValueAtRank(1));
        }


        logger.info(new ConfusionMatrix(new Trial(classifier,testing)).toString());
    }

    @Test
    public void textNB() throws IOException, ClassNotFoundException
    {
        //classifier
        Classifier classifier = Trainer.loadModel(
                "resources/datasets/sentimentanalysis/mallet_models/text.bin");

        //prepare & preprocessor
        TextPreprocessor preprocessor1 = new TextPreprocessor("resources/datasets/sentimentanalysis/");

        //read testset
        InstanceList testing = new InstanceList(classifier.getInstancePipe());
        CSVParser parser = new CSVParser(new FileReader("test_resources/test_complex.txt"),
                CSVFormat.EXCEL.withFirstRecordAsHeader());
        for (CSVRecord r : parser.getRecords())
            testing.addThruPipe(new Instance(preprocessor1.getProcessed(r.get("tweet")), r.get("target"),null,null));

        //classifying
        for (Instance tweet : testing)
        {
            LabelVector resultvector = classifier.classify(tweet).getLabelVector();
            System.out.println("_______________________________________");
            System.out.println("Tweet : "+tweet.getData());
            System.out.println("Classification Ranks : ");
            System.out.println("Class : "+resultvector.getLabelAtRank(0)+" Score : "+resultvector.getValueAtRank(0));
            System.out.println("Class : "+resultvector.getLabelAtRank(1)+" Score : "+resultvector.getValueAtRank(1));
        }

        logger.info(new ConfusionMatrix(new Trial(classifier,testing)).toString());
    }

    @Test
    public void lexiconSVM() throws IOException, ClassNotFoundException
    {
        //classifier
        SVMClassifier classifier = (SVMClassifier) Trainer.loadModel("resources/datasets/sentimentanalysis/mallet_models/lexicon.bin");

        System.out.println("targets : "+classifier.getLabelAlphabet());
        System.out.println("data : "+classifier.getAlphabet());

        //prepare & preprocessor
        TweetPreprocessor preprocessor = new TweetPreprocessor("resources/datasets/sentimentanalysis/");
        InstanceList testing = new InstanceList(new Noop(classifier.getAlphabet(),classifier.getLabelAlphabet()));

        //read testset
        for (int i = 0; i < testingInstances2.size(); i++) {
            testing.add(preprocessor.getAllInstances(testingInstances2.get(i))[3]);
            if (i == 700)
                break;
        }

        //classifying
        for (Instance tweet : testing)
        {
            LabelVector resultvector = classifier.classify(tweet).getLabelVector();
            System.out.println("__________________"+classifier.classify(tweet).getLabeling().getBestLabel()+"_____________________");
            System.out.println("Tweet : "+tweet.getData());
            System.out.println("Classification Ranks : ");
            System.out.println("Class : "+resultvector.getLabelAtRank(0)+" Score : "+resultvector.getValueAtRank(0));
            System.out.println("Class : "+resultvector.getLabelAtRank(1)+" Score : "+resultvector.getValueAtRank(1));
        }

        System.out.println(new ConfusionMatrix(new Trial(classifier,testing)).toString());
    }
}