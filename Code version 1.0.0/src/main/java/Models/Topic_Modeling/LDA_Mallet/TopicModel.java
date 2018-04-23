package Models.Topic_Modeling.LDA_Mallet;

import Models.Sentiment_Analysis.Preprocessing.TweetPreprocessor;
import Models.Sentiment_Analysis.Utils.IOUtils;
import Models.Sentiment_Analysis.Utils.TokenSequence2PorterStems;
import Models.Topic_Modeling.LabeledLDA_work.LabeledLDA;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.FileIterator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.Pattern;

public class TopicModel
{

    public static Pipe getPipe()
    {
        // Begin by importing documents from text to feature sequences
        ArrayList<Pipe> pipeList = new ArrayList<>();

        // Pipes: lowercase, tokenize, remove stopwords, map to features
        pipeList.add(new Input2CharSequence());
        pipeList.add(new CharSequenceLowercase());
        pipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        pipeList.add(new TokenSequenceRemoveStopwords(new File("resources/stoplists/en.txt"), "UTF-8", false, false, false));
        pipeList.add(new TokenSequenceRemoveNonAlpha());
        pipeList.add(new TokenSequence2PorterStems());
        pipeList.add(new TokenSequence2FeatureSequence());
        pipeList.add(new TargetStringToFeatures());
        return new SerialPipes(pipeList);
    }

    public static void main (String[] args) throws IOException
    {

        //training instances
        InstanceList training = new InstanceList(getPipe());
        FileIterator iterator = new FileIterator("resources/datasets/topicmodelling/in", FileIterator.LAST_DIRECTORY);
        training.addThruPipe(iterator);

        training.getAlphabet().stopGrowth();
        training.getDataAlphabet().stopGrowth();


        //testing instances [^_^ BOOOOOOM]
        TweetPreprocessor preprocessor = new TweetPreprocessor("resources/datasets/sentimentanalysis/");
        InstanceList in = IOUtils.readTestdataset("resources/datasets/model/test/labeledtweets.csv", true);
        InstanceList testing = new InstanceList(training.getPipe());
        for (Instance i : in)
            testing.addThruPipe(new Instance(preprocessor.startProc(i)[0].getData(),"terrorism",
                    i.getName(),i.getSource()));


        //Test Topic Modelling Algorithms :
        parallelLDA(training,testing);
        //labeledLDA(training,testing);
    }

    public static void parallelLDA(InstanceList training, InstanceList testing) throws IOException
    {
        //parallel LDA
        int numTopics = 2;
        ParallelTopicModel lda = new ParallelTopicModel (numTopics, 0.1, 0.1);
        lda.printLogLikelihood = true;
        lda.numIterations = 2000;
        lda.setTopicDisplay(500, 100);
        lda.addInstances(training);
        System.out.println("Targets : "+lda.getTopicAlphabet());
        lda.setNumThreads(7);
        lda.estimate();

        //Inference ..
        TopicInferencer inferencer = lda.getInferencer();
        inferencer.writeInferredDistributions(testing,
                new File("parallel_results.txt"),2000,1,5,0.00000000000001, 2);

    }

    public static void labeledLDA(InstanceList training, InstanceList testing) throws  IOException
    {
        LabeledLDA labeledLDA = new LabeledLDA(0.1,0.1);

        Object data = training.get(0).getData();
        if (! (data instanceof FeatureSequence)) {
            System.out.println("Topic modeling currently only supports feature sequences: ");
            System.exit(1);
        }

        System.out.println(training.getTargetAlphabet());
        labeledLDA.addInstances(training);
        System.out.println(labeledLDA.getTopicAlphabet());

        labeledLDA.setTopicDisplay(500, 100);
        labeledLDA.setNumIterations(5000);
        labeledLDA.estimate();

        // I don't want to directly inherit from ParallelTopicModel
        // because the two implementations treat the type-topic counts differently.
        // Instead, simulate a standard Parallel Topic Model by copying over the appropriate data structures.
        ParallelTopicModel topicModel = new ParallelTopicModel(labeledLDA.getTopicAlphabet(),
                labeledLDA.getAlpha() * labeledLDA.getnumofdoc(), labeledLDA.getBeta());

        topicModel.data = labeledLDA.getData();
        topicModel.alphabet = labeledLDA.getAlphabet();
        topicModel.numTypes = labeledLDA.getNumTypes();
        topicModel.betaSum = labeledLDA.getBetaSum();
        topicModel.buildInitialTypeTopicCounts();

        //Inference ..
        TopicInferencer inferencer = topicModel.getInferencer();
        inferencer.writeInferredDistributions(testing,
                new File("labeled_results2.txt"),
                2000,1,5,0.00000000000001, labeledLDA.getnumofdoc());
    }
}
