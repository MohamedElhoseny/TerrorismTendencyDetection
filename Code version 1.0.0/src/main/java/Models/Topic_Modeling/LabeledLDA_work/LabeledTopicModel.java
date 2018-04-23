package Models.Topic_Modeling.LabeledLDA_work;

import Models.Evaluation.Factory;
import Models.Sentiment_Analysis.Preprocessing.TextPreprocessor;
import Models.Sentiment_Analysis.Utils.TokenSequence2PorterStems;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.FileIterator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.topics.TopicModelDiagnostics;
import cc.mallet.types.IDSorter;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.util.FileUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.Pattern;

public class LabeledTopicModel implements Factory,Serializable
{
    static LabeledLDA labeledLDA = null;
    static TopicInferencer topicInferencer = null;

    private TextPreprocessor preprocessor;
    private Pipe pipe = null;
    private InstanceList testing = null;

    public LabeledTopicModel() //done
    {
        //LabeledLDA : The instances must be FeatureSequence, not FeatureVector
        if (LDAOptions.fromPrevious)
        {
            try
            {
                labeledLDA = LabeledLDA.read(new File(LDAOptions.outputModelFilename));
                topicInferencer = getModelInference();  //as if i want to load previous, so i need topic Inference also to infer topics
                pipe = getInstancesPipe();
            } catch (Exception e) {
                System.out.println("Error while ready previous LDA : "+e.getMessage());
                System.exit(1);
            }
            System.out.println("LabeledLDA Loaded from binary File 100%");
        }
        else
        {
            labeledLDA = new LabeledLDA(LDAOptions.alphaOption, LDAOptions.betaOption);
            System.out.println("LabeledLDA Create a new Model with alpha = "+ LDAOptions.alphaOption+" & beta = "+ LDAOptions.betaOption);
        }

        //<editor-fold desc="setting randomSeed">
        /**
         * Why does the same LDA parameters and corpus generate different topics everytime?
         Because LDA uses randomness in both training and inference steps
         * How do i stabilize the topic generation?
         By resetting the random seed to the same value every time a model is trained or inference is performed
         */
        if (LDAOptions.randomSeed != 0)
            labeledLDA.setRandomSeed(LDAOptions.randomSeed);
        System.out.println("LabeledLDA : set RandomSeed = "+ LDAOptions.randomSeed);
        //</editor-fold>
    }

    /** Used to Train labeled LDA Note That The data must be prepared first
     *  before calling to this function ..*/
    public void Train() throws Exception //done
    {
        //<editor-fold desc="Training Dataset Input">
        //training instances
        InstanceList training = new InstanceList(getLDAPipe());
        FileIterator iterator = new FileIterator("resources/datasets/topicmodelling/in", FileIterator.LAST_DIRECTORY);
        training.addThruPipe(iterator);

        System.out.println("Instances : " + training.size());
        System.out.println("Targets : " + training.getTargetAlphabet());
        training.getDataAlphabet().dump(new PrintWriter(LDAOptions.main_folder + "/Model_Vocabulary.csv"));
        training.save(new File(LDAOptions.main_folder+"/models/trainingInstances.bin"));

        //load trainingset to LabeledLDA  Model
        training.getPipe().getDataAlphabet().stopGrowth();
        training.getPipe().getTargetAlphabet().stopGrowth();
        labeledLDA.addInstances(training);
        System.out.println("LabeledLDA vocabulary Size = " + labeledLDA.getAlphabet().size());
        //</editor-fold>

        //<editor-fold desc="read the gzipped Gibbs sampling state">
        if (LDAOptions.fromPreviousState) {
            System.out.println("Initializing from previous saved state.");
            labeledLDA.initializeFromState(new File(LDAOptions.outputstateFile));
        }
        //</editor-fold>
        //<editor-fold desc="Estimating & save LabeledLDA">
        //<editor-fold desc="setting iterations, topwords, printresultInterval">
        labeledLDA.setTopicDisplay(LDAOptions.showTopicsIntervalOption, LDAOptions.numTopWords);
        labeledLDA.setNumIterations(LDAOptions.numIterationsOption);
        //</editor-fold>
        //<editor-fold desc="LDA Estimating ..."
        if (!LDAOptions.fromPrevious) {
            System.out.println("LabeledLDA Model start estimating ...");
            labeledLDA.estimate();
        }
        //</editor-fold>
        //<editor-fold desc="Saving top words for each topic and any Dirichlet parameters to file">
        if (LDAOptions.topicKeysFile != null) {
            PrintStream out = new PrintStream(new File(LDAOptions.topicKeysFile));
            out.print(labeledLDA.topWords(LDAOptions.numTopWords));
            out.close();
            System.out.println("Topwords for each topic saved .");
        }
        //</editor-fold>
        //<editor-fold desc="Saving the binary topic model at the end of the iteration">
        if (LDAOptions.outputModelFilename != null) {
            assert (labeledLDA != null);
            try {
                System.out.println("Saving Model...");
                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(LDAOptions.outputModelFilename));
                oos.writeObject(labeledLDA);
                oos.close();
                System.out.println("LabeledLDA saved.");
                oos = new ObjectOutputStream(new FileOutputStream(LDAOptions.outputpipeFile));
                oos.writeObject(training.getPipe());
                oos.close();
                System.out.println("LDA-Pipe saved.");
            } catch (Exception e) {
                System.out.println("Couldn't write topic model to filename " + LDAOptions.outputModelFilename);
            }
        }
        //</editor-fold>
        //</editor-fold>
        generateTopicInference();
    }

    private void generateTopicInference() throws IOException  //done
    {
        //  I don't want to directly inherit from ParallelTopicModel
        //  because the two implementations treat the type-topic counts differently.
        //  Instead, simulate a standard Parallel Topic Model by copying over the appropriate data structures.
        ParallelTopicModel topicModel = new ParallelTopicModel(labeledLDA.getTopicAlphabet(),
                labeledLDA.getAlpha() * labeledLDA.getnumofdoc(), labeledLDA.getBeta());

        topicModel.data = labeledLDA.getData();
        topicModel.alphabet = labeledLDA.getAlphabet();
        topicModel.numTypes = labeledLDA.getNumTypes();
        topicModel.betaSum = labeledLDA.getBetaSum();
        topicModel.buildInitialTypeTopicCounts();


        //<editor-fold desc="saving Outputs & topicInference">
        System.out.println("Saving Statistics files ..");
        //<editor-fold desc="write measures of topic quality in XML">
        if (LDAOptions.diagnosticsFile != null) {
            PrintWriter out = new PrintWriter(LDAOptions.diagnosticsFile);
            TopicModelDiagnostics diagnostics = new TopicModelDiagnostics(topicModel, LDAOptions.numTopWords); /* ERROR  */
            out.println(diagnostics.toXML());
            out.close();
        }
        //</editor-fold>
        //<editor-fold desc="write the top words for each topic and any Dirichlet parameters in XML">
        if (LDAOptions.topicReportXMLFile != null) {
            PrintWriter out = new PrintWriter(LDAOptions.topicReportXMLFile);
            topicModel.topicXMLReport(out, LDAOptions.numTopWords);
            out.close();
        }
        //</editor-fold>
        //<editor-fold desc="write the top words and phrases for each topic and any Dirichlet parameters in XML">
        if (LDAOptions.topicPhraseReportXMLFile != null) {
            PrintWriter out = new PrintWriter(LDAOptions.topicPhraseReportXMLFile);
            topicModel.topicPhraseXMLReport(out, LDAOptions.numTopWords);
            out.close();
        }
        //</editor-fold>
        //<editor-fold desc="write the Gibbs sampling state after">
        if (LDAOptions.fromPreviousState && LDAOptions.outputStateInterval == 0)
        {
            System.out.println("Starting Writing Gibbs sampling state ..");
            topicModel.printState(new File(LDAOptions.outputstateFile));
            System.out.println("GibbsSamplestate saved .");
        }
        //</editor-fold>
        //<editor-fold desc="write the most prominent documents for each topic">
       /* if (LDAOptions.topicDocsFile != null)
        {
            PrintWriter out = new PrintWriter (new FileWriter ((new File(LDAOptions.topicDocsFile))));
            topicModel.printTopicDocuments(out, LDAOptions.numTopDocs);
            out.close();
        }*/
        //</editor-fold>
        //<editor-fold desc="write the topic proportions per document"
        if (LDAOptions.docTopicsFile != null)
        {
            PrintWriter out = new PrintWriter (new FileWriter ((new File(LDAOptions.docTopicsFile))));

            if (LDAOptions.docTopicsThreshold == 0.0) {
                topicModel.printDenseDocumentTopics(out);
            }
            else {
                topicModel.printDocumentTopics(out, LDAOptions.docTopicsThreshold, LDAOptions.docTopicsMax);
            }
            out.close();
        }
        //</editor-fold>
        //<editor-fold desc="Topic-Word weights & Word-Topic Counts">
        if (LDAOptions.topicWordWeightsFile != null) {
            //write unnormalized weights for every topic and word type
            topicModel.printTopicWordWeights(new File (LDAOptions.topicWordWeightsFile));
        }

        if (LDAOptions.wordTopicCountsFile != null) {
            //write a sparse representation of topic-word assignments
            topicModel.printTypeTopicCounts(new File (LDAOptions.wordTopicCountsFile));
        }
        //</editor-fold>
        //<editor-fold desc="Saving topic inferencer applies a previously trained topic model to new documents"
        if (LDAOptions.outputinferencerFilename != null) {
            try {
                System.out.println("Saving Model Inference ..");
                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(LDAOptions.outputinferencerFilename));
                oos.writeObject(topicModel.getInferencer());
                oos.close();
            } catch (Exception e) {
                System.out.println("Couldn't create inferencer: " + e.getMessage());
            }
        }
        //</editor-fold>
        //<editor-fold desc="Saving likelihood evaluator for new documents">
        if (LDAOptions.outputevaluatorFilename != null)
        {
            try {
                System.out.println("Saving Modelevaluator ... ");
                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(LDAOptions.outputevaluatorFilename));
                oos.writeObject(topicModel.getProbEstimator());
                oos.close();
            } catch (Exception e) {
                System.out.println("Couldn't create evaluator: " + e.getMessage());
            }
        }
        //</editor-fold>
        //</editor-fold>
    }

    private Pipe getLDAPipe() //done [Don't forget the Bomb]
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

    private TopicInferencer getModelInference() //done
    {
        TopicInferencer inference = null;
       try {
        ObjectInputStream stream = new ObjectInputStream(new FileInputStream(LDAOptions.outputinferencerFilename));
         inference = (TopicInferencer)stream.readObject();
        stream.close();
            } catch (IOException | ClassNotFoundException e) {
                e.printStackTrace();
       }
        return inference;
    }

    private Pipe getInstancesPipe() //done
    {
        Pipe pipes = null;
        try {
            ObjectInputStream stream = new ObjectInputStream(new FileInputStream(LDAOptions.outputpipeFile));
            pipes = (Pipe)stream.readObject();
            stream.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return pipes;
    }

    public void removeNoisyWord(String topic, String word) //done
    {
        File[] files = new File("Resources/datasets/topicmodelling/in/"+topic+"/").listFiles();
        int i = files.length;
        for (File file: files)
        {
          if (file.isFile())  //as it may be continue directories <3
           {
               try {
                   String[] lines = FileUtils.readFile(file);
                   StringBuilder b = new StringBuilder();
                   for (String s: lines){
                       s =  s.replaceAll(word,"");
                       b.append(s).append("\n");
                   }

                   FileWriter writer = new FileWriter(file);
                   writer.write(b.toString());
                   writer.close();
               } catch (IOException e) {
                   e.printStackTrace();
               }
           }
        }
    }

    public static void PrepareDataset(String Datasetpath,String topic,String outputfilename) throws IOException  //done [For CSV]
    {
        TextPreprocessor textfilter = new TextPreprocessor("Resources/datasets/sentimentanalysis/");
        BufferedWriter writer = new BufferedWriter(new FileWriter
                ("Resources/datasets/topicmodelling/in/"+topic+"/"+outputfilename+".txt"));

        CSVParser parser = new CSVParser(new FileReader(Datasetpath), CSVFormat.EXCEL.withFirstRecordAsHeader());

        int count = 0;
        for (CSVRecord record : parser.getRecords()) {
            String tweet;
            count++;
            try {
                tweet = record.get("tweet").trim();
            } catch (Exception e) {
                tweet = record.get(record.size() - 1);
            }

            System.out.print(".");
            if (!tweet.isEmpty()) {
                String filteredtweet = textfilter.getProcessed(tweet);  //Input tweet must be processed also
                writer.write(filteredtweet + "\n");
            }

            if(count % 1000 == 0)
                writer.flush();
        }
        writer.close();
    }

    //<editor-fold desc="Evaluation functions">
    @Override
    public double evaluate(Instance tweet)  //tweet not pass to prepare !
    {
        if (topicInferencer == null)
            topicInferencer = getModelInference();

        if (pipe == null)
            pipe = getInstancesPipe();

        if (preprocessor == null)
            preprocessor = new TextPreprocessor("resources/datasets/sentimentanalysis/");

        if (testing == null)
            testing = new InstanceList(pipe);

        Instance preparedtweet = new Instance(preprocessor.getProcessed(tweet.getData().toString()),
                "terrorism",tweet.getName(),tweet.getSource());
        testing.addThruPipe(preparedtweet);

        //Inference ..
        try {
            topicInferencer.writeInferredDistributions(testing,
                    new File(LDAOptions.testinferenceOutput),
                    2000,1,5,0.00000000000001, labeledLDA.getnumofdoc());
            testing.remove(0);

            return getResultforTopic("terrorism");
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        return 0.0;
    }

    private double getResultforTopic(String topic)  //done
    {
        //0	Instance197	1	0.5071619420006517	0	0.4928380579993483
        String[] lines = null;
        try
        {
             lines = FileUtils.readFile(new File(LDAOptions.testinferenceOutput));
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert lines != null;
        String[] tokens = lines[1].split("\t");  //6
        IDSorter[] sortedTopics = new IDSorter[labeledLDA.numTopics]; //2

        System.out.println("------------------------------------------");
        int c = 2;
        for (int i = 0; i < sortedTopics.length; i++,c++)
            sortedTopics[i] = new IDSorter(Integer.valueOf(tokens[c]), Double.valueOf(tokens[++c]));

        Arrays.sort(sortedTopics);

        double t_terr = 0.0;
        for (IDSorter sortedTopic : sortedTopics)
        {
            String t = labeledLDA.getTopicAlphabet().lookupLabel(sortedTopic.getID()).toString();  //get topic label by its id
            System.out.println("Topic : " + t + "\t" + "Score : " + sortedTopic.getWeight());

            if (t.equals(topic))  //terrorism for test
                t_terr = sortedTopic.getWeight();
        }
        System.out.println("------------------------------------------");

        return t_terr;
    }


    //used for evaluating dataset not a single tweet
    public double[] EvaluateTestingSet(InstanceList instances, boolean isprepared)  // prepare tweet issue
    {
        if (topicInferencer == null)
            topicInferencer = getModelInference();

        if (pipe == null)
            pipe = getInstancesPipe();

        if (preprocessor == null)
            preprocessor = new TextPreprocessor("resources/datasets/sentimentanalysis/");

        testing = new InstanceList(pipe);

        for (Instance tweet: instances)
            if (isprepared)
                testing.addThruPipe(new Instance(tweet.getData(),
                        "terrorism",tweet.getName(),tweet.getSource()));
            else
                testing.addThruPipe(new Instance(preprocessor.getProcessed(tweet.getData().toString()),
                        "terrorism", tweet.getName(),tweet.getSource()));

            System.out.println(testing.get(7).getData());
        //Inference ..
        try
        {
            topicInferencer.writeInferredDistributions(testing,
                        new File(LDAOptions.testinferenceOutput),
                        2000,1,5,0.00000000000001, labeledLDA.getnumofdoc());
            } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }

        double[] r = getsetResult("terrorism",testing.size());
        testing = new InstanceList(pipe);
        return r;
    }

    private double[] getsetResult(String topic, int size)   //done
    {
        double[] results = new double[size];
        try
        {
            String[] lines = FileUtils.readFile(new File(LDAOptions.testinferenceOutput));
            int count = 0;
            for (int i = 1; i < lines.length; i++)
            {
                String[] tokens = lines[i].split("\t");
                IDSorter[] sortedTopics = new IDSorter[labeledLDA.numTopics];

                System.out.println("------------------------------------------");

                int c = 2;  //first index of topic [in File]
                for (int t = 0; t < sortedTopics.length; t++, c++)
                    sortedTopics[t] = new IDSorter(Integer.valueOf(tokens[c]), Double.valueOf(tokens[++c]));

                Arrays.sort(sortedTopics);

                double t_terr = 0.0;
                for (IDSorter sortedTopic : sortedTopics)
                {
                    String t = labeledLDA.getTopicAlphabet().lookupLabel(sortedTopic.getID()).toString();
                    System.out.println("Topic : " + t + "\t" + "Score : " + sortedTopic.getWeight());

                    if (t.equals(topic))
                        t_terr = sortedTopic.getWeight();   //return weight for specific topic
                }

                System.out.println("------------------------------------------");
                results[count++] = t_terr;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return results;
    }
    //</editor-fold>
}
