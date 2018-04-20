package Models.Topic_Modeling.LabeledLDA_work;

import Models.Sentiment_Analysis.Utils.DBInstanceIterator;

public class LDAOptions
{
    //to save current state, save random seed with previous state and comment train() and make fromPrevious to true
    public final static String main_folder = "resources/datasets/topicmodelling/out";
    public final static String stopwords = "resources/stoplists/en.txt";

    //Model IO files
    public final static String outputModelFilename = main_folder+"/models/LabeledLDAModel.bin"; //The filename in which to write the binary topic model at the end of the iterations
    public final static String outputstateFile = main_folder+"/models/GibbsSamplingrateState.gz"; //The filename in which to write the Gibbs sampling state after at the end of the iterations;
    public final static String outputpipeFile = main_folder+"/models/Instance.pipe";
    public final static String outputinferencerFilename = main_folder+"/models/ModelInferencer.bin"; //A topic inferencer applies a previously trained topic model to new documents
    public final static String outputevaluatorFilename = main_folder+"/models/Modelevaluator.bin"; //A held-out likelihood evaluator for new documents.
    public final static String topicKeysFile = main_folder+"/top_words_foreach_topic.txt"; //The filename in which to write the top words for each topic and any Dirichlet parameters.
    public final static String topicWordWeightsFile = main_folder+"/alphabets_weights_foreach_topic.txt"; //The filename in which to write unnormalized weights for every topic and word type.
    public final static String wordTopicCountsFile = main_folder+"/word_SPARSE_representation.txt"; //The filename in which to write a sparse representation of topic-word assignments.
    public final static String diagnosticsFile = main_folder+"/TopicModellingDiagnostics.xml"; //The filename in which to write measures of topic quality, in XML format.
    public final static String topicReportXMLFile = main_folder+"/top_words_foreach_topic.xml"; //The filename in which to write the top words for each topic and any Dirichlet parameters in XML format.
    public final static String topicPhraseReportXMLFile = main_folder+"/top_words_Weights_foreach_topic.xml"; //The filename in which to write the top words and phrases for each topic and any Dirichlet parameters in XML format.
    public final static String testinferenceOutput = main_folder+"/test_inference.txt";
    public final static String topicDocsFile = main_folder+"/Prominent_docs_foreach_topic.txt"; //The filename in which to write the most prominent documents for each topic, at the end of the iterations.
    public final static String docTopicsFile = main_folder+"/topicproportions_perdoc.txt"; //The filename in which to write the topic proportions per document, at the end of the iterations.

    //Model parameters
    public final static Integer outputStateInterval = 500; //The number of iterations between writing the sampling state to a text file.
    public final static Integer numTopWords = 100; //The number of most probable words to print for each topic after model estimation
    public final static Integer showTopicsIntervalOption = 500; //The number of iterations between printing a brief summary of the topics so far
    public final static Integer numTopDocs = 10; //When writing topic documents, report this number of top documents."
    public final static Double  docTopicsThreshold = 0.00000000000001; //When writing topic proportions per document, do not print topics with proportions less than this threshold value.
    public final static Integer docTopicsMax = -1; //When writing topic proportions per document,do not print more than INTEGER number of topics, negative-> printall
    public final static Integer randomSeed = 10; //The random seed for the Gibbs sampler.  Default is 0, which will use the clock
    public final static Integer numIterationsOption = 10000; //The number of iterations of Gibbs sampling
    public static boolean fromPrevious = true; //The filename from which to read the binary topic model, null -> indicating that no file will be read.
    public static boolean fromPreviousState = true; //"The filename from which to read the gzipped Gibbs sampling state created by --output-state, The original input file must be included, using --input.

    //Hyperparameters
    public static Double alphaOption = 0.1; //Alpha parameter: smoothing over doc topic distribution (NOT the sum over topics)
    public static Double betaOption = 0.1; //Beta parameter: smoothing over word distributions.

    //Last Good INPUT  :  2.0 , 0.1
    /*
        Symmetric Distribution
            If you don’t know whether your LDA distribution is symmetric or asymmetric,
            it’s most likely symmetric. Here,
            # alpha represents document-topic density
             - higher alpha, documents are made up of more topics,
             - lower alpha, documents contain fewer topics.

            # Beta represents topic-word density
            - high beta, topics are made up of most of the words in the corpus,
            - low beta they consist of few words.

        Asymmetric Distribution
            Asymmetric distributions are similar, but slightly different: higher alpha results
            in a more specific topic distribution per document. Likewise, beta results in a more
            specific word distribution per topic.

         -> the topic distributions (theta) and the topic-word distributions (phi) respectively

         -> Appropriate values for ALPHA and BETA depend on the number of topics and the number of words in vocabulary.
          For most applications, good results can be obtained by setting ALPHA = 50 / T and BETA = 200 / W
         -> alpha = 2.5 (5/2) beta = 0.01   'can get efficient results'
     */
}
