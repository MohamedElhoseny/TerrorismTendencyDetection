package Models.Topic_Modeling.LabeledLDA_work;

import Models.Sentiment_Analysis.Methodology.SentimentAnalyser;
import Models.Sentiment_Analysis.Preprocessing.TextPreprocessor;
import Models.Sentiment_Analysis.Preprocessing.TweetPreprocessor;
import Models.Sentiment_Analysis.Utils.IOUtils;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

import java.io.*;

public class test
{
    static LabeledTopicModel model = new LabeledTopicModel();

    public static void main(String[] args) throws Exception
    {
        if(!LDAOptions.fromPrevious)
            model.Train();

        InstanceList testing = IOUtils.readTestdataset("resources/datasets/model/test/labeledtweets.csv", true);
        model.EvaluateTestingSet(testing,false);




        //String folder = "Resources/datasets/topicmodelling/in/";
        //breakdataset(folder+"general/g2.txt", folder+"general/140.txt");
        //deleteWhitespace(folder+"general/140.txt",folder+"general/140.txt");
        //Trainer.repaircsv("Resources/t99.csv");
        //LabeledTopicModel.PrepareDataset("resources/t0.csv","terrorism","t10");
        //LabeledTopicModel.removeNoisyWord("terrorism","amp");
        //splitdataset("Resources/datasets/topicmodelling/in/general/140.txt",'g');


        /*double result = model.evaluate("I d probably say something like this: 'Anytime we want to talk about " +
                "probabilities  we re really integrating a density. In Bayesian analysis " +
                " a lot of the densities we come up with aren t analytically tractable: " +
                "you can only integrate them -- if you can integrate them at all -- with a great deal of suffering. ");


        result = model.evaluate("propaganda war syria isis group support is the best way to encourage people in country against " +
                "presidents");
        System.out.println(result);*/

        //splitdataset("terr.txt","resources/datasets/topicmodelling/in/terrorism/t",719);
    }

    public static String traceLDA(String tweet)
    {
        TextPreprocessor preprocessor = new TextPreprocessor(SentimentAnalyser.directory);
        tweet = preprocessor.getProcessed(tweet);   //ANY Tweet must preprocess before topic modelling & word weightning

        System.out.println(tweet);
        Instance topic1 = new Instance(tweet, "terrorism", null, null);

        double score = model.evaluate(topic1); //topicmodelling will preprocess it in its pipe & evaluate
        return "Tweet : "+tweet+"\nScore = "+score+"\n--------------------";
    }
    public static void breakdataset(String readerfile, String outputfile)
    {
        BufferedWriter writer;
        try
        {
            BufferedReader reader = new BufferedReader(new FileReader(readerfile));
            writer = new BufferedWriter(new FileWriter(outputfile));
            String str;
            int count = 0;

            while ((str = reader.readLine()) != null) {
                writer.write(str);
                writer.newLine();
                count++;
                if (count == 73344) break;
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static void deleteWhitespace(String readerfile, String outputfile)
    {
        BufferedWriter writer;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(readerfile));
            writer = new BufferedWriter(new FileWriter(outputfile));
            String str;

            while ((str = reader.readLine()) != null) {
                 if (!str.trim().isEmpty())
                     writer.write(str+"\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void splitdataset(String dataset, String filelabel, int from) throws IOException
    {
        BufferedReader reader =new BufferedReader(new FileReader(dataset));
        String str;
        int count = 0;
        int num = from;
        BufferedWriter writer;
        writer = new BufferedWriter(new FileWriter(filelabel+num+".txt"));

        while ((str = reader.readLine()) != null)
        {
            writer.write(str);
            writer.newLine();
            count++;
            if (count % 700 == 0)
            {
                writer.close();
                num++;
                writer = new BufferedWriter(new FileWriter(filelabel+num+".txt"));
            }
        }
        writer.close();
    }

    public void splitandPreparedataset(String dataset) throws IOException
    {

        BufferedReader reader =new BufferedReader(new FileReader("Resources/general.csv"));

        String str;
        int count = 0;
        int num = 1;
        boolean f = false;
        BufferedWriter writer;
        writer = new BufferedWriter(new FileWriter("temp"+num+".csv"));
        while ((str = reader.readLine()) != null)
        {
            writer.write(str);
            f = false;
            writer.newLine();
            if (count % 100000 == 0)
            {
                writer.close();
                LabeledTopicModel.PrepareDataset("temp"+num+".csv","general","g"+num);
                num++;
                writer = new BufferedWriter(new FileWriter("temp"+num+".csv"));
                f = true;
            }
            count++;
        }
        writer.close();

        if (!f)
            LabeledTopicModel.PrepareDataset("temp"+num+".csv","general","g"+num);

    }
}
