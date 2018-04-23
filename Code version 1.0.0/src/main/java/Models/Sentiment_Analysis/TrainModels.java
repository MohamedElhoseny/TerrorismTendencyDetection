package Models.Sentiment_Analysis;

import Models.Sentiment_Analysis.Methodology.Trainer;
import Models.Sentiment_Analysis.Preprocessing.TweetPreprocessor;
import Models.Sentiment_Analysis.Utils.DBInstanceIterator;
import Models.Sentiment_Analysis.Utils.SVMClassifier;
import Models.Sentiment_Analysis.Utils.SVMClassifierTrainer;
import cc.mallet.util.FileUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;

public class TrainModels
{
    private static String main_folder = "Resources/datasets/sentimentanalysis/";
    private static String dataset = main_folder+"mallet_domain/events.csv";

    public static void main(String[] args) throws Exception
    {
        Trainer tr = new Trainer();
        //TweetPreprocessor tweetPreprocessor = new TweetPreprocessor(main_folder);
        //tr.PrepareDataset(tweetPreprocessor, dataset);
        tr.train();

        //String testdataset = main_folder+"datasets/testdata.csv";
        //tr.PrepareTestDataset(testdataset,true);
        //tr.trainTest();


        //mergeCSV();
        //InsertOL();
    }

    public static void Breakdataset() throws IOException
    {
        BufferedReader reader = new BufferedReader(new FileReader(main_folder + "mallet_test/negative/negative.txt"));
        BufferedWriter writer = new BufferedWriter(new FileWriter(main_folder + "mallet_test/negative/dataset1.txt"));

        int count = 0;
        String str;
        int i = 1;
        while ((str = reader.readLine()) != null) {
            writer.write(str);
            count++;

            if (count % 50000 == 0) {
                writer.close();
                i++;
                writer = new BufferedWriter(new FileWriter(main_folder + "mallet_test/negative/dataset" + i + ".txt"));
            }
        }
    }
    public static void mergeCSV() throws IOException
    {
        String[] file1 = FileUtils.readFile(new File(main_folder + "mallet_domain/neg/negative.csv"));
        String[] file2 = FileUtils.readFile(new File(main_folder + "mallet_domain/pos/positive.csv"));

        BufferedWriter writer = new BufferedWriter(new FileWriter(main_folder + "mallet_domain/dataset.csv"));

        ArrayList<String> shuffleMerged = new ArrayList<>();

        Collections.addAll(shuffleMerged, file1);
        Collections.addAll(shuffleMerged, file2);
        Collections.shuffle(shuffleMerged);

        for (String line : shuffleMerged)
            writer.write(line + "\n");

        writer.close();
    }
    public static void InsertOL() throws IOException
    {
        CSVParser parser = new CSVParser(new FileReader(
                "resources/datasets/sentimentanalysis/mallet_domain/out/0L.csv"),CSVFormat.EXCEL.withFirstRecordAsHeader());
        //verb,noun,adj,adv,wordnet,polarity,target

        StringBuilder sql = new StringBuilder("INSERT INTO d_lexicon (verb,noun,adj,adv,wordnet,polarity,target) VALUES ");

        for (CSVRecord record: parser.getRecords()) {
            sql.append("(").append(Double.valueOf(record.get("verb"))).append(",")
                           .append(Double.valueOf(record.get("noun"))).append(",")
                           .append(Double.valueOf(record.get("adj"))).append(",")
                           .append(Double.valueOf(record.get("adv"))).append(",")
                           .append(Double.valueOf(record.get("wordnet"))).append(",")
                           .append(Double.valueOf(record.get("polarity"))).append(",")
                           .append("'"+record.get("target").trim()+"'").append("),\n");
        }
        sql = sql.deleteCharAt(sql.length()-2).append(";");
        DBInstanceIterator iterator = DBInstanceIterator.getInstance();
        try {
            iterator.INSERT(sql);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}