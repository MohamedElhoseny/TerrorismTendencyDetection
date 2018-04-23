package Models.Sentiment_Analysis.Utils;

import cc.mallet.pipe.Noop;
import cc.mallet.types.*;
import cc.mallet.util.IoUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.Random;

public class IOUtils
{

    //from db
    private static InstanceList readTestlabeldata(DBInstanceIterator.DBSource source) throws Exception
    {

        InstanceList instances = new InstanceList(new Noop());
        DBInstanceIterator dbInstanceIterator = DBInstanceIterator.getInstance();
        dbInstanceIterator.setDataSource(source);
        instances.addThruPipe(dbInstanceIterator);

        return instances;
    }

    //from file
    public static InstanceList readTestdataset(String pathfile, boolean islabeled)
    {
        InstanceList testInstances = null;
        if (islabeled)
            testInstances = readlabeled(pathfile);
        else
            testInstances = readunlabeled(pathfile);

        return testInstances;
    }

    //from sparse label instance for direct train & test
    public static InstanceList readSparseDataset(String csvpathfile)
    {
        BufferedReader reader;

        Alphabet dataalphabet = new Alphabet();
        dataalphabet.lookupIndex("w",true);  //weight for word weighting feature
        dataalphabet.lookupIndex("s",true);  //polarity for sentiment analysis
        dataalphabet.lookupIndex("t",true);  //topic modelling

        LabelAlphabet targetalphabet = new LabelAlphabet();
        targetalphabet.lookupLabel("negative",true);
        targetalphabet.lookupLabel("positive",true);
        targetalphabet.lookupLabel("neutral",true);

        InstanceList instances = new InstanceList(new Noop(dataalphabet,targetalphabet));
        try
        {
            reader = new BufferedReader(new FileReader(csvpathfile));
            String str = reader.readLine();
            int[] indices = new int[]{0,1,2};

            while ((str = reader.readLine()) != null)
            {
                //1.0,0:0.6556845841555697,1:0.5,2:0.5723795330352708
                String[] split = str.split(",");
                Label label;
                double[] values = new double[3];


                double labelindex = Double.valueOf(split[0]);
                if (labelindex == 1.0)
                    label = targetalphabet.lookupLabel(0);
                else if (labelindex == 2.0)
                    label = targetalphabet.lookupLabel(1);
                else
                    label = targetalphabet.lookupLabel(2);

                for (int i = 1; i < split.length; i++)
                {
                    String[] feature = split[i].split(":");
                    values[i - 1] = Double.valueOf(feature[1]);
                }

                FeatureVector fv = new FeatureVector(dataalphabet,indices,values);
                instances.addThruPipe(new Instance(fv,label,null,null));
            }


        } catch (IOException e) {
            e.printStackTrace();
        }

        return instances;
    }


    private static InstanceList readlabeled(String pathfile)
    {
        CSVParser parser = null;
        InstanceList instances = null;
        try {
            parser = new CSVParser(new FileReader(pathfile), CSVFormat.EXCEL.withFirstRecordAsHeader());
            instances = new InstanceList(new Noop());
            int i = 1;

            for (CSVRecord record : parser.getRecords())
            {
                //if (!record.get("target").equals("neutral"))
                instances.add(new Instance(record.get("tweet"), record.get("target"), "Instance" + i, null));
                //if (i % 3 == 0)
                //    break;
                i++;
            }

            System.out.println("read " + instances.size() + " instances.");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return instances;
    }
    private static InstanceList readunlabeled(String pathfile)
    {
        InstanceList tweets = new InstanceList(new Noop());
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(pathfile));
            String str;
            int i = 1;
            while ((str = reader.readLine()) != null)
            {
                if (!str.isEmpty()) tweets.add(new Instance(str.trim(),
                        "negative",
                        "Instance" + i++,
                        null));
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return tweets;
    }
}
