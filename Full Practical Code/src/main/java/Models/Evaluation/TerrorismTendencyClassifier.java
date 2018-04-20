package Models.Evaluation;

import Models.Sentiment_Analysis.Utils.SVMClassifier;
import Models.Sentiment_Analysis.Utils.SVMClassifierTrainer;
import ca.uwo.csd.ai.nlp.common.SparseVector;
import ca.uwo.csd.ai.nlp.kernel.LinearKernel;
import cc.mallet.classify.Classification;
import cc.mallet.classify.Trial;
import cc.mallet.classify.evaluate.ConfusionMatrix;
import cc.mallet.pipe.Noop;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.List;

public class TerrorismTendencyClassifier implements Serializable
{
    private Alphabet dataalphabet;
    private LabelAlphabet targetalphabet;
    private InstanceList trainingInstances;

    private Evaluator featureExtraction;
    private SVMClassifierTrainer svmClassifierTrainer;
    private SVMClassifier svmClassifier;
    private Trial result;

    /**
     *  Constructing Classifier Pipe, FeatureExtractor, SVMTrainer with linearKernel
     */
    public TerrorismTendencyClassifier()
    {
        dataalphabet = new Alphabet();
        dataalphabet.lookupIndex("w",true);  //weight for word weighting feature
        dataalphabet.lookupIndex("s",true);  //polarity for sentiment analysis
        dataalphabet.lookupIndex("t",true);  //topic modelling

        targetalphabet = new LabelAlphabet();
        targetalphabet.lookupLabel("negative",true);
        targetalphabet.lookupLabel("positive",true);
        targetalphabet.lookupLabel("neutral",true);

        dataalphabet.stopGrowth();
        targetalphabet.stopGrowth();
        featureExtraction = Evaluator.getEvaluator();
        svmClassifierTrainer = new SVMClassifierTrainer(new LinearKernel(),false);
    }

    /** Responsible for passing given tweets to evaluate them and get Feature Vectors with feature results
     * for training it
     * Features Mapping :
     * ------------------
     *     wordweighting -> w
     *     sentimentanalysis -> s
     *     topicmodelling -> t
     * @param instances the tweet that SVM will train on them after doing featureExtraction on them
     */
    public void train(InstanceList instances, boolean isevaluated)
    {
        Noop predefinedPipe = new Noop(dataalphabet, targetalphabet);
        trainingInstances = new InstanceList(predefinedPipe);
        StringBuilder r = new StringBuilder();
        int[] indices = new int[]{0, 1, 2};
        int count = 0;

        if (!isevaluated)
        {
            for (Instance instance : instances)
            {
                //[0] -> wordweighting [1] -> sentimentanalysis  [2] -> topicmodelling
                double[] featuresValues = featureExtraction.getFeatureValues(instance);
                //create feature vector with methodology result
                FeatureVector vector = new FeatureVector(dataalphabet, indices, featuresValues);
                //Labelling this tweet according to Evaluation function [?]
                String evaluated = featureExtraction.classifyFeatureValues(featuresValues);
                Label target = targetalphabet.lookupLabel(evaluated);
                //Adding to the training instances after getting feature vector with its label
                trainingInstances.addThruPipe(new Instance(vector, target, instance.getName(), instance.getSource()));

                //saving
                r.append("Tweet : "+instance.getData()).append("\n");
                r.append("Class : "+instance.getTarget()).append("\n");
                r.append("Features Scores  \n------------------")
                        .append("\n\tWord-Weighting : "+featuresValues[0])
                        .append("\n\tPolarity       : "+featuresValues[1])
                        .append("\n\tTopicModelling : "+featuresValues[2]).append("\n");
                r.append("Evaluation equation Result = "+ Evaluator.applyEvaluationEquation(featuresValues)).append("\n");
                r.append("Learned as 'evaluated' = "+evaluated).append("\n");
                r.append("________________________________________________________________________\n");
                r.append("________________________________________________________________________\n");
                System.out.println("Tweet Added For Train : " + ++count);
            }
        }else
            trainingInstances.addAll(instances);


        if (!isevaluated)
        {
            try {
                FileWriter writer = new FileWriter("Resources/datasets/model/out/track.txt");
                writer.write(r.toString());
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        System.out.println("SVM starts Training on Methodology results ..%");
        svmClassifier = svmClassifierTrainer.train(trainingInstances);
        System.out.println("SVM Trained.");
    }

    /** Given a tweet instance[unlabeled], first get feature values after applying Methodology then classify using SVM
     * @param tweet Given tweet to predict its class ('positive', 'negative', 'neutral')
     * @return Object[] where [0] is the prediction score [X] evaluation Score [/], [1] tweet class prediction
     */
    public Object[] predict(Instance tweet)
    {
         double[] featureValues = featureExtraction.getFeatureValues(tweet);  //pass tweet to Methodology
         FeatureVector vector = new FeatureVector(dataalphabet,new int[]{0,1,2},featureValues); //create FeatureVector

         Classification prediction = svmClassifier.classify(new Instance(vector,tweet.getTarget(),
                 tweet.getName(),tweet.getSource()));

         String tweetclass = prediction.getLabelVector().getBestLabel().toString();
         double score = Evaluator.applyEvaluationEquation(featureValues);

         System.out.println("----------- Methodology Result ----------------");
         System.out.println("Tweet : "+tweet.getData());
         System.out.println("Tweet Target : "+tweet.getTarget());
         System.out.println("Features Scores : w = "+featureValues[0]+", s = "+featureValues[1]+", t = "+featureValues[2]);
         System.out.println("Evaluation equation Result = "+ score);
         System.out.println("Classified as : "+tweetclass);
         System.out.println("------------------------------------------------");

         return new Object[]{score,tweetclass};
    }

    /** Predict a list of unlabeled tweets and save results
     * @param testingset the test dataset contining unlabeled tweets
     */
    public void predictAll(InstanceList testingset)
    {
        double[] featureValues;
        FeatureVector featureVector;
        Classification prediction;

        StringBuilder r = new StringBuilder();

        for (Instance tweet: testingset)
        {
            featureValues = featureExtraction.getFeatureValues(tweet); //pass tweet to Methodology
            featureVector = new FeatureVector(dataalphabet,new int[]{0,1,2},featureValues); //create FeatureVector
            prediction  = svmClassifier.classify(new Instance(featureVector,tweet.getTarget(),null,null));

            LabelVector scores = prediction.getLabelVector();
            String tweetclass = prediction.getLabelVector().getBestLabel().toString();

            r.append("Tweet : "+tweet.getData()).append("\n");
            r.append("Features Scores : w = "+featureValues[0]+", s = "+featureValues[1]+", t = "+featureValues[2]).append("\n");
            r.append("Evaluation equation Result = "+ Evaluator.applyEvaluationEquation(featureValues)).append("\n");
            r.append("Classified as : "+tweetclass).append("\n");
            r.append("Classified Details \n------------------")
                    .append("\n\tNegative : "+scores.value("negative"))
                    .append("\n\tPositive : "+scores.value("positive"))
                    .append("\n\tNeutral  : "+scores.value("neutral")).append("\n");

            r.append("________________________________________________________________________\n");
            r.append("________________________________________________________________________\n");
        }

        try {
            FileWriter writer = new FileWriter("resources/datasets/model/test/results.txt");
            writer.write(r.toString());
            writer.close();
            System.out.println("Predicting saved.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //<editor-fold desc="Evaluate & Visualize" default-state="Capsulated">

    /** labelInstances must be a fresh! and classifier will lead them accross Methodology then predict them and evaluates
     * @param labelInstances given tweet labeled by their classes to Evaluate SVM Accuracy
     * @param isPreprocessed This parameter used to indicate that tweet instances vectorized by their score across
     *                       SVM pipe or not if not so SVM will pass them first to Methodology to vectorize them.
     * @throws IOException
     */
    public void evaluateModel(InstanceList labelInstances, boolean isPreprocessed)
    {
        InstanceList instances = new InstanceList(new Noop(dataalphabet,targetalphabet));
        int[] indices = new int[]{0,1,2};
        for (Instance i: labelInstances)
        {
            if (!isPreprocessed)
            {
                //[Must be Ensure that Methodology not change label of tweet under any condition]
                double[] featureValues = featureExtraction.getFeatureValues(i);
                FeatureVector vector = new FeatureVector(dataalphabet,indices,featureValues);
                Label target = targetalphabet.lookupLabel(i.getTarget()); //so tweet must labeled as pos,neg,neu

                instances.add(new Instance(vector,target,i.getName(),i.getSource()));
            }else
                instances.add(new Instance(i.getData(),i.getTarget(),i.getName(),i.getSource()));
        }

        //get The Trial [results]
        result = new Trial(svmClassifier,instances);

        System.out.println("Saving Evaluation Results ..");
        try
        {
            CSVPrinter csvwriter = new CSVPrinter(new FileWriter("Resources/datasets/model/result/R.csv"),
                    CSVFormat.EXCEL.withHeader("Tweet","Scores","True","Predicted"));

            for (int i = 0; i < labelInstances.size(); i++)
            {
                String d = "";
                if (!isPreprocessed)
                    d = labelInstances.get(i).getData().toString();  //tweet
                else
                    d = labelInstances.get(i).getName().toString();  //as its data is sparse vector

                String t = labelInstances.get(i).getTarget().toString(); //true
                String p = result.get(i).getLabeling().getBestLabel().toString();  //predicted
                FeatureVector fv = (FeatureVector) result.get(i).getInstance().getData();
                String scores = "[w = "+fv.getValues()[0]+", s = "+fv.getValues()[1]+", t = "+fv.getValues()[2]+"]";

                csvwriter.printRecord(d,scores,t,p);
            }
            csvwriter.close();
            FileWriter writer = new FileWriter("Resources/datasets/model/result/E.txt");
            writer.write(new ConfusionMatrix(result).toString());
            writer.close();

            instances.save(new File("Resources/datasets/model/result/I.bin"));
            System.out.println("Evaluation Saved.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /** The Visualize Function using to plot result after applying a new Trial on the svm Classifier
     * @param trial the trial that done on svm classifier that used to get accuracy,recall,...
     */
    public void visualizeClassifier()
    {
        System.out.println("Visualizing Results ..");
        Visulatizer visulatizer = new Visulatizer();
        visulatizer.VisualizeT_Classifier(result,trainingInstances);
    }
    //</editor-fold>

    //<editor-fold desc="Pipe" default-state="Capsulated">
    /** Responsible for labelling and vectorize tweet according Methodology specified in the Evaluator
     *  used if testing dataset not labeled and we want to evaluate system so it must be labeled
     * @param tweet Given
     * @return
     */
    public Instance Pipe(Instance tweet)
    {
        double[] featureValues = featureExtraction.getFeatureValues(tweet); //pass tweet to Methodology
        FeatureVector vector = new FeatureVector(dataalphabet,new int[]{0,1,2},featureValues);
        Label label = targetalphabet.lookupLabel(featureExtraction.classifyFeatureValues(featureValues));

        return new Instance(vector,label,tweet.getName(),tweet.getSource());
    }

    public InstanceList PipeInstances(InstanceList tweets)
    {
        InstanceList instances = new InstanceList(new Noop(dataalphabet,targetalphabet));
        for (Instance i: tweets)
            instances.add(Pipe(i));
        return instances;
    }
    //</editor-fold>

    //<editor-fold desc="Save & Load" default-state="Capsulated">
    /**  responsible for saving trainer & classifier and also sparseVector for trained Instances
     */
    public void saveModel()
    {
        try
        {
            this.save();  //save this model
            System.out.println("Terrorism-Tendency-Classifier saved.");
            this.saveSparseInstance();  //save sparse instances
            System.out.println("Trained Sparse Instances Saved.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /** Used to load previous Model in order to set it and used for predict new Instances
     * @param classifierpath path to SVM Classifier
     * @return
     */
    public void loadPreviousClassifier(String classifierpath)
    {
        try {
            ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(classifierpath));
            svmClassifier = (SVMClassifier)inputStream.readObject();
            inputStream.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    /** Used to load previous Trainer it must be called before train inorder to train
     * your previous trainer with ned=w data & this will produce anew svm classifier
     * after training which must be saved in order to load it again
     * @param trainerpath
     */
    public void loadPreviousTrainer(String trainerpath)
    {

        try {
            ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(trainerpath));
            svmClassifierTrainer = (SVMClassifierTrainer) inputStream.readObject();
            inputStream.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
    //</editor-fold>

    //<editor-fold desc="Getter & Setter" default-state="Capsulated">
    public Pipe getModelPipe()
    {
        //
        return new Noop(dataalphabet,targetalphabet);
    }
    /** Create InstanceList from sparse values [sparse.csv] then passing through evaluator
     *  to evaluate with evaluation function [after update it]
     *  Provide Fast Training as all features values found only apply equation then train it directly and trace
     * @return ready instancelist for training after applying updated evaluation function
     */
    public InstanceList getEvaluatedSparse()
    {

        InstanceList instances = new InstanceList(new Noop(dataalphabet,targetalphabet));
        int[] indices = new int[]{0,1,2};
        try
        {
            CSVParser parser = new CSVParser(new FileReader(
                    "Resources/datasets/model/out/sparse.csv"),CSVFormat.EXCEL.withFirstRecordAsHeader());

            FileWriter writer = new FileWriter("Evaluator.txt");

            int count = 0;
            for (CSVRecord record: parser.getRecords())
            {
                //double[] featuresValues = featureExtraction.getFeatureValues(instance);
                double[] featuresValues = new double[3];
                featuresValues[0] = Double.parseDouble(record.get("wordweight"));
                featuresValues[1] = Double.parseDouble(record.get("sentiment"));
                featuresValues[2] = Double.parseDouble(record.get("topicmodelling"));

                FeatureVector vector = new FeatureVector(dataalphabet, indices, featuresValues);
                //Labelling this tweet according to Evaluation function [always after new updates in equation ^_^]
                writer.write("Sending to Evaluator : \nW = "+featuresValues[0]+", S = "+featuresValues[1]+", T = "+featuresValues[2]);
                String evaluated = featureExtraction.classifyFeatureValues(featuresValues);
                Label target = targetalphabet.lookupLabel(evaluated);
                writer.write("\nEvaluator setting this values as : "+target.toString()+"\n-----------------------------");
                instances.add(new Instance(vector,target,"Instance"+(++count),null));
            }
            System.out.println("Read Sparse Instance : "+count);

        } catch (IOException e) {
            e.printStackTrace();
        }

        return instances;
    }
    public InstanceList getTrainedInstances()
    {
        //
        return this.trainingInstances;
    }
    public SVMClassifierTrainer getSvmClassifierTrainer()
    {
        //
        return this.svmClassifierTrainer;
    }
    public SVMClassifier getSvmClassifier()
    {
        //
        return this.svmClassifier;
    }
    private void saveTrainingDistribution()
    {
        try
        {
            CSVPrinter writer = new CSVPrinter(new FileWriter("resources/model/result/distribution.csv"),
                    CSVFormat.EXCEL.withHeader("Classifier,Positive,Negative,Neutral"));

            double pos =  trainingInstances.targetLabelDistribution().value("positive");
            double neg =  trainingInstances.targetLabelDistribution().value("negative");
            double neu =  trainingInstances.targetLabelDistribution().value("neutral");
            writer.printRecord("TerrorismTendencyClassifier",pos,neg,neu);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void save()
    {
        try {
            ObjectOutputStream stream = new ObjectOutputStream(new
                    FileOutputStream("Resources/datasets/model/out/Classifier.bin"));
            stream.writeObject(svmClassifier);
            stream.close();


            stream = new ObjectOutputStream(new
                    FileOutputStream("Resources/datasets/model/out/Trainer.bin"));
            stream.writeObject(svmClassifierTrainer);
            stream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void saveSparseInstance() throws IOException
    {
        CSVPrinter writer1 = new CSVPrinter(new FileWriter("Resources/datasets/model/out/train.csv"),
                CSVFormat.EXCEL.withHeader("class","tweet"));

        CSVPrinter writer2 = new CSVPrinter(new FileWriter("Resources/datasets/model/out/sparse.csv"),
                CSVFormat.EXCEL.withHeader("wordweight","sentiment","topicmodelling"));

        List<ca.uwo.csd.ai.nlp.libsvm.ex.Instance> instances = svmClassifierTrainer.getSVMInstances(trainingInstances);
        double label;
        for (ca.uwo.csd.ai.nlp.libsvm.ex.Instance instance: instances)
        {
            SparseVector vector = (SparseVector) instance.getData();
            label = instance.getLabel();
            writer1.print(label);

            for (int i = 0; i < vector.size(); i++) {
                SparseVector.Element e = vector.get(i);
                writer1.print(e.index+":"+e.value);
                writer2.print(e.value);
            }
            writer1.println();
            writer2.println();
        }

        writer1.close();
        writer2.close();
    }
    //</editor-fold>
}
