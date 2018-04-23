package Models.Evaluation;

import Models.Evaluation.visualization.Charts.ChartsUtils;
import Models.Evaluation.visualization.malllet.ConfusionMatrix;
import cc.mallet.classify.Trial;
import cc.mallet.types.InstanceList;
import com.sun.istack.internal.NotNull;
import com.sun.istack.internal.Nullable;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.Map;

public class Visulatizer
{

    /** Responsible for visualize Sentiment analysis acurracies
     * @param trials textNB,FeatureNB,CompleNB,LexiconSVM and ClarifyNB  Trials
     */
    public void VisualizeSentiment(Trial[] ClassifierTrials, Map<String,Double> MethodologyScores)
    {
        PlotConfusionMatrix(ClassifierTrials);
        DisplayAccuracyTable(ClassifierTrials,MethodologyScores);
        PlotAccuracyLine(ClassifierTrials);
        plotPRClassifiers(ClassifierTrials);
    }

    //<editor-fold desc="Sentiment" default-state="Capsulated">
    private void plotPRClassifiers(Trial[] trials)
    {
        double[] fscores;
        double[] percision;
        double[] recall;
        String[] names = new String[]{"TextNB","FeatureNB","ComplexNB","LexiconSVM","ClarifyNB"};

        for (int i = 0; i < trials.length; i++)
        {
            fscores = new double[2];
            fscores[0] = trials[i].getF1("negative");
            fscores[1] = trials[i].getF1("positive");

            percision = new double[2];
            percision[0] = trials[i].getPrecision("negative");
            percision[1] = trials[i].getPrecision("positive");

            recall = new double[2];
            recall[0] = trials[i].getRecall("negative");
            recall[1] = trials[i].getRecall("positive");

            ChartsUtils.Displaydetails(2,names[i],fscores,percision,recall).show();
        }
    }

    private void PlotAccuracyLine(Trial[] trials)
    {
        double[] values = new double[trials.length];
        for (int i = 0; i < trials.length; i++)
            values[i] = trials[i].getAccuracy();
        ChartsUtils.DisplayAccuracyLine(values).show();
    }

    private void PlotConfusionMatrix(Trial[] trials)
    {
        int i = 0;
        String[] names = new String[]{"TextNB","FeatureNB","ComplexNB","LexiconSVM","ClarifyNB"};

        for (Trial t : trials)
        {
            ConfusionMatrix matrix = new ConfusionMatrix(t);
            System.out.println(matrix.toString()); //important

            double[] values = new double[]
                     {matrix.value(0,0), matrix.value(0,1),
                      matrix.value(1,0), matrix.value(1,1)};

            int[] totals = matrix.getTotals();

            ChartsUtils.DisplayConfusionMatrix(2,names[i++],values,totals).show();
        }
    }

    private void DisplayAccuracyTable(Trial[] trials, Map<String,Double> scores)
    {
        //tb,fb,cb,lb,hc,lc,agree,disagree,hc_lc,cnb,total
        double[] values = new double[11];
        values[0] = trials[0].getAccuracy();
        values[1] = trials[1].getAccuracy();
        values[2] = trials[2].getAccuracy();
        values[3] = trials[3].getAccuracy();
        values[4] = scores.get("hc");
        values[5] = scores.get("lc");
        values[6] = scores.get("agree");
        values[7] = scores.get("disagree");
        values[8] = scores.get("hc_lc");
        values[9] = scores.get("clarifymodel");
        values[10] = scores.get("methodology");
        ChartsUtils.DisplayAccuracyTable(values).show();
    }
    //</editor-fold>


    public void VisualizeT_Classifier(@NotNull Trial trial,@Nullable InstanceList training)
    {
        VisualizeClassifier(trial);
        if (training != null)
            VisualizeDistribution(training);
        VisualizeSVMClassifier();
    }

    //<editor-fold desc="Evaluator" default-state="Capsulated">
    private void VisualizeClassifier(Trial trial)
    {
        ConfusionMatrix matrix = new ConfusionMatrix(trial);
        System.out.println(matrix.toString()); //important

        double[] values = new double[]{matrix.value(0,0),matrix.value(0,1),matrix.value(0,2),
                                       matrix.value(1,0), matrix.value(1,1), matrix.value(1,2),
                                       matrix.value(2,0),matrix.value(2,1),matrix.value(2,2),
                                                                                        trial.getAccuracy()};

        int[] totals = matrix.getTotals();

        ChartsUtils.DisplayConfusionMatrix(3,"TerrorismTendencyClassifier",values,totals).show();

        double[] fscores = new double[]{trial.getF1("negative"),
                trial.getF1("positive"),trial.getF1("neutral")};
        double[] pr =  new double[]{trial.getPrecision("negative"), trial.getPrecision("positive"), trial.getPrecision("neutral")};
        double[] recall = new double[]{trial.getRecall("negative"), trial.getRecall("positive"), trial.getRecall("neutral")};

        ChartsUtils.Displaydetails(3,"TerrorismTendencyClassifier",fscores,pr,recall).show();
    }
    private void VisualizeDistribution(InstanceList instances)
    {
        ChartsUtils.DisplayDatasetDistribution("Terrorism Dataset Distribution",
                instances.targetLabelDistribution().value("positive"),
                instances.targetLabelDistribution().value("negative"),
                instances.targetLabelDistribution().value("neutral")).show();
    }
    private void VisualizeSVMClassifier()
    {
        //
        ChartsUtils.DisplaySVM().show();
    }
    //</editor-fold>
}
