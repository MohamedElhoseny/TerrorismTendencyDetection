package Models.Sentiment_Analysis.Utils;

import ca.uwo.csd.ai.nlp.common.SparseVector;
import ca.uwo.csd.ai.nlp.kernel.CustomKernel;
import ca.uwo.csd.ai.nlp.kernel.KernelManager;
import ca.uwo.csd.ai.nlp.libsvm.ex.SVMTrainer;
import ca.uwo.csd.ai.nlp.libsvm.svm_model;
import ca.uwo.csd.ai.nlp.libsvm.svm_parameter;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Label;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class used to generate an SVMClassifier.
 * @author Syeed Ibn Faiz
 */
public class SVMClassifierTrainer extends ClassifierTrainer<SVMClassifier> implements Serializable
{

    private SVMClassifier classifier;
    private Map<String, Double> mLabel2sLabel;
    private CustomKernel kernel;
    private int numClasses;
    private boolean predictProbability;
    private svm_parameter param;

    public SVMClassifierTrainer(CustomKernel kernel) {
        this(kernel, false);
    }

    public SVMClassifierTrainer(CustomKernel kernel, boolean predictProbability) {
        super();
        mLabel2sLabel = new HashMap<>();
        this.kernel = kernel;
        this.predictProbability = predictProbability;
        init();
    }

    private void init() {
        param = new svm_parameter();
        if (predictProbability) {
            param.probability = 1;
        }
    }

    public svm_parameter getParam() {
        return param;
    }

    public void setParam(svm_parameter param) {
        this.param = param;
    }

    public CustomKernel getKernel() {
        return kernel;
    }

    public void setKernel(CustomKernel kernel) {
        this.kernel = kernel;
    }

    @Override
    public SVMClassifier getClassifier() {
        return classifier;
    }

    @Override
    public SVMClassifier train(InstanceList trainingSet)
    {
        FileWriter writer = null;
        cleanUp();
        KernelManager.setCustomKernel(kernel);
        List<ca.uwo.csd.ai.nlp.libsvm.ex.Instance> learned = getSVMInstances(trainingSet);

        //writing detail what svm learned
        try {
            writer = new FileWriter("svmTrainer.txt");
            for (ca.uwo.csd.ai.nlp.libsvm.ex.Instance x: learned)
                writer.write("Learned : "+x.getLabel()+" , "+x.getData()+"\n");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        svm_model model = SVMTrainer.train(learned, param);
        classifier = new SVMClassifier(model, kernel, mLabel2sLabel, trainingSet.getPipe(), predictProbability);
        return classifier;
    }

    public SVMClassifier trainfeatures(ca.uwo.csd.ai.nlp.libsvm.ex.Instance[] trainingset, cc.mallet.pipe.Pipe p)
    {
        cleanUp();
        KernelManager.setCustomKernel(kernel);
        svm_model model = SVMTrainer.train(trainingset,param);
        return new SVMClassifier(model,kernel,mLabel2sLabel,p,predictProbability);
    }

    private void cleanUp() {
        mLabel2sLabel.clear();
        numClasses = 0;
    }

    public List<ca.uwo.csd.ai.nlp.libsvm.ex.Instance> getSVMInstances(InstanceList instanceList) {
        List<ca.uwo.csd.ai.nlp.libsvm.ex.Instance> list = new ArrayList<>();
        for (Instance instance : instanceList) {
            SparseVector vector = getVector(instance);
            list.add(new ca.uwo.csd.ai.nlp.libsvm.ex.Instance(getLabel((Label) instance.getTarget()), vector));
        }
        return list;
    }

    private double getLabel(Label target)
    {
        Double label = mLabel2sLabel.get(target.toString());
        if (label == null) {
            numClasses++;
            label = 1.0 * numClasses;
            mLabel2sLabel.put(target.toString(), label);
        }
        return label;
    }

    public static SparseVector getVector(Instance instance)
    {
        FeatureVector fv = (FeatureVector) instance.getData();
        int[] indices = fv.getIndices();
        double[] values = fv.getValues();
        SparseVector vector = new SparseVector();
        for (int i = 0; i < indices.length; i++) {
            vector.add(indices[i], values[i]);
        }
        vector.sortByIndices();
        return vector;
    }
}
