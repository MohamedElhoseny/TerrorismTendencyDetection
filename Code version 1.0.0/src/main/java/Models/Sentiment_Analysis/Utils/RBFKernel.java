package Models.Sentiment_Analysis.Utils;

import ca.uwo.csd.ai.nlp.common.SparseVector;
import ca.uwo.csd.ai.nlp.kernel.CustomKernel;
import ca.uwo.csd.ai.nlp.libsvm.svm_node;
import ca.uwo.csd.ai.nlp.libsvm.svm_parameter;

import java.io.Serializable;

public class RBFKernel implements CustomKernel,Serializable {
    svm_parameter param;

    public RBFKernel(svm_parameter param) {
        this.param = param;
    }

    public double evaluate(svm_node x, svm_node y) {
        if (x.data instanceof SparseVector && y.data instanceof SparseVector) {
            SparseVector v1 = (SparseVector)x.data;
            SparseVector v2 = (SparseVector)y.data;
            double result = 0.0D;
            int i = 0;
            int j = 0;

            SparseVector.Element e1;
            while(i < v1.size() && j < v2.size()) {
                e1 = v1.get(i);
                SparseVector.Element e2 = v2.get(j);
                if (e1.index == e2.index) {
                    double d = e1.value - e2.value;
                    result += d * d;
                    ++i;
                    ++j;
                } else if (e1.index < e2.index) {
                    result += e1.value * e1.value;
                    ++i;
                } else {
                    result += e2.value * e2.value;
                    ++j;
                }
            }

            while(i < v1.size()) {
                e1 = v1.get(i);
                result += e1.value * e1.value;
                ++i;
            }

            while(j < v2.size()) {
                e1 = v2.get(j);
                result += e1.value * e1.value;
                ++j;
            }

            return Math.exp(-this.param.gamma * result);
        } else {
            throw new RuntimeException("svm_nodes should contain sparse vectors.");
        }
    }
}
