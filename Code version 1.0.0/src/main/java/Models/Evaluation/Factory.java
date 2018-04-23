package Models.Evaluation;


import cc.mallet.types.Instance;

public interface Factory
{
    double evaluate(Instance tweet) throws Exception;
}
