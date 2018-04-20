package Models.Word_Weighting;

public class test
{
    public static void main(String[] args) throws Exception
    {
        String corpus = "Resources/datasets/wordweighting/corpus2.txt";
        WordweightingEvaluator evaluator = new WordweightingEvaluator(false);  //must pass false if i want to train
        evaluator.Train(corpus);

        //evaluator.extendCorpus("Resources/datasets/topicmodelling/in/terrorism/t9.txt", true);
        //evaluator.updateDictionary("Resources/datasets/wordweighting/vocabulary.txt");
        //WordWeightingModel model = new WordWeightingModel(WordweightingEvaluator.corpus,true);
        //model.removeWordFromCorpus("moral",false,true);
        //evaluator.Train();
        //deletelines();
/*
        WordweightingEvaluator.cleanNoisyVocabulary("resources/datasets/wordweighting/out/trainingcorpus.txt",
                "resources/datasets/wordweighting/vocab.txt");*/
    }
}
