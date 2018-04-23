import Models.Sentiment_Analysis.Preprocessing.TextPreprocessor;
import Models.Sentiment_Analysis.Utils.TokenSequence2PorterStems;
import cc.mallet.pipe.*;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import cc.mallet.util.FileUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Model_Dataset
{
    String datasetpath = "";
    String vocabpath = "";


    public void Preparedataset() throws IOException
    {
        System.out.println("Preprocessing WordWeighting Corpus ..");
        TextPreprocessor preprocessor = new TextPreprocessor("resources/datasets/sentimentanalysis/");
        InstanceList instances = new InstanceList(getTextRepresentationPipe());

        String[] lines = FileUtils.readFile(new File(datasetpath));
        for (String line : lines)
            instances.addThruPipe(new Instance(preprocessor.getProcessed(line),
                    "", null, null));


        BufferedWriter writer = new BufferedWriter(new FileWriter("p_dataset.txt"));
        TokenSequence tokens;
        StringBuilder builder;

        //write preprocessed instances to file (for saving and fast training)
        for (Instance in : instances) {
            tokens = (TokenSequence) in.getData();
            builder = new StringBuilder();

            for (Token token : tokens)
                builder.append(token.getText()).append(" ");

            if (!builder.toString().isEmpty()) {
                writer.write(builder.toString().trim());
                writer.newLine();
            }
        }

        writer.close();
    }




    public static Pipe getTextRepresentationPipe()
    {
        ArrayList<Pipe> pipes = new ArrayList<>();
        pipes.add(new Input2CharSequence("UTF-8"));
        pipes.add(new CharSequence2TokenSequence());
        pipes.add(new TokenSequenceLowercase());
        pipes.add(new TokenSequenceRemoveStopwords(new File("Resources/stoplists/en.txt"),
                "UTF-8",false,false,false));
        pipes.add(new TokenSequenceRemoveNonAlpha());
        pipes.add(new TokenSequence2PorterStems());
        return new SerialPipes(pipes);
    }
}
