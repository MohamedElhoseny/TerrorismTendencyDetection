package Models.Topic_Modeling.LabeledLDA_work;

import Models.Sentiment_Analysis.Utils.IOUtils;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;

import static Models.Topic_Modeling.LabeledLDA_work.test.traceLDA;

public class testUI extends Application
{

    LabeledTopicModel model;
    boolean useprevious = false;
    TextArea out;

    @Override
    public void start(Stage primaryStage) throws Exception
    {

        BorderPane pane = new BorderPane();


        HBox box = new HBox(10);


        Button train = new Button("New Train");
        Button use = new Button("Use Previous");
        TextField in1 = new TextField("alpha");
        in1.setText(LDAOptions.alphaOption+"");
        TextField in2 = new TextField("beta");
        in2.setText(LDAOptions.betaOption+"");
        TextField in3 = new TextField("Input Tweet");
        Button trace = new Button("Test");
        trace.setOnAction(event -> {out.setText(traceLDA(in3.getText()));});
        out = new TextArea();
        out.setWrapText(true);

        box.getChildren().addAll(train,use,in1,in2,in3,trace);

        use.setOnAction(event ->
        {
            out.clear();
            LDAOptions.fromPrevious = true;
            model = new LabeledTopicModel();
            test();
            useprevious = false;
        });

        train.setOnAction(event ->
        {
            model = new LabeledTopicModel();
            box.setDisable(true);
            LDAOptions.fromPrevious = false;
            LDAOptions.alphaOption = Double.valueOf(in1.getText());
            LDAOptions.betaOption = Double.valueOf(in2.getText());
            run();
            useprevious = true;
            box.setDisable(false);
        });

        box.setAlignment(Pos.CENTER);
        in3.setMinWidth(400.0);
        pane.setCenter(out);
        pane.setBottom(box);
        Scene scene = new Scene(pane,1000,500);
        primaryStage.setScene(scene);
        primaryStage.show();
        primaryStage.setTitle("Tracing LDA");

    }


    public void run()
    {
        out.clear();
        if(!LDAOptions.fromPrevious) {
            try {
                LDAOptions.fromPreviousState = false;
                model.Train();
                LDAOptions.fromPreviousState = true;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        //String folder = "Resources/datasets/topicmodelling/in/";
        //breakdataset(folder+"general/g2.txt", folder+"general/140.txt");
        //deleteWhitespace(folder+"general/140.txt",folder+"general/140.txt");
        //Trainer.repaircsv("Resources/t99.csv");
        //LabeledTopicModel.PrepareDataset("resources/t0.csv","terrorism","t10");
        //LabeledTopicModel.removeNoisyWord("terrorism","amp");
        //splitdataset("Resources/datasets/topicmodelling/in/general/140.txt",'g');
    }

    public void test()
    {

      /*  String tweet = "Sport is a very important as it give as a very healthy body #train #football";

        out.setText(out.getText()+"\n"+traceLDA(tweet));
        tweet = "RT @pressfreedom #Syria #isis #anti_isis : Video: @Raqqa_SL of #Syria honored with CPJ's International Press Freedom Awards." + " #IPFA https://t.co/vSkqxO7BCV https://t";
        out.setText(out.getText()+"\n"+traceLDA(tweet));

        tweet = "In Sochi, the Iranians, Russians, and Turks ostensibly agreed on one key point: that all parties should respect Syria’s territorial integrity.";
        out.setText(out.getText()+"\n"+traceLDA(tweet));

        tweet = " The film is very Fantastic , i am very happy to see it";
        out.setText(out.getText()+"\n"+traceLDA(tweet));

        tweet = "RT @Veexmxr: I love animals but I could never turn vegan sorry meat is too bomb";
        out.setText(out.getText()+"\n"+traceLDA(tweet));
        tweet = "protesters clash with #Turkish police in #Istanbul over “hiding information " +
                "about burning Turkish soldiers by #ISIS… https://t.co/jQN5pRE6ii";
        out.setText(out.getText()+"\n"+traceLDA(tweet));
        tweet = "RT @ishaantharoor: Context: The country's Supreme Court overthrew a politically trumped up " +
                "terrorism charge on key opposition leader";
        out.setText(out.getText()+"\n"+traceLDA(tweet));
        tweet = "Syria: IS destroys part of Palmyra's Roman Theatre #StopTerror #IslamicState We must " +
                "fight! https://t.co/6sTIPTHOUS";
        out.setText(out.getText()+"\n"+traceLDA(tweet));*/

        InstanceList instances = IOUtils.readTestdataset(
                "resources/datasets/model/test/labeledtweets.csv",true);

        for (Instance i: instances) {
            out.setText(out.getText()+"\n"+traceLDA(i.getData().toString()));
        }
    }
}
