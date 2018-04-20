package Models.Evaluation.visualization.Charts;

import javafx.application.Platform;
import javafx.embed.swing.SwingNode;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Label;
import javafx.scene.layout.StackPane;
import smile.plot.Palette;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.HashMap;
import java.util.ResourceBundle;

public class _PlotSVM implements Initializable
{
    @FXML
    private StackPane stackpane;
    @FXML
    private Label c1;
    @FXML
    private Label c2;
    @FXML
    private Label c3;

    @Override
    public void initialize(URL location, ResourceBundle resources)
    {
       /* stackpane.getChildren().setAll(
                getSVMConvas("Resources/datasets/model/out/train.csv",100));
        stackpane.setVisible(true);*/

        getSVMConvas("Resources/datasets/model/out/train.csv",5000);
    }

    public StackPane getSVMConvas(String trainedfile, int instancesize)
    {

        double[][] data = readTrainData(trainedfile,instancesize);

        char pointLegend;
        if (data.length < 10000)
            pointLegend = 'o';
        else
            pointLegend = '.';


        PlotCanvas canvas3d = ScatterPlot.plot(data,pointLegend);
        canvas3d.setTitle("SVM Terrorism Tendency Detection Classifier");
        canvas3d.setTitleColor(Color.RED);

        HashMap<Integer,Color> legandside = new HashMap<>();
        Color[] colors = new Color[]{Palette.COLORS[0],Palette.COLORS[1], Palette.COLORS[2],Palette.COLORS[3]};

        for (double[] instance : data)
        {
            int targetclass = (int)instance[0];
            canvas3d.point(pointLegend, colors[targetclass], instance);

            if (!legandside.containsKey(targetclass))
                legandside.put(targetclass,colors[targetclass]);
        }

        int r = legandside.get(1).getRed();
        int g = legandside.get(1).getGreen();
        int b = legandside.get(1).getBlue();

        c1.setStyle("-fx-background-color: rgb("+r+","+g+","+b+")");

         r = legandside.get(2).getRed();
         g = legandside.get(2).getGreen();
         b = legandside.get(2).getBlue();
        c2.setStyle("-fx-background-color: rgb("+r+","+g+","+b+")");

         r = legandside.get(3).getRed();
         g = legandside.get(3).getGreen();
         b = legandside.get(3).getBlue();
        c3.setStyle("-fx-background-color: rgb("+r+","+g+","+b+")");


        //StackPane pane = new StackPane();
        final SwingNode swingNode1 = new SwingNode();
        Platform.runLater(() ->
        {
            swingNode1.setContent(canvas3d);
            stackpane.getChildren().add(swingNode1);

        });

        return stackpane;
    }

    private double[][] readTrainData(String csvpathfile, int instanceSize)
    {
        BufferedReader reader = null;
        double[][] data = new double[instanceSize][3];   //3 ->  features
        try
        {
            reader = new BufferedReader(new FileReader(csvpathfile));
            String str = reader.readLine();

            int count = 0;
            while ((str = reader.readLine()) != null)
            {
                //1.0,0:0.6556845841555697,1:0.5,2:0.5723795330352708
                String[] split = str.split(",");

                //feature values
                double[] values = new double[3];
                for (int i = 0; i < split.length - 1; i++)
                {
                    if (i == 0)
                        values[i] = Double.valueOf(split[i]);
                    else
                    {
                        String[] feature = split[i].split(":");
                        values[i] = Double.valueOf(feature[1]);
                    }
                }

                data[count] = values;

                count++;

                if (count % instanceSize == 0)
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }
}
