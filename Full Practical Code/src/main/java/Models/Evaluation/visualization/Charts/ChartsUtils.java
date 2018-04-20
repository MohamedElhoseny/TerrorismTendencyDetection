package Models.Evaluation.visualization.Charts;

import Models.Evaluation.visualization.jsatfx.Plot;
import cc.mallet.util.FileUtils;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.io.CSV;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class ChartsUtils
{

    public static Stage DisplayDatasetDistribution(String title,double pos, double neg, double neu)
    {
        Stage window = new Stage();
        BorderPane layout = new BorderPane();

        //Model Data
        ObservableList<PieChart.Data> pieChartData = FXCollections.observableArrayList();

        PieChart.Data data1 = new PieChart.Data("Positive",pos);
        PieChart.Data data2 = new PieChart.Data("Negative",neg);
        PieChart.Data data3 = new PieChart.Data("Neutral",neu);
        pieChartData.addAll(data1, data2, data3);

        //Create Chart
        PieChart pieChart = new PieChart(pieChartData);
        pieChart.setTitle(title);

        layout.setCenter(pieChart);
        Scene scene = new Scene(layout,500,500);
        window.setScene(scene);

        String css = ChartsUtils.class.getResource("charts.css").toExternalForm();
        scene.getStylesheets().add(css);
        return window;
    }

    public static Stage Displaydetails(int dimension,String classifiername,
                                        double[] fscores,double[] percisions,double[] recalls)
    {
        //textnb, feature, complex, lexicon

        //Defining the x axis
        CategoryAxis xAxis = new CategoryAxis();

        xAxis.setCategories(FXCollections.observableArrayList(Arrays.asList("F-Score", "Percision" , "Recall")));
        xAxis.setLabel("category");

        //Defining the y axis
        NumberAxis yAxis = new NumberAxis(0.0,1,0.1); //without specifying parameters, it will specify it according inputs
        yAxis.setLabel("score");

        //Prepare XYChart.Series objects by setting data
        XYChart.Series<String, Number> series2 = new XYChart.Series<>();
        series2.setName("Negative");
        series2.getData().add(new XYChart.Data<>("F-Score", fscores[0]));
        series2.getData().add(new XYChart.Data<>("Percision", percisions[0]));
        series2.getData().add(new XYChart.Data<>("Recall", recalls[0]));

        XYChart.Series<String, Number> series1 = new XYChart.Series<>();
        series1.setName("Positive");
        series1.getData().add(new XYChart.Data<>("F-Score", fscores[1]));
        series1.getData().add(new XYChart.Data<>("Percision", percisions[1]));
        series1.getData().add(new XYChart.Data<>("Recall", recalls[1]));

        //Creating the Bar chart
        BarChart<String, Number> barChart = new BarChart<>(xAxis, yAxis);
        barChart.setTitle("Accuracy representation for "+classifiername);

        //Setting the data to bar chart
        barChart.getData().addAll(series1, series2);



        if (dimension == 3)
        {
            barChart.setTitle(classifiername);
            XYChart.Series<String, Number> series3 = new XYChart.Series<>();
            series3.setName("Neutral");
            series3.getData().add(new XYChart.Data<>("F-Score", fscores[2]));
            series3.getData().add(new XYChart.Data<>("Percision", percisions[2]));
            series3.getData().add(new XYChart.Data<>("Recall", recalls[2]));
            barChart.getData().add(series3);
        }


        Stage primaryStage = new Stage();
        BorderPane layout = new BorderPane();
        layout.setCenter(barChart);
        Scene scene = new Scene(layout,300,300);
        String css = ChartsUtils.class.getResource("charts.css").toExternalForm();
        scene.getStylesheets().add(css);
        primaryStage.setScene(scene);

        return primaryStage;
    }

    public static Stage DisplayAccuracyLine(double[] accur)
    {
        //Defining X axis , y axis  Types
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Accuracy");

        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setCategories(FXCollections.observableArrayList(Arrays.asList(
                "TextNB", "FeatureNB" , "ComplexNB", "LexiconSVM","TestNB")));
        xAxis.setLabel("Classifier");


        //prepare series
        XYChart.Series series1 = new XYChart.Series();
        series1.setName("Accuracy of Sentiment analysis Classifiers");
        series1.getData().add(new XYChart.Data("TextNB", accur[0]));
        series1.getData().add(new XYChart.Data("FeatureNB", accur[1]));
        series1.getData().add(new XYChart.Data("ComplexNB", accur[2]));
        series1.getData().add(new XYChart.Data("LexiconSVM", accur[3]));
        series1.getData().add(new XYChart.Data("TestNB", accur[4]));

        //create and assign Data
        LineChart lineChart = new LineChart(xAxis,yAxis);
        lineChart.getData().addAll(series1);

        Stage primaryStage = new Stage();
        BorderPane layout = new BorderPane();
        layout.setCenter(lineChart);
        Scene scene = new Scene(layout,300,300);
        String css = ChartsUtils.class.getResource("charts.css").toExternalForm();
        scene.getStylesheets().add(css);
        primaryStage.setScene(scene);

        return primaryStage;
    }

    public static Stage DisplayUniBidatails(double[] unigram, double[] bigram)
    {
        //Defining the X axis
        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setLabel("Classifier");
        //Defining the y Axis
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Accuracy");

        //Creating the Area chart
        AreaChart<String, Number> areaChart = new AreaChart(xAxis, yAxis);

        //set title
        areaChart.setTitle("Classifiers accuracy in Unigram & Bigram");

        //Prepare XYChart.Series objects by setting data
        XYChart.Series series1 = new XYChart.Series();

        series1.setName("Unigram");
        series1.getData().add(new XYChart.Data("TextNB", unigram[0]));
        series1.getData().add(new XYChart.Data("FeatureNB", unigram[1]));
        series1.getData().add(new XYChart.Data("ComplexNB", unigram[2]));
        series1.getData().add(new XYChart.Data("LexiconSVM", unigram[3]));
        series1.getData().add(new XYChart.Data("TestNB", unigram[4]));

        XYChart.Series series2 = new XYChart.Series();
        series2.setName("Bigram");
        series2.getData().add(new XYChart.Data("TextNB", bigram[0]));
        series2.getData().add(new XYChart.Data("FeatureNB", bigram[1]));
        series2.getData().add(new XYChart.Data("ComplexNB", bigram[2]));
        series2.getData().add(new XYChart.Data("LexiconSVM", bigram[3]));
        series2.getData().add(new XYChart.Data("TestNB", bigram[4]));

        //Setting the XYChart.Series objects to area chart
        areaChart.getData().addAll(series1,series2);


        Stage primaryStage = new Stage();
        BorderPane layout = new BorderPane();
        layout.setCenter(areaChart);
        Scene scene = new Scene(layout,300,300);
        String css = ChartsUtils.class.getResource("charts.css").toExternalForm();
        scene.getStylesheets().add(css);
        primaryStage.setScene(scene);

        return primaryStage;
    }

    private static File reformatEvaluatorResult(File csvfile)
    {
        File file = new File(csvfile.getParent()+"/reformatedtrain.txt");
        try {
            String[] lines = FileUtils.readFile(csvfile);
            StringBuilder n_lines = new StringBuilder();

            for (int i = 1; i < lines.length; i++)
                n_lines.append(lines[i].replaceAll("[0-9][:]","")).append("\n");

            FileWriter writer = new FileWriter(file);
            writer.write(n_lines.toString());
            writer.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
        return file;
    }

    public static Stage plotSVM(File datafile)
    {
        ScatterChart<Number, Number> plot = null;
        File f = reformatEvaluatorResult(datafile);

        try {
            Set<Integer> featureIndices = new HashSet<>();
            featureIndices.add(0);
            DataSet dataSet1 = CSV.read(new FileReader(f), ',', featureIndices);
            ClassificationDataSet data = new ClassificationDataSet(dataSet1, 0);
            plot = Plot.scatterC(data);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Stage primaryStage = new Stage();
        BorderPane layout = new BorderPane();
        layout.setCenter(plot);
        Scene scene = new Scene(layout,500,400);
        String css = ChartsUtils.class.getResource("charts.css").toExternalForm();
        scene.getStylesheets().add(css);
        primaryStage.setScene(scene);

        return primaryStage;
    }

    public static Stage DisplayConfusionMatrix(int dimension,String classifiername, double[] values,int[] totals)
    {
        Stage stage = new Stage();
        try
        {
            if (dimension == 3)
            {
                FXMLLoader loader = new FXMLLoader(ChartsUtils.class.getResource("confusion3d.fxml"));
                stage = new Stage(StageStyle.DECORATED);
                stage.setScene(new Scene(loader.load()));
                _3dConfusionMatrix controller = loader.getController();
                controller.setClassifiername(classifiername);
                controller.setMatrixvalues(values,totals);
            }else if (dimension == 2){
                FXMLLoader loader = new FXMLLoader(ChartsUtils.class.getResource("confusion2d.fxml"));
                stage = new Stage(StageStyle.DECORATED);
                stage.setScene(new Scene(loader.load()));
                _2dConfusionMatrix controller = loader.getController();
                controller.setClassifiername(classifiername);
                controller.setMatrixvalues(values,totals);
            }


        } catch (IOException e) {
            e.printStackTrace();
        }
        return stage;
    }

    public static Stage DisplayAccuracyTable(double[] values)
    {
        Stage stage = new Stage();
        try {
            FXMLLoader loader = new FXMLLoader(ChartsUtils.class.getResource("accuracytable.fxml"));
            stage = new Stage(StageStyle.DECORATED);
            stage.setScene(new Scene(loader.load()));
            _Accuracy controller = loader.getController();
            controller.setTabeldata(values);

        } catch (IOException e) {
            e.printStackTrace();
        }
        return stage;
    }

    public static Stage DisplaySVM()
    {
        Stage stage = new Stage();
        try {
            Parent root = FXMLLoader.load(ChartsUtils.class.getResource("PlotSVM.fxml"));
            stage.setScene(new Scene(root));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return stage;
    }
}
