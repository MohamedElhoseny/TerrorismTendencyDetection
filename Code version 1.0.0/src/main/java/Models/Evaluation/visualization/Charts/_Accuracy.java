package Models.Evaluation.visualization.Charts;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.paint.Color;

import java.text.DecimalFormat;

public class _Accuracy
{

    @FXML
    private Label tb;

    @FXML
    private Label fb;

    @FXML
    private Label cb;

    @FXML
    private Label hc,hclc,cnb;

    @FXML
    private Label lb,lc;

    @FXML
    private Label agree;

    @FXML
    private Label disagree;

    @FXML
    private Label total;

    public void setTabeldata(double[] values) //tb,fb,cb,lb,hc,lc,agree,disagree,hc_lc,cnb,total
    {
        DecimalFormat numberFormat = new DecimalFormat("#.##");
        tb.setText(numberFormat.format(values[0])+"%");
        if (values[0] > 0.50) tb.setTextFill(Color.BLUE); else tb.setTextFill(Color.RED);

        fb.setText(numberFormat.format(values[1])+"%");
        if (values[1] > 0.50) fb.setTextFill(Color.BLUE); else fb.setTextFill(Color.RED);

        cb.setText(numberFormat.format(values[2])+"%");
        if (values[2] > 0.50) cb.setTextFill(Color.BLUE); else cb.setTextFill(Color.RED);

        lb.setText(numberFormat.format(values[3])+"%");
        if (values[3] > 0.50) lb.setTextFill(Color.BLUE); else lb.setTextFill(Color.RED);

        hc.setText(numberFormat.format(values[4])+"%");
        if (values[4] > 0.50) hc.setTextFill(Color.BLUE); else hc.setTextFill(Color.RED);

        lc.setText(numberFormat.format(values[5])+"%");
        if (values[5] > 0.50) lc.setTextFill(Color.BLUE); else lc.setTextFill(Color.RED);

        agree.setText(numberFormat.format(values[6])+"%");
        disagree.setText(numberFormat.format(values[7])+"%");

        hclc.setText(numberFormat.format(values[8])+"%");
        if (values[8] > 0.50) hclc.setTextFill(Color.BLUE); else hclc.setTextFill(Color.RED);

        cnb.setText(numberFormat.format(values[9])+"%");
        if (values[9] > 0.50) cnb.setTextFill(Color.BLUE); else cnb.setTextFill(Color.RED);

        total.setText(numberFormat.format(values[10])+"%");
        if(values[10] > 0.50) total.setTextFill(Color.BLUE); else total.setTextFill(Color.RED);
    }
}
