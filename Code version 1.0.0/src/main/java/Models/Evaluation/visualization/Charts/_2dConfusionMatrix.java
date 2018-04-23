package Models.Evaluation.visualization.Charts;

import javafx.fxml.FXML;
import javafx.scene.control.Label;

public class _2dConfusionMatrix
{

    @FXML
    private Label r1,r2,r3,r4;
    @FXML
    public Label classifiername;
    @FXML
    private Label t1,t2;

    public void setClassifiername(String name)
    {
        this.classifiername.setText(name);
    }

    public void setMatrixvalues(double[] values,int[] totals)
    {
        r1.setText(String.valueOf(values[0]));
        r2.setText(String.valueOf(values[1]));
        r3.setText(String.valueOf(values[2]));
        r4.setText(String.valueOf(values[3]));

        t1.setText(String.valueOf(totals[0]));
        t2.setText(String.valueOf(totals[1]));
    }
}
