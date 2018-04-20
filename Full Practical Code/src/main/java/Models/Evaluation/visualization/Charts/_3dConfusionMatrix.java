package Models.Evaluation.visualization.Charts;

import javafx.fxml.FXML;
import javafx.scene.control.Label;

import java.util.Collections;

public class _3dConfusionMatrix
{

    @FXML
    private Label r1,r2,r3,r4,r5,r6,r7,r8,r9;
    @FXML
    public Label classifiername;
    @FXML
    private Label t1,t2,t3;
    @FXML
    private Label trainingset;
    @FXML
    private Label accuracy;

    public void setClassifiername(String name)
    {
        classifiername.setText(name);
    }

    public void setMatrixvalues(double[] values,int[] totals)
    {
        r1.setText(String.valueOf(values[0]));
        r2.setText(String.valueOf(values[1]));
        r3.setText(String.valueOf(values[2]));
        r4.setText(String.valueOf(values[3]));
        r5.setText(String.valueOf(values[4]));
        r6.setText(String.valueOf(values[5]));
        r7.setText(String.valueOf(values[6]));
        r8.setText(String.valueOf(values[7]));
        r9.setText(String.valueOf(values[8]));
        t1.setText(String.valueOf(totals[0]));
        t2.setText(String.valueOf(totals[1]));
        t3.setText(String.valueOf(totals[2]));

        //for accuracy
        int r = 0;
        for (int x: totals)
            r+=x;

        trainingset.setText(r+"");
        accuracy.setText(String.valueOf(Math.round(values[9] * 100)));

    }
}
