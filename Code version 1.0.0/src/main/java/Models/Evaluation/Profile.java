package Models.Evaluation;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.util.FileUtils;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class Profile
{
    boolean isSuspected;
    int userid;
    List<Instance> tweets;

    public Profile(int userid, List<Instance> tweets)
    {
        this.userid = userid;
        this.tweets = tweets;
    }

    public double detect_behaviour(TerrorismTendencyClassifier classifier)
    {
        StringBuilder builder = new StringBuilder();
        HashMap<String,List<Double>> tendencyoccurrence = new HashMap(3){{
          put("negative", new LinkedList<>()); put("positive",new LinkedList<>()); put("neutral",new LinkedList<>());
        }};

        for (Instance tweet: tweets)
        {
            Object[] out = classifier.predict(tweet);
            tendencyoccurrence.get(out[1].toString()).add((Double) out[0]);
            builder.append("Tweet : "+tweet.getData().toString()+"\nClass : "+out[1]+", Score : "+out[0]+"\n----------\n");
        }

        double NumNeg, NumPos, NumNorm, MaxNeg=0.0, MaxPos=0.0, MaxNorm=0.0, percentage=0.0;

        //Getting Number of positive,negative,normal tweets
        NumNeg  = (double) tendencyoccurrence.get("negative").size()  / tweets.size();
        NumPos  = (double) tendencyoccurrence.get("positive").size() / tweets.size();
        NumNorm = (double) tendencyoccurrence.get("neutral").size() / tweets.size();

        //Getting Max score for each class
        if (!tendencyoccurrence.get("negative").isEmpty())
            MaxNeg = Collections.max(tendencyoccurrence.get("negative")) * 0.4;
        if (!tendencyoccurrence.get("positive").isEmpty())
            MaxPos = Collections.max(tendencyoccurrence.get("positive")) * 0.4;
        if (!tendencyoccurrence.get("neutral").isEmpty())
            MaxNorm = Collections.max(tendencyoccurrence.get("neutral")) * 0.4;

        builder.append("___________ Results ____________\nNumNeg = "+NumNeg+", NumPos = "+NumPos+", NumNorm = "+NumNorm+"\n");
        builder.append("MaxNeg = "+MaxNeg+", MaxPos = "+MaxPos+", MaxNorm = "+MaxNorm+"\n");


        if(NumNorm == 0.0)
        {
            if(NumNeg == 0.0)
                percentage = (Collections.min(tendencyoccurrence.get("positive")) - 0.9);
            else
                percentage = ((NumNeg * MaxNeg) - (NumPos * MaxPos));
        }else
            percentage = ((NumNeg * MaxNeg) + (NumPos * MaxPos)) - (NumNorm * MaxNorm);

        //Applying absolute
        percentage = Math.abs(percentage * 100);

        builder.append("Tendency Percentage : "+Math.round(percentage)+" %");
        try {
            FileWriter writer = new FileWriter("resources/output/user_1001.txt");
            writer.write(builder.toString()); writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return Math.round(percentage);
    }
}
