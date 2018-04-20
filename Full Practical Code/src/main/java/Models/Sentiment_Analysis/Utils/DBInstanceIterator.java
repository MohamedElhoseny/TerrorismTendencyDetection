package Models.Sentiment_Analysis.Utils;

import cc.mallet.pipe.Noop;
import cc.mallet.types.*;
import java.sql.*;
import java.util.Arrays;
import java.util.Iterator;

public class DBInstanceIterator implements Iterator<Instance>
{
    public enum DBSource
    {
        COMPLEX,
        FEATURE,
        TEXT,
        LEXICON,
        TEST, //for test dataset
        Topics_Train,
        Topics_Test;
    }

    //DB components
    Connection connection;  //connection string
    Statement statement;   //for execute query
    ResultSet instanceResults;  //Data from Instance Table

    int instancesReturned = 0;
    boolean atLeastOneMore = false; //need to check for hasnext()
    private static DBInstanceIterator dbInstanceIterator = null;

    /**
     * Steps to Create instances from this iterator
     *    1) setDataSource
     *    2) put this iterator to instancelist addthupipe Function
     */
    private DBInstanceIterator(String dbName, String user, String pass) throws SQLException
    {
        String url = "jdbc:mysql://localhost:3306/"+dbName+"?useUnicode=true&character_set_server=utf8mb4";
        connection = DriverManager.getConnection(url,user,pass);
        statement = connection.createStatement();
    }

    public static DBInstanceIterator getInstance()
    {
        if (dbInstanceIterator != null)
            return dbInstanceIterator;
        else
        {
            try {
                dbInstanceIterator = new DBInstanceIterator("ml_db","root","12131234");
            } catch (SQLException e) {
                System.out.println("Error while establish connection to db : "+e.getMessage());
            }

            return dbInstanceIterator;
        }
    }


    //Set DataSource For Reading
    public void setDataSource(DBSource source) throws Exception
    {
        instancesReturned = 0;
        switch (source)
        {
            case TEXT:
                instanceResults = statement.executeQuery("SELECT * FROM d_text ORDER BY instance_id");
                break;
            case COMPLEX:
                instanceResults = statement.executeQuery("SELECT * FROM d_complex ORDER BY instance_id");
                break;
            case FEATURE:
                instanceResults = statement.executeQuery("SELECT * FROM d_feature ORDER BY instance_id");
                break;
            case TEST:
                instanceResults = statement.executeQuery("SELECT * FROM d_test ORDER BY instance_id");
                break;
            case Topics_Train:
                instanceResults = statement.executeQuery("SELECT * FROM topics_train ORDER BY instance_id");
                break;
            case Topics_Test:
                instanceResults = statement.executeQuery("SELECT * FROM topics_test ORDER BY instance_id");
                break;
            default:
                throw new Exception("Wrong DataSource is selected !");
        }
        atLeastOneMore = instanceResults.next(); //return true if next[first] instance returned
        instancesReturned++;
    }

    //Read Instance
    public Instance next()
    {
        //Instance Fields
        Object name = null;
        Object data = null;
        Object target = null;
        Object source = null;

        try
        {
            int instanceID = instanceResults.getInt(1); //read instance_id
            name = instanceResults.getString(4);
            target = instanceResults.getString(3);
            if (target.toString().equals("?"))
                target = "positive";
            data = instanceResults.getString(2);
            source = instanceResults.getString(5);
            atLeastOneMore = instanceResults.next();

        } catch (Exception e) {
            System.err.println("problem returning instance " + instancesReturned + ": " + e.getMessage());
        }
        instancesReturned++;
        return new Instance(data, target, name, source);
    }
    public boolean hasNext()
    {
        //check for if table has one more instance that can be read
        return atLeastOneMore;
    }

    public InstanceList getInstancesFromAlphabets(DBSource source) throws Exception
    {
        switch (source)
        {
            case LEXICON:
                return getLexiconInstances();
        }

        return null;
    }

    private InstanceList getLexiconInstances() throws SQLException
    {

        Alphabet dataAlphabet = new Alphabet();
        dataAlphabet.lookupIndex("verb",true);
        dataAlphabet.lookupIndex("noun",true);
        dataAlphabet.lookupIndex("adj",true);
        dataAlphabet.lookupIndex("adv",true);
        dataAlphabet.lookupIndex("wordnet",true);
        dataAlphabet.lookupIndex("polarity",true);
        LabelAlphabet target = new LabelAlphabet();
        target.lookupLabel("negative",true);
        target.lookupLabel("positive",true);

        Noop predefinedPipe = new Noop(dataAlphabet,target);
        InstanceList instanceList = new InstanceList(predefinedPipe);
        instanceResults = statement.executeQuery("SELECT * FROM d_lexicon ORDER BY instance_id");

        int numAttributes = 6;
        while (instanceResults.next())
        {
            double values[] = {
                    instanceResults.getDouble(2), //verb
                    instanceResults.getDouble(3), //noun
                    instanceResults.getDouble(4), //adj
                    instanceResults.getDouble(5), //adv
                    instanceResults.getDouble(6), //wordnet
                    instanceResults.getDouble(7)}; //polarity

            int indices[] = new int[numAttributes];
            int count = 0;
            for (int j = 0; j < values.length;j++)
            {
                if (values[j] != 0.0)  //j != classindex to skip class value
                {                                         //values[j] != 0.0 'sparsevector'
                    values[count] = values[j]; //hna values[j] deh kda msh class attribute w feature 2l value bt3ha msh zero
                    indices[count] = j;  //kda ana b3ml deh 3shan yb2a msln indices[0] shayl index bta3 values[0] double value
                    count++; //for next feature
                }
            }

            indices = Arrays.copyOf(indices, count);
            values = Arrays.copyOf(values, count);

            FeatureVector fv = new FeatureVector(dataAlphabet, indices, values);

            String classValue = instanceResults.getString(8);
            int instancenum = instanceResults.getInt(1);

            Label classLabel = target.lookupLabel(classValue); //Target2Label  0 -> negative 1 -> positive
            Instance lexiconInstance = new Instance(fv, classLabel, "Instance"+instancenum, "DB");

            instanceList.add(lexiconInstance);//Insert Instance to InstanceList which will pass through Noop Pipe (doing nothing)
        }
        return instanceList;
    }

    //Some utils
    public void INSERT(StringBuilder sql) throws SQLException
    {
        statement.executeUpdate(sql.toString());
    }

    public ResultSet SELECT(String sql) throws SQLException
    {
        return statement.executeQuery(sql);
    }
    public void cleanup() throws Exception
    {
        String sqlState = "";

        instanceResults.close();
        statement.close();
        connection.close();
    }
    public void remove ()
    {
        //
        throw new IllegalStateException ("This Iterator<Instance> does not support remove().");
    }
}