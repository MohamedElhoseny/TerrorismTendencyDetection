package Models.Topic_Modeling.LabeledLDA_work;

import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.*;
import cc.mallet.util.MalletLogger;
import cc.mallet.util.Randoms;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * LabeledLDA
 * @author David Mimno
 */
public class LabeledLDA implements Serializable
{
	protected static Logger logger = MalletLogger.getLogger(LabeledLDA.class.getName());
	// the training instances and their topic assignments
	protected ArrayList<TopicAssignment> data;
	// the alphabet for the input data
	protected Alphabet alphabet;
	// this alphabet stores the string meanings of the labels/topics
	protected Alphabet labelAlphabet;
	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet;
	// The number of topics requested
	protected int numTopics;
	// The size of the vocabulary
	protected int numTypes;
	// Prior parameters
	protected double alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
	protected double beta;   // Prior on per-topic multinomial distribution over words
	protected double betaSum;
	public static final double DEFAULT_BETA = 0.01;
	// An array to put the topic counts for the current document. 
	// Initialized locally below.  Defined here to avoid
	// garbage collection overhead.
	protected int[] oneDocTopicCounts; // indexed by <document index, topic index>
	// Statistics needed for sampling.
	protected int[][] typeTopicCounts; // indexed by <feature index, topic index>
	protected int[] tokensPerTopic; // indexed by <topic index>
	public int numIterations = 1000;
	public int showTopicsInterval = 50;
	public int wordsPerTopic = 10;
	protected Randoms random;
	protected boolean printLogLikelihood = false;
	
	public LabeledLDA(double alpha, double beta)
	{
		this.data = new ArrayList<>();
		this.alpha = alpha;
		this.beta = beta;
		this.random = new Randoms();
		logger.info("Labeled LDA");
	}

	/** Getter */
	public Alphabet getAlphabet(){ return alphabet; }
	public LabelAlphabet getTopicAlphabet(){ return topicAlphabet; }
	public double getAlpha(){return alpha;}
	public int getnumofdoc(){ return numTopics;}
	public double getBeta(){return  beta;}
	public int getNumTypes(){ return numTypes;}
	public double getBetaSum(){return  betaSum;}
	public ArrayList<TopicAssignment> getData(){ return data; }
	public int[][] getTypeTopicCounts(){ return typeTopicCounts; }
	public int[] getTopicTotals(){ return tokensPerTopic; }

	/** Setter */
	public void setTopicDisplay(int interval, int n) {
		this.showTopicsInterval = interval;
		this.wordsPerTopic = n;
	}
	public void setRandomSeed(int seed) {
		random = new Randoms(seed);
	}
	public void setNumIterations (int numIterations) {
		this.numIterations = numIterations;
	}


	/** setting alphabets to LDA and for each instance in training set and randomly assign to topic
	 * @param training instancelist training dataset used to initialize LDA
	 */
	public void addInstances (InstanceList training)
	{
		System.out.println("LabeledLDA Model : Adding training instances ..%");
		//Data Alphabet
		alphabet = training.getDataAlphabet();
		numTypes = alphabet.size();
		betaSum = beta * numTypes;
		
		// We have one topic for every possible label.
		labelAlphabet = training.getTargetAlphabet();
		numTopics = labelAlphabet.size();
		oneDocTopicCounts = new int[numTopics];
		tokensPerTopic = new int[numTopics];
		typeTopicCounts = new int[numTypes][numTopics];

		//topicAlphabet = AlphabetFactory.labelAlphabetOfSize(numTopics); //generate d0,d1 [3shan target string w hwa 3wzo label]
        topicAlphabet = new LabelAlphabet();
		for (int i = 0; i < numTopics; i++) {
			String target = labelAlphabet.lookupObject(i).toString();
           topicAlphabet.lookupLabel(target);
		}
        //System.out.println(topicAlphabet);

        for (Instance instance : training)
		{
			FeatureSequence tokens = (FeatureSequence) instance.getData(); //read data featureSequence
			LabelSequence topicSequence = new LabelSequence(topicAlphabet, new int[ tokens.size() ]);
            FeatureVector labels = (FeatureVector) instance.getTarget(); //read target featurevector

			int[] topics = topicSequence.getFeatures();
			for (int position = 0; position < tokens.size(); position++)
			{
				int topic = labels.indexAtLocation(random.nextInt(labels.numLocations()));
				topics[position] = topic;
				tokensPerTopic[topic]++;
				int type = tokens.getIndexAtPosition(position);
				typeTopicCounts[type][topic]++;
			}
			TopicAssignment t = new TopicAssignment(instance, topicSequence);
			data.add(t);
		}
		/*
            for (Instance instance : training)
            {
                FeatureSequence tokens = (FeatureSequence) instance.getData();
                LabelSequence topicSequence = new LabelSequence(topicAlphabet, new int[ tokens.size() ]);

                int[] topics = topicSequence.getFeatures();
                for (int position = 0; position < topics.length; position++) {
                    int topic = random.nextInt(numTopics);
                    topics[position] = topic;
                }
                TopicAssignment t = new TopicAssignment(instance, topicSequence);
                data.add(t);
            }

            buildInitialTypeTopicCounts();
            initializeHistograms();
		 */
		System.out.println("Data loaded.");
	}
	public void initializeFromState(File stateFile) throws IOException
	{
		String line;
		String[] fields;

		BufferedReader reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(stateFile))));
		line = reader.readLine();

		// Skip some lines starting with "#" that describe the format and specify hyperparameters
		while (line.startsWith("#")) {
			line = reader.readLine();
		}
		
		fields = line.split(" ");

		for (TopicAssignment document: data)
		{
			FeatureSequence tokens = (FeatureSequence) document.instance.getData();
			FeatureSequence topicSequence =  (FeatureSequence) document.topicSequence;

			int[] topics = topicSequence.getFeatures();
			for (int position = 0; position < tokens.size(); position++) {
				int type = tokens.getIndexAtPosition(position);
				
				if (type == Integer.parseInt(fields[3])) {
					int topic = Integer.parseInt(fields[5]);
					topics[position] = topic;
					
					// This is the difference between the dense type-topic representation used here
					//  and the sparse version used in ParallelTopicModel.
					typeTopicCounts[type][topic]++;
				}
				else {
					System.err.println("instance list and state do not match: " + line);
					throw new IllegalStateException();
				}

				line = reader.readLine();
				if (line != null) {
					fields = line.split(" ");
				}
			}
		}
	}
	public void estimate()
	{
		for (int iteration = 1; iteration <= numIterations; iteration++)
		{
			//long iterationStart = System.currentTimeMillis();

			// Loop over every document in the corpus
			for (TopicAssignment aData : data)
			{
				FeatureSequence tokenSequence = (FeatureSequence) aData.instance.getData();
				FeatureVector labels = (FeatureVector) aData.instance.getTarget();
				LabelSequence topicSequence = (LabelSequence) aData.topicSequence;
				sampleTopicsForOneDoc(tokenSequence, labels, topicSequence);
			}
		
            //long elapsedMillis = System.currentTimeMillis() - iterationStart;
			//logger.info(iteration + "\t" + elapsedMillis + "ms\t");

			// Occasionally print more information
			if (showTopicsInterval != 0 && iteration % showTopicsInterval == 0)
				logger.info("<" + iteration + "> Log Likelihood: " + modelLogLikelihood() + "\n" +
							topWords (wordsPerTopic));
		}
	}
	public void sampleTopicsForOneDoc(FeatureSequence tokenSequence, FeatureVector labels, FeatureSequence topicSequence)
	{

		int[] possibleTopics = labels.getIndices();
		int numLabels = labels.numLocations();

		int[] oneDocTopics = topicSequence.getFeatures();

		int[] currentTypeTopicCounts;
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLength = tokenSequence.getLength();

		int[] localTopicCounts = new int[numTopics];

		//		populate topic counts
		for (int position = 0; position < docLength; position++) {
			localTopicCounts[oneDocTopics[position]]++;
		}

		double score, sum;
		double[] topicTermScores = new double[numLabels];

		//	Iterate over the positions (words) in the document 
		for (int position = 0; position < docLength; position++) {
			type = tokenSequence.getIndexAtPosition(position);
			oldTopic = oneDocTopics[position];

			// Grab the relevant row from our two-dimensional array
			currentTypeTopicCounts = typeTopicCounts[type];

			//	Remove this token from all counts. 
			localTopicCounts[oldTopic]--;
			tokensPerTopic[oldTopic]--;
			assert(tokensPerTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
			currentTypeTopicCounts[oldTopic]--;

			// Now calculate and add up the scores for each topic for this word
			sum = 0.0;
			
			// Here's where the math happens! Note that overall performance is 
			//  dominated by what you do in this loop.
			for (int labelPosition = 0; labelPosition < numLabels; labelPosition++) {
				int topic = possibleTopics[labelPosition];
				score =
					(alpha + localTopicCounts[topic]) *
					((beta + currentTypeTopicCounts[topic]) /
					 (betaSum + tokensPerTopic[topic]));
				sum += score;
				topicTermScores[labelPosition] = score;
			}
			
			// Choose a random point between 0 and the sum of all topic scores
			double sample = random.nextUniform() * sum;

			// Figure out which topic contains that point
			int labelPosition = -1;
			while (sample > 0.0) {
				labelPosition++;
				sample -= topicTermScores[labelPosition];
			}

			// Make sure we actually sampled a topic
			if (labelPosition == -1) {
				throw new IllegalStateException ("LabeledLDA: New topic not sampled.");
			}

			newTopic = possibleTopics[labelPosition];

			// Put that new topic into the counts
			oneDocTopics[position] = newTopic;
			localTopicCounts[newTopic]++;
			tokensPerTopic[newTopic]++;
			currentTypeTopicCounts[newTopic]++;
		}
	}
	public double modelLogLikelihood()
	{
		double logLikelihood = 0.0;
		int nonZeroTopics;
		// The likelihood of the model is a combination of a 
		// Dirichlet-multinomial for the words in each topic
		// and a Dirichlet-multinomial for the topics in each
		// document.

		// The likelihood function of a dirichlet multinomial is
		//	 Gamma( sum_i alpha_i )	 prod_i Gamma( alpha_i + N_i )
		//	prod_i Gamma( alpha_i )	  Gamma( sum_i (alpha_i + N_i) )

		// So the log likelihood is 
		//	logGamma ( sum_i alpha_i ) - logGamma ( sum_i (alpha_i + N_i) ) + 
		//	 sum_i [ logGamma( alpha_i + N_i) - logGamma( alpha_i ) ]

		// Do the documents first

		int[] topicCounts = new int[numTopics];
		double[] topicLogGammas = new double[numTopics];
		int[] docTopics;

		for (int topic=0; topic < numTopics; topic++) {
			topicLogGammas[ topic ] = Dirichlet.logGamma( alpha );
		}
	
		for (int doc=0; doc < data.size(); doc++) {
			LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
			FeatureVector labels = (FeatureVector) data.get(doc).instance.getTarget();
			
			docTopics = topicSequence.getFeatures();

			for (int token=0; token < docTopics.length; token++) {
				topicCounts[ docTopics[token] ]++;
			}

			for (int topic=0; topic < numTopics; topic++) {
				if (topicCounts[topic] > 0) {
					logLikelihood += (Dirichlet.logGamma(alpha + topicCounts[topic]) -
									  topicLogGammas[ topic ]);
				}
			}

			// add the parameter sum term
			logLikelihood += Dirichlet.logGamma(alpha * labels.numLocations());

			// subtract the (count + parameter) sum term
			logLikelihood -= Dirichlet.logGamma(alpha * labels.numLocations() + docTopics.length);

			Arrays.fill(topicCounts, 0);
		}
	
		// And the topics

		// Count the number of type-topic pairs
		int nonZeroTypeTopics = 0;

		for (int type=0; type < numTypes; type++) {
			// reuse this array as a pointer

			topicCounts = typeTopicCounts[type];

			for (int topic = 0; topic < numTopics; topic++) {
				if (topicCounts[topic] == 0) { continue; }
				
				nonZeroTypeTopics++;
				logLikelihood += Dirichlet.logGamma(beta + topicCounts[topic]);

				if (Double.isNaN(logLikelihood)) {
					System.out.println(topicCounts[topic]);
					System.exit(1);
				}
			}
		}
	
		for (int topic=0; topic < numTopics; topic++) {
			logLikelihood -= 
				Dirichlet.logGamma( (beta * numTopics) +
											tokensPerTopic[ topic ] );
			if (Double.isNaN(logLikelihood)) {
				System.out.println("after topic " + topic + " " + tokensPerTopic[ topic ]);
				System.exit(1);
			}

		}
	
		logLikelihood += 
			(Dirichlet.logGamma(beta * numTopics)) -
			(Dirichlet.logGamma(beta) * nonZeroTypeTopics);

		if (Double.isNaN(logLikelihood)) {
			System.out.println("at the end");
			System.exit(1);
		}


		return logLikelihood;
	}
	public String topWords (int numWords)
	{

		StringBuilder output = new StringBuilder();

		IDSorter[] sortedWords = new IDSorter[numTypes];

		for (int topic = 0; topic < numTopics; topic++) {
			if (tokensPerTopic[topic] == 0) { continue; }
			
			for (int type = 0; type < numTypes; type++) {
				sortedWords[type] = new IDSorter(type, typeTopicCounts[type][topic]);
			}

			Arrays.sort(sortedWords);
			
			output.append(topic + "\t" + labelAlphabet.lookupObject(topic) + "\t" + tokensPerTopic[topic] + "\t");
			for (int i=0; i < numWords; i++) {
				if (sortedWords[i].getWeight() == 0) { break; }
				output.append(alphabet.lookupObject(sortedWords[i].getID()) + " ");
			}
			output.append("\n");
		}

		return output.toString();
	}


		
	// Serialization
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	private static final int NULL_INTEGER = -1;
	
	public void write (File f)
	{
		try {
			ObjectOutputStream oos = new ObjectOutputStream (new FileOutputStream(f));
			oos.writeObject(this);
			oos.close();
		}
		catch (IOException e) {
			System.err.println("Exception writing file " + f + ": " + e);
		}
	}
	public static LabeledLDA read (File f) throws Exception
	{
		LabeledLDA topicModel = null;

		ObjectInputStream ois = new ObjectInputStream (new FileInputStream(f));
		topicModel = (LabeledLDA) ois.readObject();
		ois.close();

		return topicModel;
	}
	private void writeObject (ObjectOutputStream out) throws IOException
	{
		out.writeInt (CURRENT_SERIAL_VERSION);

		// Instance lists
		out.writeObject (data);
		out.writeObject (alphabet);
		out.writeObject (topicAlphabet);

		out.writeInt (numTopics);
		out.writeObject (alpha);
		out.writeDouble (beta);
		out.writeDouble (betaSum);

		out.writeInt(showTopicsInterval);
		out.writeInt(wordsPerTopic);

		out.writeObject(random);
		out.writeBoolean(printLogLikelihood);

		out.writeObject (typeTopicCounts);

		for (int ti = 0; ti < numTopics; ti++) {
			out.writeInt (tokensPerTopic[ti]);
		}
	}
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException
	{
		int featuresLength;
		int version = in.readInt ();

		data = (ArrayList<TopicAssignment>) in.readObject ();
		alphabet = (Alphabet) in.readObject();
		topicAlphabet = (LabelAlphabet) in.readObject();

		numTopics = in.readInt();
		alpha = (Double)in.readObject();
		beta = in.readDouble();
		betaSum = in.readDouble();

		showTopicsInterval = in.readInt();
		wordsPerTopic = in.readInt();

		random = (Randoms) in.readObject();
		printLogLikelihood = in.readBoolean();
		
		int numDocs = data.size();
		this.numTypes = alphabet.size();

		typeTopicCounts = (int[][]) in.readObject();
		tokensPerTopic = new int[numTopics];
		for (int ti = 0; ti < numTopics; ti++) {
			tokensPerTopic[ti] = in.readInt();
		}
	}
}
