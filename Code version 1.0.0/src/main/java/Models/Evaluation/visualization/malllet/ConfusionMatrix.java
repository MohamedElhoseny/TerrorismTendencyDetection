/* Copyright (C) 2002 Dept. of Computer Science, Univ. of Massachusetts, Amherst

   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet

   This program toolkit free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  For more
   details see the GNU General Public License and the file README-LEGAL.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
   02111-1307, USA. */


/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package Models.Evaluation.visualization.malllet;


import cc.mallet.classify.Classification;
import cc.mallet.classify.Trial;
import cc.mallet.types.*;
import cc.mallet.util.MalletLogger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.logging.Logger;

/**
 * Calculates and prints confusion matrix, accuracy,
 * and precision for a given clasification trial.
 */
public class ConfusionMatrix implements Serializable {

	private static Logger logger = MalletLogger.getLogger(cc.mallet.classify.evaluate.ConfusionMatrix.class.getName());

	int numClasses;
	/**
	 * the list of classifications from the trial
	 */
	ArrayList classifications;
	/**
	 * 2-d confusion matrix, indexed by [actual class][predicted class]
	 */
	int[][] values;
	int[] totals;

	Trial trial;

	/**
	 * Constructs matrix and calculates values
	 *
	 * @param t the trial to build matrix from
	 */
	public ConfusionMatrix(Trial t) {
		this.trial = t;
		this.classifications = t;
		Labeling tempLabeling = ((Classification) classifications.get(0)).getLabeling();
		this.numClasses = tempLabeling.getLabelAlphabet().size();
		values = new int[numClasses][numClasses];
		totals = new int[numClasses];

		for (Object classification : classifications) {
			LabelVector lv = ((Classification) classification).getLabelVector();
			Instance inst = ((Classification) classification).getInstance();
			int bestIndex = lv.getBestIndex();
			int correctIndex = inst.getLabeling().getBestIndex();
			assert (correctIndex != -1);
			//System.out.println("Best index="+bestIndex+". Correct="+correctIndex);

			values[correctIndex][bestIndex]++;
		}
	}

	/**
	 * Return the count at row i (true) , column j (predicted)
	 */
	public double value(int i, int j) {
		assert (i >= 0 && j >= 0 && i < numClasses && j < numClasses);
		return values[i][j];
	}

	public double[][] getmatrix(int dimention)  //2 -> 2*2  3 -> 3*3
	{
		double[][] m = new double[dimention][];
		for (int i = 0; i < dimention; i++) {
			for (int j = 0; j < dimention; j++) {
				m[i][j] = value(i, j);
			}
		}
		return m;
	}

	static private void appendJustifiedInt(StringBuffer sb, int i, boolean zeroDot) {
		if (i < 100) sb.append(' ');
		if (i < 10) sb.append(' ');
		if (i == 0 && zeroDot) {
			sb.append(".");
		} else {
			sb.append("" + i);
		}
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		int maxLabelNameLength = 0;
		LabelAlphabet labelAlphabet = trial.getClassifier().getLabelAlphabet();
		for (int i = 0; i < numClasses; i++) {
			int len = labelAlphabet.lookupLabel(i).toString().length();
			if (maxLabelNameLength < len) {
				maxLabelNameLength = len;
			}
		}

		// These counts will be integers, but we'll keep them as doubles so we can divide later
		double[] correctLabelCounts = new double[values.length];

		for (int i = 0; i < correctLabelCounts.length; i++) {
			// This sum is the number of instances whose correct class is i
			correctLabelCounts[i] = MatrixOps.sum(values[i]);
		}
		// Find the count of the most frequent class and divide that by 
		//  the total number of instances.
		double baselineAccuracy = MatrixOps.max(correctLabelCounts) / MatrixOps.sum(correctLabelCounts);

		sb.append("Confusion Matrix, row=true, column=predicted  accuracy=" + trial.getAccuracy() + "" + " most-frequent-tag baseline=" + baselineAccuracy + "\n");

		for (int i = 0; i < maxLabelNameLength - 5 + 4; i++) {
			sb.append(' ');
		}
		sb.append("label");
		for (int c2 = 0; c2 < Math.min(10, numClasses); c2++) {
			sb.append("   " + c2);
		}
		for (int c2 = 10; c2 < numClasses; c2++) {
			sb.append("  " + c2);
		}
		sb.append("  |total\n");

		for (int c = 0; c < numClasses; c++) {
			appendJustifiedInt(sb, c, false);

			String labelName = labelAlphabet.lookupLabel(c).toString();
			for (int i = 0; i < maxLabelNameLength - labelName.length(); i++) {
				sb.append(' ');
			}
			sb.append(" " + labelName + " ");
			for (int c2 = 0; c2 < numClasses; c2++) {
				appendJustifiedInt(sb, values[c][c2], true);
				sb.append(' ');
			}
			totals[c] = MatrixOps.sum(values[c]);
			sb.append(" |" + totals[c]);
			sb.append('\n');
		}
		return sb.toString();
	}

	/**
	 * Returns the precision of this predicted class
	 */
	public double getPrecision(int predictedClassIndex) {
		int total = 0;
		for (int trueClassIndex = 0; trueClassIndex < this.numClasses; trueClassIndex++) {
			total += values[trueClassIndex][predictedClassIndex];
		}

		if (total == 0) {
			return 0.0;
		} else {
			return (double) (values[predictedClassIndex][predictedClassIndex]) / total;
		}
	}

	/**
	 * Returns percent of time that class2 is true class when
	 * class1 is predicted class
	 */
	public double getConfusionBetween(int class1, int class2) {
		int total = 0;
		for (int trueClassIndex = 0; trueClassIndex < this.numClasses; trueClassIndex++) {
			total += values[trueClassIndex][class1];
		}
		if (total == 0) {
			return 0.0;
		} else {
			return (double) (values[class2][class1]) / total;
		}
	}

	/**
	 * Returns the percentage of instances with
	 * true label = classIndex
	 */
	public double getClassPrior(int classIndex) {
		double sum = 0;
		for (int i = 0; i < numClasses; i++) {
			sum += values[classIndex][i];
		}
		return sum / classifications.size();
	}

    public int[] getTotals(){return  this.totals;}
}











