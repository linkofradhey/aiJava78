package aiJava.service;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;
public class TicketClassificationService {

	private static final String SOURCE_PATH = "data/tickets.csv"; // columns: text, category (text must be String type)
	private static Instances trainSet;
	private static Instances testSet;
	private static StringToWordVector tfidfFilter;

	public static void main(String[] args) {
		try {
			// ---------- Task 1: Load ----------
			Instances rawData = loadData(SOURCE_PATH);
			System.out.println("Loaded " + rawData.numInstances() + " tickets");

			// ---------- Task 2 & 3: Preprocess + TF-IDF vectorization ----------
			// StringToWordVector handles tokenizing, lowercasing, stopword removal,
			// and TF-IDF weighting all in one filter — Weka's equivalent of
			// Python's TfidfVectorizer.
			Instances vectorized = vectorizeText(rawData);

			// ---------- Task 4: Train classifier ----------
			prepareData(vectorized, 0.75);
			runModelPipeline(new NaiveBayes(), "Naive Bayes");
			runModelPipeline(new Logistic(), "Logistic Regression");

			// ---------- Task 5: Predict new tickets ----------
			String[] newTickets = { "Cannot login", "Upload failed", "Network timeout", "Permission denied" };
			Classifier finalModel = new NaiveBayes();
			finalModel.buildClassifier(trainSet);
			for (String ticket : newTickets) {
				String prediction = predictNewText(ticket, finalModel);
				System.out.println("\"" + ticket + "\" -> " + prediction);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// =========================================================
	// Task 1 - Load
	// =========================================================
	private static Instances loadData(String path) throws Exception {
		// Using CSVLoader directly (not DataSource) so we can force the "text"
		// column to load as STRING type. Left to auto-detect, Weka's loader
		// treats short repeated-looking text as NOMINAL (categorical) instead
		// of free text, which silently breaks StringToWordVector downstream —
		// it will vectorize almost nothing and every model will score ~0%.
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(path));
		loader.setStringAttributes("1"); // column 1 = "text"
		Instances data = loader.getDataSet();
		data.setClassIndex(data.attribute("category").index());
		return data;
	}

	// =========================================================
	// Task 2 & 3 - Preprocessing + TF-IDF Vectorization
	// =========================================================
	private static Instances vectorizeText(Instances data) throws Exception {
		tfidfFilter = new StringToWordVector();
		tfidfFilter.setAttributeIndices("first"); // "text" column is the string attribute to vectorize
		tfidfFilter.setIDFTransform(true);  // apply IDF weighting
		tfidfFilter.setTFTransform(true);   // apply TF weighting -> together, TF-IDF
		tfidfFilter.setLowerCaseTokens(true);   // lowercase, same as Python's text.lower()
		tfidfFilter.setOutputWordCounts(false); // false = TF-IDF weights, not raw counts

		tfidfFilter.setInputFormat(data);
		Instances vectorized = Filter.useFilter(data, tfidfFilter);

		// class index shifts after filtering — re-point it at "category"
		vectorized.setClassIndex(vectorized.attribute("category").index());

		System.out.println("Vocabulary size (attributes): " + (vectorized.numAttributes() - 1));
		return vectorized;
	}

	// =========================================================
	// Task 4 - Train Classifier
	// =========================================================
	public static void prepareData(Instances data, double trainRatio) {
		data.randomize(new Random(42));
		int trainSize = (int) Math.round(data.numInstances() * trainRatio);
		int testSize = data.numInstances() - trainSize;
		trainSet = new Instances(data, 0, trainSize);
		testSet = new Instances(data, trainSize, testSize);
		System.out.println("Train instances: " + trainSet.numInstances() + "  Test instances: " + testSet.numInstances());
	}

	public static void runModelPipeline(Classifier model, String modelName) throws Exception {
		System.out.println("\n================================");
		System.out.println(modelName);
		System.out.println("================================");

		model.buildClassifier(trainSet);
		Evaluation evaluation = new Evaluation(trainSet);
		evaluation.evaluateModel(model, testSet);

		System.out.printf("Accuracy : %.2f%%\n", evaluation.pctCorrect());
		System.out.printf("Precision: %.4f\n", evaluation.weightedPrecision());
		System.out.printf("Recall   : %.4f\n", evaluation.weightedRecall());
	}

	// =========================================================
	// Task 5 - Predict new/unseen text
	// =========================================================
	private static String predictNewText(String text, Classifier model) throws Exception {
		// Build a single-row Instances with the same string attribute Weka expects,
		// then apply the SAME fitted TF-IDF filter (never refit on new data).
		ArrayList<Attribute> attrs = new ArrayList<>();
		attrs.add(new Attribute("text", (ArrayList<String>) null));
		ArrayList<String> categories = new ArrayList<>();
		for (int i = 0; i < trainSet.classAttribute().numValues(); i++) {
			categories.add(trainSet.classAttribute().value(i));
		}
		attrs.add(new Attribute("category", categories));

		Instances newData = new Instances("newTicket", attrs, 1);
		newData.setClassIndex(1);

		Instance inst = new DenseInstance(2);
		inst.setValue(attrs.get(0), text);
		newData.add(inst);

		Instances vectorizedNew = Filter.useFilter(newData, tfidfFilter);
		vectorizedNew.setClassIndex(vectorizedNew.attribute("category").index());

		double result = model.classifyInstance(vectorizedNew.instance(0));
		return vectorizedNew.classAttribute().value((int) result);
	}
}
