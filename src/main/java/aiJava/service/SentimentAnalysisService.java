package aiJava.service;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stopwords.Rainbow;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;

public class SentimentAnalysisService {

	private static final String SOURCE_PATH = "data/reviews.csv"; // columns: review, sentiment
	private static Instances trainSet;
	private static Instances testSet;
	private static StringToWordVector tfidfFilter;

	public static void main(String[] args) {
		try {
			// ---------- Task 1: Load ----------
			Instances rawData = loadData(SOURCE_PATH);
			System.out.println("Loaded " + rawData.numInstances() + " reviews");

			// ---------- Task 2 & 3: Preprocess + TF-IDF vectorization ----------
			Instances vectorized = vectorizeText(rawData);

			// ---------- Task 4: Train sentiment model ----------
			prepareData(vectorized, 0.75);
			Classifier model = new NaiveBayesMultinomial();
			runModelPipeline(model, "Naive Bayes (Multinomial)");

			// ---------- Task 5: Predict sentiment for new reviews ----------
			String[] newReviews = { "Product is amazing", "Worst experience" };
			for (String review : newReviews) {
				String prediction = predictNewText(review, model);
				System.out.println("\"" + review + "\" -> " + prediction);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// =========================================================
	// Task 1 - Load
	// =========================================================
	private static Instances loadData(String path) throws Exception {
		// Force the "review" column to load as STRING (see TicketClassificationService
		// for why this matters — Weka's CSV auto-detection otherwise treats it
		// as NOMINAL and silently breaks TF-IDF vectorization).
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(path));
		loader.setStringAttributes("1"); // column 1 = "review"
		Instances data = loader.getDataSet();
		data.setClassIndex(data.attribute("sentiment").index());
		return data;
	}

	// =========================================================
	// Task 2 & 3 - Preprocessing + TF-IDF Vectorization
	// =========================================================
	private static Instances vectorizeText(Instances data) throws Exception {
		tfidfFilter = new StringToWordVector();
		tfidfFilter.setAttributeIndices("first"); // "review" column
		tfidfFilter.setIDFTransform(true);
		tfidfFilter.setTFTransform(true);
		tfidfFilter.setLowerCaseTokens(true);

		// Unlike Week 7, we do NOT enable stopword removal here — "not"/"no"
		// are stopwords in most default lists but flip sentiment meaning,
		// so keeping them is a deliberate choice (mirrors the Python version).
		// If you do want stopword removal, use a custom list that excludes negations:
		// tfidfFilter.setStopwordsHandler(new Rainbow());

		tfidfFilter.setInputFormat(data);
		Instances vectorized = Filter.useFilter(data, tfidfFilter);
		vectorized.setClassIndex(vectorized.attribute("sentiment").index());

		System.out.println("Vocabulary size (attributes): " + (vectorized.numAttributes() - 1));
		return vectorized;
	}

	// =========================================================
	// Task 4 - Train Sentiment Model
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
	// Task 5 - Predict Sentiment for New Reviews
	// =========================================================
	private static String predictNewText(String text, Classifier model) throws Exception {
		ArrayList<Attribute> attrs = new ArrayList<>();
		attrs.add(new Attribute("review", (ArrayList<String>) null));
		ArrayList<String> sentiments = new ArrayList<>();
		for (int i = 0; i < trainSet.classAttribute().numValues(); i++) {
			sentiments.add(trainSet.classAttribute().value(i));
		}
		attrs.add(new Attribute("sentiment", sentiments));

		Instances newData = new Instances("newReview", attrs, 1);
		newData.setClassIndex(1);

		Instance inst = new DenseInstance(2);
		inst.setValue(attrs.get(0), text);
		newData.add(inst);

		Instances vectorizedNew = Filter.useFilter(newData, tfidfFilter);
		vectorizedNew.setClassIndex(vectorizedNew.attribute("sentiment").index());

		double result = model.classifyInstance(vectorizedNew.instance(0));
		return vectorizedNew.classAttribute().value((int) result);
	}
}
