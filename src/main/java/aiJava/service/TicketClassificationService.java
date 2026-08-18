package aiJava.service;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.CSVLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;

public class TicketClassificationService {

	private static final String SOURCE_PATH = "data/tickets.csv";
	private static final int NUM_FOLDS = 5;
	private static final int TOP_N_FEATURES = 50;

	private static Instances fullData; 
	private static Instances trainSet; 
	private static StringToWordVector tfidfFilter;
	private static AttributeSelection attrSelectFilter;

	public static void main(String[] args) {
		try {
			Instances rawData = loadData(SOURCE_PATH);
			System.out.println("Loaded " + rawData.numInstances() + " tickets");

			Instances vectorized = vectorizeText(rawData);

			// ---------- Task 3b: Attribute selection (reduce overfitting) ----------
			Instances selected = selectTopFeatures(vectorized, TOP_N_FEATURES);
			fullData = selected;

			// ---------- Task 4: Train + evaluate classifiers via cross-validation
			// ----------
			runModelPipeline(new NaiveBayes(), "Naive Bayes");
			runModelPipeline(new Logistic(), "Logistic Regression");
			runModelPipeline(new SMO(), "SVM (SMO)");

			// ---------- Task 5: Predict new tickets ----------
			// Train final model on ALL available data (not just a split) for real
			// predictions
			trainSet = fullData;
			Classifier finalModel = new SMO();
			finalModel.buildClassifier(trainSet);

			String[] newTickets = { "Cannot login", "Upload failed", "Network timeout", "Permission denied" };//Based on the data we are suggesting the issue
			for (String ticket : newTickets) {
				String prediction = predictNewText(ticket, finalModel);
				System.out.println("\"" + ticket + "\" -> " + prediction);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

		private static Instances loadData(String path) throws Exception {
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(path));
		loader.setStringAttributes("1");
		Instances data = loader.getDataSet();
		data.setClassIndex(data.attribute("category").index());
		return data;
	}

		private static Instances vectorizeText(Instances data) throws Exception {
		tfidfFilter = new StringToWordVector();
		tfidfFilter.setAttributeIndices("first");
		tfidfFilter.setIDFTransform(true);
		tfidfFilter.setTFTransform(true);
		tfidfFilter.setLowerCaseTokens(true);
		tfidfFilter.setOutputWordCounts(false);

		// Remove noise words ("to", "my", "with", Fetc.)
		tfidfFilter.setStopwordsHandler(new Rainbow());

		// Drop words that appear fewer than 2 times across the corpus
		tfidfFilter.setMinTermFreq(2);

		// Normalize so longer tickets don't dominate the vector space
		tfidfFilter.setNormalizeDocLength(
				new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL, StringToWordVector.TAGS_FILTER));

		// Capture short phrases like "cannot login" as well as single words
		NGramTokenizer tokenizer = new NGramTokenizer();
		tokenizer.setNGramMinSize(1);
		tokenizer.setNGramMaxSize(2);
		tfidfFilter.setTokenizer(tokenizer);

		// Collapse word variants: login/logging/logged -> log
		tfidfFilter.setStemmer(new SnowballStemmer());

		// Cap vocabulary to avoid an explosion of sparse attributes
		tfidfFilter.setWordsToKeep(1000);

		tfidfFilter.setInputFormat(data);
		Instances vectorized = Filter.useFilter(data, tfidfFilter);
		vectorized.setClassIndex(vectorized.attribute("category").index());

		System.out.println("Vocabulary size (attributes): " + (vectorized.numAttributes() - 1));
		return vectorized;
	}

	// =========================================================
	// Task 3b - Attribute Selection (keep only the most informative terms)
	// =========================================================
	private static Instances selectTopFeatures(Instances data, int topN) throws Exception {
		attrSelectFilter = new AttributeSelection();
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(Math.min(topN, data.numAttributes() - 1));

		attrSelectFilter.setEvaluator(eval);
		attrSelectFilter.setSearch(ranker);
		attrSelectFilter.setInputFormat(data);

		Instances reduced = Filter.useFilter(data, attrSelectFilter);
		reduced.setClassIndex(reduced.attribute("category").index());

		System.out.println("Reduced to " + (reduced.numAttributes() - 1) + " informative attributes");
		return reduced;
	}

	// =========================================================
	// Task 4 - Train + Evaluate via k-fold Cross-Validation
	// =========================================================
	public static void runModelPipeline(Classifier model, String modelName) throws Exception {
		System.out.println("\n================================");
		System.out.println(modelName + " (" + NUM_FOLDS + "-fold CV)");
		System.out.println("================================");

		Evaluation evaluation = new Evaluation(fullData);
		evaluation.crossValidateModel(model, fullData, NUM_FOLDS, new Random(42));

		System.out.printf("Accuracy : %.2f%%\n", evaluation.pctCorrect());
		System.out.printf("Precision: %.4f\n", evaluation.weightedPrecision());
		System.out.printf("Recall   : %.4f\n", evaluation.weightedRecall());
		System.out.printf("F1       : %.4f\n", evaluation.weightedFMeasure());
	}

	// =========================================================
	// Task 5 - Predict new/unseen text
	// =========================================================
	private static String predictNewText(String text, Classifier model) throws Exception {
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
		Instances selectedNew = Filter.useFilter(vectorizedNew, attrSelectFilter);
		selectedNew.setClassIndex(selectedNew.attribute("category").index());

		double result = model.classifyInstance(selectedNew.instance(0));
		return selectedNew.classAttribute().value((int) result);
	}
}