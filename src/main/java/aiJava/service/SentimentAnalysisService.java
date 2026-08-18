package aiJava.service;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class SentimentAnalysisService {

    private static final String SOURCE_PATH = "data/reviews.csv";
    private static Instances trainSet;
    private static Instances testSet;
    private static StringToWordVector tfidfFilter;

    public static void main(String[] args) {
        try {
            // ---------- Task 1: Load ----------
            Instances rawData = loadData(SOURCE_PATH);
            System.out.println("Loaded " + rawData.numInstances() + " reviews");

            // ---------- Task 2: Check class balance ----------
            checkClassBalance(rawData);// checking whether the dataset is balanced

            // ---------- Task 3: Preprocess + TF-IDF ----------
            Instances vectorized = vectorizeText(rawData);

            // ---------- Task 4: Stratified Split + Train ----------
            stratifiedSplit(vectorized, 5); // 5-fold = 80/20 split so here we are splitting the dataset 
            Classifier model = new NaiveBayesMultinomial();
            runModelPipeline(model, "Naive Bayes (Multinomial)");

            // ---------- Task 5: Cross-Validation (more reliable) ----------
            System.out.println("\n=== 10-Fold Cross Validation ===");
            crossValidate(vectorized, new NaiveBayesMultinomial(), 10);

            // ---------- Task 6: Predict new reviews ----------
            String[] newReviews = {
                "Product is amazing",
                "Worst experience",
                "the product is very good",
                "the product is worst",
                "the product is bad",
                "bad experience",
                "very clean",
                "not upto the mark user experience"
            };

            for (String review : newReviews) {
                String prediction = predictNewText(review, model);
                System.out.println("\"" + review + "\" -> " + prediction);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Task 1 - Load
    private static Instances loadData(String path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        loader.setStringAttributes("1");
        Instances data = loader.getDataSet();
        data.setClassIndex(data.attribute("sentiment").index());
        return data;
    }

       private static void checkClassBalance(Instances data) {
        int[] classCounts = new int[data.numClasses()];
        for (int i = 0; i < data.numInstances(); i++) {
            classCounts[(int) data.instance(i).classValue()]++;
        }
        System.out.println("\n=== Class Distribution ===");
        for (int i = 0; i < data.numClasses(); i++) {
            System.out.println(data.classAttribute().value(i) + ": " + classCounts[i]);
        }
        System.out.println();
    }

    private static Instances vectorizeText(Instances data) throws Exception {
        tfidfFilter = new StringToWordVector();
        tfidfFilter.setAttributeIndices("first");
        tfidfFilter.setIDFTransform(true);
        tfidfFilter.setTFTransform(true);
        tfidfFilter.setLowerCaseTokens(true);
        tfidfFilter.setWordsToKeep(500); //  Limit vocabulary size
        tfidfFilter.setMinTermFreq(2);   //  Ignore rare words

        //  Add N-gram tokenizer (captures "not good", "very bad")
        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(1); // Each word will be divide into separte words
        tokenizer.setNGramMaxSize(2); // Unigrams + Bigrams
        tfidfFilter.setTokenizer(tokenizer);

        tfidfFilter.setInputFormat(data);
        Instances vectorized = Filter.useFilter(data, tfidfFilter);
        vectorized.setClassIndex(vectorized.attribute("sentiment").index());

        System.out.println("Vocabulary size (attributes): " + (vectorized.numAttributes() - 1));
        return vectorized;
    }

    // Task 4 - Stratified Split (Balanced Train/Test)
    private static void stratifiedSplit(Instances data, int folds) throws Exception {
        data.randomize(new Random(42));
        data.stratify(folds);

        trainSet = data.trainCV(folds, 0);
        testSet = data.testCV(folds, 0);

        System.out.println("Train instances: " + trainSet.numInstances() +
                "  Test instances: " + testSet.numInstances());
    }

    // Train & Evaluate Model
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
        System.out.printf("F1-Score : %.4f\n", evaluation.weightedFMeasure());
        System.out.println("\n=== Confusion Matrix ===");
        System.out.println(evaluation.toMatrixString());
    }

    // Cross Validation (More Reliable than Single Split)
    private static void crossValidate(Instances data, Classifier model, int folds) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, folds, new Random(42));

        System.out.printf("CV Accuracy : %.2f%%\n", eval.pctCorrect());
        System.out.printf("CV Precision: %.4f\n", eval.weightedPrecision());
        System.out.printf("CV Recall   : %.4f\n", eval.weightedRecall());
        System.out.printf("CV F1-Score : %.4f\n", eval.weightedFMeasure());
        System.out.println("\n=== CV Confusion Matrix ===");
        System.out.println(eval.toMatrixString());
    }

    // Task 5 - Predict Sentiment for New Reviews
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
        inst.setValue(attrs.get(0), text.toLowerCase()); // Lowercase to match training
        newData.add(inst);

        Instances vectorizedNew = Filter.useFilter(newData, tfidfFilter);
        vectorizedNew.setClassIndex(vectorizedNew.attribute("sentiment").index());

        double result = model.classifyInstance(vectorizedNew.instance(0));
        return vectorizedNew.classAttribute().value((int) result);
    }
}