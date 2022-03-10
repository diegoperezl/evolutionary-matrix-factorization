package mf;


import es.upm.etsisi.cf4j.data.Item;
import es.upm.etsisi.cf4j.data.User;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.data.DataModel;
import sym_derivation.symderivation.SymFunction;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class EMF extends Recommender {

    private double learningRate;
    private double regularization;
    private int numFactors;
    private int numIters;

    private SymFunction sf;

    private double[][] p;
    private double[][] q;

    private Random rand;

    /**
     * Model constructor from a Map containing the model's hyper-parameters values. Map object must
     * contains the following keys:
     *
     * <ul>
     *   <li><b>func</b>: String value with the function to evaluate.
     *   <li><b>numFactors:</b>: int value with the number of factors.
     *   <li><b>numIters:</b>: int value with the number of iterations.
     *   <li><b>regularization:</b>: double value with the regularization.
     *   <li><b>learningRate:</b>: double value with the learning rate.
     *   <li><b><em>seed</em></b> (optional): random seed for random numbers generation. If missing,
     *       random value is used.
     * </ul>
     *
     * @param datamodel DataModel instance
     * @param params    Model's hyper-parameters values
     */
    public EMF(DataModel datamodel, Map<String, Object> params) {
        this(
                datamodel,
                (String) params.get("func"),
                (int) params.get("numFactors"),
                (int) params.get("numIters"),
                (double) params.get("regularization"),
                (double) params.get("learningRate"),
                params.containsKey("seed") ? (long) params.get("seed") : System.currentTimeMillis());
    }


    /**
     * Model constructor
     *
     * @param datamodel      DataModel instance
     * @param func           Function to evaluate
     * @param numFactors     Number of factors
     * @param numIters       Number of iterations
     * @param regularization Regularization
     * @param learningRate   Learning rate
     */
    public EMF(DataModel datamodel, String func, int numFactors, int numIters, double regularization, double learningRate, long seed) {
        super(datamodel);

        // create model function
        this.sf = SymFunction.parse(func);

        // model hyper-parameters
        this.numFactors = numFactors;
        this.numIters = numIters;
        this.regularization = regularization;
        this.learningRate = learningRate;

        this.rand = new Random(seed);

        // users factors initialization
        this.p = new double[datamodel.getNumberOfUsers()][numFactors];
        for (User user : super.getDataModel().getUsers()) {
            p[user.getUserIndex()] = this.random(this.numFactors, 0, 1);
        }

        // items factors initialization
        this.q = new double[datamodel.getNumberOfItems()][numFactors];
        for (Item item : super.getDataModel().getItems()) {
            q[item.getItemIndex()] = this.random(this.numFactors, 0, 1);
        }
    }

    public void fit() {

        System.out.println("\nProcessing EMF...");

        // partial derivatives of the model function

        SymFunction[] puSfDiff = new SymFunction[this.numFactors];
        SymFunction[] qiSfDiff = new SymFunction[this.numFactors];

        for (int k = 0; k < this.numFactors; k++) {
            puSfDiff[k] = sf.diff("pu" + k);
            qiSfDiff[k] = sf.diff("qi" + k);
        }

        // repeat numIters times
        for (int iter = 1; iter <= this.numIters; iter++) {

            // compute gradient
            double[][] dp = new double[super.getDataModel().getNumberOfUsers()][this.numFactors];
            double[][] dq = new double[super.getDataModel().getNumberOfItems()][this.numFactors];

            for (User user : super.getDataModel().getUsers()) {
                for (int i = 0; i < user.getNumberOfRatings(); i++) {
                    int itemIndex = user.getItemAt(i);


                    HashMap<String, Double> params = getParams(p[user.getUserIndex()], q[itemIndex]);

                    double prediction = sf.eval(params);
                    double error = user.getRatingAt(i) - prediction;

                    for (int k = 0; k < this.numFactors; k++) {
                        dp[user.getUserIndex()][k] += this.learningRate * (error * puSfDiff[k].eval(params) - this.regularization * p[user.getUserIndex()][k]);
                        dq[itemIndex][k] += this.learningRate * (error * qiSfDiff[k].eval(params) - this.regularization * q[itemIndex][k]);
                    }
                }
            }

            // update users factors
            for (User user : super.getDataModel().getUsers()) {
                for (int k = 0; k < this.numFactors; k++) {
                    p[user.getUserIndex()][k] += dp[user.getUserIndex()][k];
                }
            }

            // update items factors
            for (Item item : super.getDataModel().getItems()) {
                for (int k = 0; k < this.numFactors; k++) {
                    q[item.getItemIndex()][k] += dq[item.getItemIndex()][k];
                }
            }

            if ((iter % 10) == 0) System.out.print(".");
            if ((iter % 100) == 0) System.out.println(iter + " iterations");
        }
    }

    public double predict(int userIndex, int itemIndex) {
        HashMap<String, Double> params = getParams(this.p[userIndex], this.q[itemIndex]);
        return sf.eval(params);
    }

    private double random(double min, double max) {
        return Math.random() * (max - min) + min;
    }

    private double[] random(int size, double min, double max) {
        double[] d = new double[size];
        for (int i = 0; i < size; i++) d[i] = this.random(min, max);
        return d;
    }

    private HashMap<String, Double> getParams(double[] pu, double[] qi) {
        HashMap<String, Double> map = new HashMap<>();
        for (int k = 0; k < this.numFactors; k++) {
            map.put("pu" + k, pu[k]);
            map.put("qi" + k, qi[k]);
        }
        return map;
    }
}
