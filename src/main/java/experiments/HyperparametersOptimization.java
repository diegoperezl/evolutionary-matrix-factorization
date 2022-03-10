package experiments;


import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MSE;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BNMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BiasedMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.NMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;
import es.upm.etsisi.cf4j.util.Range;
import es.upm.etsisi.cf4j.util.optimization.GridSearchCV;
import es.upm.etsisi.cf4j.util.optimization.ParamsGrid;

import java.io.IOException;

public class HyperparametersOptimization {


    private static int NUM_ITERS = 100;
    private static Long seed = 42L;

    public static void main(String[] args) throws IOException {

        DataModel datamodel = BenchmarkDataModels.MovieLens100K();
        //DataModel datamodel = BenchmarkDataModels.FilmTrust();

        // Test PMF model

        ParamsGrid paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{NUM_ITERS});
        paramsGrid.addParam("numFactors", Range.ofIntegers(4, 2, 5));
        paramsGrid.addParam("lambda", Range.ofDoubles(0.005, 0.005, 20));
        paramsGrid.addParam("gamma", Range.ofDoubles(0.005, 0.005, 18));

        paramsGrid.addFixedParam("seed", seed);

        GridSearchCV gridSearchCVPMF = new GridSearchCV(datamodel, paramsGrid, PMF.class, MSE.class, 5, seed);
        gridSearchCVPMF.fit();
        gridSearchCVPMF.exportResults("gridSearchCVPMF", ";", true);

        // Test BiasedMF model

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{NUM_ITERS});
        paramsGrid.addParam("numFactors", Range.ofIntegers(4, 2, 5));
        paramsGrid.addParam("lambda", Range.ofDoubles(0.005, 0.005, 20));
        paramsGrid.addParam("gamma", Range.ofDoubles(0.005, 0.005, 20));

        paramsGrid.addFixedParam("seed", seed);

        GridSearchCV gridSearchCVBiasedMF = new GridSearchCV(datamodel, paramsGrid, BiasedMF.class, MSE.class, 5, seed);
        gridSearchCVBiasedMF.fit();
        gridSearchCVBiasedMF.exportResults("gridSearchCVBiasedMF", ";", true);

        // Test NMF model

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{NUM_ITERS});
        paramsGrid.addParam("numFactors", Range.ofIntegers(4, 2, 5));

        paramsGrid.addFixedParam("seed", seed);

        GridSearchCV gridSearchCVNMF = new GridSearchCV(datamodel, paramsGrid, NMF.class, MSE.class, 5, seed);
        gridSearchCVNMF.fit();
        gridSearchCVNMF.exportResults("gridSearchCVNMF", ";", true);

        // Test BNMF model

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{NUM_ITERS});
        paramsGrid.addParam("numFactors", Range.ofIntegers(4, 2, 5));
        paramsGrid.addParam("alpha", Range.ofDoubles(0.1, 0.1, 9));
        paramsGrid.addParam("beta", Range.ofDoubles(5, 5, 5));

        paramsGrid.addFixedParam("seed", seed);

        GridSearchCV gridSearchCVBNMF = new GridSearchCV(datamodel, paramsGrid, BNMF.class, MSE.class, 5, seed);
        gridSearchCVBNMF.fit();
        gridSearchCVBNMF.exportResults("gridSearchCVBNMF", ";", true);

        // Print results
        gridSearchCVPMF.printResults(5);
        gridSearchCVBiasedMF.printResults(5);
        gridSearchCVNMF.printResults(5);
        gridSearchCVBNMF.printResults(5);
    }
}