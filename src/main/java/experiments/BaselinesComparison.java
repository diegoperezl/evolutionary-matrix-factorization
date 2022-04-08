package experiments;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MSE;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BNMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BiasedMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.NMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;
import mf.EMF;

import java.io.IOException;

public class BaselinesComparison {

    private static int NUM_ITERS = 100;

    //private static final String BINARY_FILE = "datasets/ml100k.cf4j";

    private static int PMF_NUM_TOPICS = 4;
    private static double PMF_LAMBDA = 0.1;
    private static double PMF_GAMMA = 0.005;

    private static int BIASED_MF_NUM_TOPICS = 4;
    private static double BIASED_MF_LAMBDA = 0.1;
    private static double BIASED_MF_GAMMA = 0.01;

    private static int NMF_NUM_TOPICS = 8;

    private static int BNMF_NUM_TOPICS = 8;
    private static double BNMF_ALPHA = 0.8;
    private static int BNMF_BETA = 10;

    private static int EMF_NUM_TOPICS = 6;
    private static double EMF_LEARNING_RATE = 0.001;
    private static double EMF_REGULARIZARION = 0.095;

    public static long seed = 42L;

    private static String[] EMF_FUNCS = {
            "+(-(pu4,-(*(*(pu5,qi3),qi3),pu1)),qi3)"
    };

//    private static final String BINARY_FILE = "datasets/filmtrust.cf4j";
//
//    private static int PMF_NUM_TOPICS = 6;
//    private static double PMF_LAMBDA = 0.085;
//    private static double PMF_GAMMA = 0.01;
//
//    private static int BIASED_MF_NUM_TOPICS = 12;
//    private static double BIASED_MF_LAMBDA = 0.095;
//    private static double BIASED_MF_GAMMA = 0.03;
//
//    private static int NMF_NUM_TOPICS = 4;
//
//    private static int BNMF_NUM_TOPICS = 4;
//    private static double BNMF_ALPHA = 0.8;
//    private static int BNMF_BETA = 5;
//
//    private static int EMF_NUM_TOPICS = 10;
//    private static double EMF_LEARNING_RATE = 0.0035;
//    private static double EMF_REGULARIZARION = 0.095;
//
//    private static String [] EMF_FUNCS = {
//            "exp atan + cos pu5 exp + + pu3 qi2 pu3",
//            "-- -- + sin -- -- pu0 + One exp sin cos exp + pu6 -- + sin pu9 + sin qi4 exp sin -- + One exp sin cos -- + sin sin pu9 + One exp sin cos exp + pu6 -- + sin qi4 exp sin -- sin pu9",
//            "inv exp atan - pu2 exp exp atan - exp - exp - pu2 exp atan - pu2 pu7 exp - - pu7 qi3 exp atan - pu2 exp - pu2 exp atan - pu2 exp exp atan - exp - pu7 exp - - pu2 exp exp atan - exp - pu2 exp - - pu7 qi3 exp atan - pu2 exp - pu2 exp atan - pu2 exp exp - pu2 exp atan - pu2 - pu2 pu7 - pu7 pu2 pu7 - pu7 pu2 - pu7 pu2",
//            "+ pu4 exp cos qi6",
//            "exp atan - exp exp pu1 -- qi9",
//            "exp atan + qi4 + + pu3 atan atan pu2 inv pu2",
//            "+ atan exp qi2 exp cos pu2",
//            "+ exp pu3 exp cos sin exp -- atan qi0",
//            "exp inv cos cos exp cos exp atan - * One exp * qi6 cos cos exp * pu7 qi2 exp atan - * One exp * qi6 cos pu4 pu0",
//            "exp atan + * atan + + atan + * atan + + qi5 atan + qi5 atan - inv pu8 * qi5 inv inv pu8 inv pu8 + qi5 atan - inv pu8 * qi5 inv inv pu8 inv pu8 atan + qi5 atan - inv pu8 * qi5 inv + qi5 atan qi5 inv pu8 + qi5 atan - inv pu8 * qi5 inv inv pu8 inv pu8"
//    };

    public static void main(String[] args) throws IOException {

        // define series

        String[] series = new String[4 + EMF_FUNCS.length];

        series[0] = "PMF";
        series[1] = "BiasedMF";
        series[2] = "NMF";
        series[3] = "BNMF";

        for (int i = 1; i <= EMF_FUNCS.length; i++) {
            series[3 + i] = "EMF_" + i;
        }

        // define quality measures

        double[] mae = new double[series.length];
        double[] mse = new double[series.length];

        // load dataset

        DataModel datamodel = BenchmarkDataModels.MovieLens100K();
        //DataModel datamodel = BenchmarkDataModels.FilmTrust();


        // test series

        for (int s = 0; s < series.length; s++) {
            String serie = series[s];

            Recommender recommender;

            if (serie.equals("PMF")) {
                recommender = new PMF(datamodel, PMF_NUM_TOPICS, NUM_ITERS, PMF_LAMBDA, PMF_GAMMA, seed);

            } else if (serie.equals("BiasedMF")) {
                recommender = new BiasedMF(datamodel, BIASED_MF_NUM_TOPICS, NUM_ITERS, BIASED_MF_LAMBDA, BIASED_MF_GAMMA, seed);

            } else if (serie.equals("NMF")) {
                recommender = new NMF(datamodel, NMF_NUM_TOPICS, NUM_ITERS, seed);

            } else if (serie.equals("BNMF")) {
                recommender = new BNMF(datamodel, BNMF_NUM_TOPICS, NUM_ITERS, BNMF_ALPHA, BNMF_BETA, seed);

            } else { // serie.equals("EMF_<id>")
                int index = Integer.parseInt(serie.split("_")[1]) - 1;
                String func = EMF_FUNCS[index];

                func = func
                        .replace("(", " ")
                        .replace(")", " ")
                        .replace(",", " ");

                recommender = new EMF(datamodel, func, EMF_NUM_TOPICS, NUM_ITERS, EMF_REGULARIZARION, EMF_LEARNING_RATE, seed);
            }

            recommender.fit();
            //Processor.getInstance().testUsersProcess(new FactorizationPrediction(fm));

            mae[s] = new MAE(recommender).getScore();
            mse[s] = new MSE(recommender).getScore();
        }

            // print results

            System.out.println("\nMethod;MAE;MSE");
            for (int i = 0; i < series.length; i++) {
                System.out.println(series[i] + ";" + mae[i] + ";" + mse[i]);
            }
        }


}
