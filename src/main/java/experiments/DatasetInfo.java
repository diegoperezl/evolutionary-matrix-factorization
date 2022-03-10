package experiments;


import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;

import java.io.IOException;

public class DatasetInfo {

    public static void main(String[] args) throws IOException {
        DataModel datamodel = BenchmarkDataModels.MovieLens100K();
        //DataModel datamodel = BenchmarkDataModels.FilmTrust();

        System.out.println(datamodel.toString());
    }
}
