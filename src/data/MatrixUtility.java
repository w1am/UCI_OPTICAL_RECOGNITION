package data;

import java.util.Arrays;

public class MatrixUtility {

    public static double[][] add(double[][] matrixOne, double[][] matrixTwo){
        double[][] out = new double[matrixOne.length][matrixOne[0].length];

        for(int i = 0; i < matrixOne.length; i++){
            for(int j = 0; j < matrixOne[0].length; j++){
                out[i][j] = matrixOne[i][j] + matrixTwo[i][j];
            }
        }

        return out;
    }

    public static double[] add(double[] matrixOne, double[] matrixTwo){
        double[] out = new double[matrixOne.length];

        for(int i = 0; i < matrixOne.length; i++){
            out[i] = matrixOne[i] + matrixTwo[i];
        }

        return out;
    }

    public static double[][] multiply(double[][] a, double scalar){
        double[][] out = new double[a.length][a[0].length];
        Arrays.setAll(out, i -> Arrays.stream(a[i]).map(x -> x * scalar).toArray());
        return out;
    }

    public static double[] multiply(double[] a, double scalar){

        double[] out = new double[a.length];

        for(int i = 0; i < a.length; i++){
            out[i] = a[i] * scalar;
        }

        return out;

    }

}
