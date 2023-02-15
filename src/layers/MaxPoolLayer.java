package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{

    private final int _stepSize;
    private final int _windowSize;

    private final int _inLength;
    private final int _inRows;
    private final int _inCols;

    List<int[][]> _lastMaxRow;
    List<int[][]> _lastMaxCol;


    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int _inRows, int _inCols) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
    }

    /**
     * Returns the number of rows in the output of the max pool layer
     * @param input the input to the max pool layer
     * @return the number of rows in the output of the max pool layer
     */
    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {

        List<double[][]> output = new ArrayList<>();

        // Two new ArrayLists are created to store the location of each max value in the input arrays
        _lastMaxRow = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();

        for (double[][] doubles : input) {
            output.add(pool(doubles)); // The output of the pooling operation is added to the output List
        }

        return output;

    }

    /**
     * Returns the number of rows in the output of the max pool layer
     * @param input the input to the max pool layer
     * @return the number of rows in the output of the max pool layer
     */
    public double[][] pool(double[][] input){

        double[][] output = new double[getOutputRows()][getOutputCols()];

        // Create two 2D arrays to keep track of the row and column indices of the maximum values in each window.
        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for(int r = 0; r < getOutputRows(); r+= _stepSize){
            for(int c = 0; c < getOutputCols(); c+= _stepSize){

                double max = 0.0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                // Loop through each element in the current window.
                for(int x = 0; x < _windowSize; x++){
                    for(int y = 0; y < _windowSize; y++) {

                        // If the current element is greater than the current maximum, update the maximum and index variables.
                        if(max < input[r+x][c+y]){
                            max = input[r+x][c+y];
                            maxRows[r][c] = r+x;
                            maxCols[r][c] = c+y;
                        }

                    }
                }

                output[r][c] = max;

            }
        }

        // Add the row and column index arrays to the lists for later use.
        _lastMaxRow.add(maxRows);
        _lastMaxCol.add(maxCols);

        return output;

    }


    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrixList);
    }

    @Override
    public void backPropagation(double[] dLdO, int iteration) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrixList, iteration);
    }

    /**
     * Backpropagation for the max pooling layer
     * @param dLdO List of dL/dO matrices
     * @param iteration Current iteration
     */
    @Override
    public void backPropagation(List<double[][]> dLdO, int iteration) {

        List<double[][]> dXdL = new ArrayList<>();

        // Initialize a counter for the index of the dL/dO matrix being processed
        int l = 0;

        for(double[][] array: dLdO){
            double[][] error = new double[_inRows][_inCols];

            // Loop through each element in the output matrix
            for(int r = 0; r < getOutputRows(); r++){
                for(int c = 0; c < getOutputCols(); c++){
                    // Find the index of the max element in the corresponding input submatrix
                    int max_i = _lastMaxRow.get(l)[r][c];
                    int max_j = _lastMaxCol.get(l)[r][c];

                    if (max_i != -1) {
                        // Add the corresponding element from the dL/dO matrix to the error matrix
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }

            // Add the error matrix to the list of dX/dL matrices
            dXdL.add(error);
            l++;
        }

        if(_previousLayer!= null){
            _previousLayer.backPropagation(dXdL, iteration);
        }

    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inLength*getOutputCols()*getOutputRows();
    }
}
