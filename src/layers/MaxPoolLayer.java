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

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for (int row = 0, i = 0; row < getOutputRows(); row += _stepSize, i++) {
            for (int col = 0, j = 0; col < getOutputCols(); col += _stepSize, j++) {
                double max = input[row][col];

                // Perform the pooling operation
                maxRows[i][j] = row;
                maxCols[i][j] = col;

                // Find the maximum value in the window
                if (input[row][col + 1] > max) {
                    max = input[row][col + 1]; maxRows[i][j] = row; maxCols[i][j] = col + 1;
                }

                if (input[row + 1][col] > max) {
                    max = input[row + 1][col]; maxRows[i][j] = row + 1; maxCols[i][j] = col;
                }

                if (input[row + 1][col + 1] > max) {
                    max = input[row + 1][col + 1]; maxRows[i][j] = row + 1; maxCols[i][j] = col + 1;
                }

                // store the location of the max value
                output[i][j] = max;
            }
        }

        // Remember the last position of the max value
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

        List<double[][]> dXdL = new ArrayList<>(dLdO.size());

        final int outputRows = getOutputRows();
        final int outputCols = getOutputCols();

        int l = 0;
        for (double[][] array : dLdO) {
            double[][] error = new double[_inRows][_inCols];

            // Find where the error came from and add it to the error matrix
            final int outputSize = outputRows * outputCols;
            for (int i = 0; i < outputSize; i++) {
                final int row = i / outputCols;
                final int col = i % outputCols;

                // Get previous x and y position of the max value
                final int max_i = _lastMaxRow.get(l)[row][col];
                final int max_j = _lastMaxCol.get(l)[row][col];

                // If there was a max value, add the corresponding coordinate to where our max value came from
                // Then add the error to the error matrix
                if (max_i != -1) {
                    error[max_i][max_j] += array[row][col];
                }
            }

            dXdL.add(error);
            l++;
        }

        if (_previousLayer != null) {
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
