package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class ConvolutionLayer extends Layer{

    private final long SEED;

    private List<double[][]> _filters;
    private final int _filterSize;
    private final int _stepsize;

    private final int _inLength;
    private final int _inRows;
    private final int _inCols;
    private final double _learningRate;

    private List<double[][]> _lastInput;

    public ConvolutionLayer(int _filterSize, int _stepsize, int _inLength, int _inRows, int _inCols, long SEED, int numFilters, double learningRate) {
        this._filterSize = _filterSize;
        this._stepsize = _stepsize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
        this.SEED = SEED;
        _learningRate = learningRate;

        generateRandomFilters(numFilters);

    }

    private void generateRandomFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(SEED);

        for(int n = 0; n < numFilters; n++) {
            double[][] newFilter = new double[_filterSize][_filterSize];

            for(int i = 0; i < _filterSize; i++){
                for(int j = 0; j < _filterSize; j++){

                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }

            filters.add(newFilter);

        }

        _filters = filters;

    }

    /**
     * This method performs a convolution operation between a 2D input matrix and a 2D filter.
     * @param list The 2D input matrix
     * @return The output matrix obtained by convolving the input matrix with the filter
     */
    public List<double[][]> convolutionForwardPass(List<double[][]> list){

        _lastInput = list;

        List<double[][]> output = new ArrayList<>();

        for (double[][] doubles : list) {
            for (double[][] filter : _filters) {
                output.add(convolve(doubles, filter, _stepsize));
            }

        }

        return output;

    }

    /**
     * This method performs a convolution operation between a 2D input matrix and a 2D filter.
     * The output matrix is calculated by sliding the filter over the input matrix with a specified step size,
     * and computing the dot product between the filter and the corresponding sub-matrix of the input.
     * @param input The 2D input matrix
     * @param filter The 2D filter matrix
     * @param stepSize The step size for the convolution operation
     * @return The output matrix obtained by convolving the input matrix with the filter
     **/
    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        // Calculate output matrix dimensions
        int outRows = (input.length - filter.length) / stepSize + 1;
        int outCols = (input[0].length - filter[0].length) / stepSize + 1;

        // Store the filter matrix
        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                double sum = 0.0;

                int rowOffset = i * stepSize;
                int colOffset = j * stepSize;

                // Apply filter to input matrix at current position
                for (int x = 0; x < fRows; x++) {
                    int inputRowIndex = rowOffset + x;

                    double[] inputRow = input[inputRowIndex];
                    double[] filterRow = filter[x];

                    for (int y = 0; y < fCols; y++) {
                        int inputColIndex = colOffset + y;

                        // Multiply filter value with corresponding input value and add to sum
                        double value = filterRow[y] * inputRow[inputColIndex];
                        sum += value;
                    }
                }

                output[i][j] = sum;
            }
        }

        return output;
    }

    /**
     * Takes in a 2D array and returns a new 2D array that has been expanded by a step size.
     * If the step size is 1, returns the input array. Otherwise, expands the input array
     * by filling in the gaps with zeros.
     * @param input the input 2D array to be expanded
     * @return the expanded 2D array
     **/
    public double[][] spaceArray(double[][] input){

        if(_stepsize == 1){
            return input;
        }

        int outRows = (input.length - 1)*_stepsize + 1;
        int outCols = (input[0].length -1)*_stepsize+1;

        double[][] output = new double[outRows][outCols];

        for(int i = 0; i < input.length; i++){
            for(int j = 0; j < input[0].length; j++){
                output[i*_stepsize][j*_stepsize] = input[i][j];
            }
        }

        return output;
    }


    @Override
    public double[] getOutput(List<double[][]> input) {

        List<double[][]> output = convolutionForwardPass(input);

        return _nextLayer.getOutput(output);

    }

    @Override
    public double[] getOutput(double[] input) {

        List<double[][]> matrixInput = vectorToMatrix(input, _inLength, _inRows, _inCols);

        return getOutput(matrixInput);
    }

    @Override
    public void backPropagation(double[] dLdO, int iteration) {
        List<double[][]> matrixInput = vectorToMatrix(dLdO, _inLength, _inRows, _inCols);
        backPropagation(matrixInput, iteration);
    }

    /**
     * Performs back propagation on the convolution layer by computing the gradients of the
     * filters and the errors for the previous layer.
     * @param dLdO List of gradients of the loss with respect to the output of the layer.
     * @param iteration The current iteration number of the training process.
     */
    @Override
    public void backPropagation(List<double[][]> dLdO, int iteration) {

        // Initialize list to store changes in each filter
        List<double[][]> filtersDelta = new ArrayList<>();
        for(int f = 0; f < _filters.size(); f++){
            filtersDelta.add(new double[_filterSize][_filterSize]);
        }

        // Initialize list to store error from previous layer
        List<double[][]> dLdOPreviousLayer= new ArrayList<>();

        // Loop through each input in the previous layer
        for(int i = 0; i < _lastInput.size(); i++){

            // Initialize array to store error for this input
            double[][] errorForInput = new double[_inRows][_inCols];

            // Loop through each filter in the current layer
            for(int f = 0; f < _filters.size(); f++){

                // Get the current filter and the error for the current filter
                double[][] currFilter = _filters.get(f);
                double[][] error = dLdO.get(i*_filters.size() + f);

                // Calculate delta for the current filter
                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(_lastInput.get(i), spacedError, 1);
                double[][] delta = multiply(dLdF, _learningRate*-1);

                // Update filter delta for the current filter
                double[][] newTotalDelta = add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                // Calculate error for the previous layer
                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = add(errorForInput, fullConvolve(currFilter, flippedError));
            }

            // Add the error for this input to the list of errors for the previous layer
            dLdOPreviousLayer.add(errorForInput);
        }

        // Update filters for the current layer
        for(int f = 0; f < _filters.size(); f++){
            double[][] modified = add(filtersDelta.get(f), _filters.get(f));
            _filters.set(f,modified);
        }

        // Recursively call backpropagation on previous layer
        if(_previousLayer!= null){
            _previousLayer.backPropagation(dLdOPreviousLayer, iteration);
        }
    }

    /**
     * Flips the array horizontally
     * @param array the array to flip
     * @return the flipped array
     */
    public double[][] flipArrayHorizontal(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i = 0; i < rows; i++){
            System.arraycopy(array[i], 0, output[rows - i - 1], 0, cols);
        }
        return output;
    }

    /**
     * Flips the array vertically
     * @param array the array to flip
     * @return the flipped array
     */
    public double[][] flipArrayVertical(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                output[i][cols-j-1] = array[i][j];
            }
        }
        return output;
    }

    /**
     * Returns the full convolution of the input and filter matrices.
     * The output matrix dimensions are input.length + filter.length + 1 and input[0].length + filter[0].length + 1.
     *
     * @param input the input matrix to convolve
     * @param filter the filter matrix to convolve with the input
     * @return the output matrix resulting from the full convolution of the input and filter
     */
    private double[][] fullConvolve(double[][] input, double[][] filter) {

        int outRows = (input.length + filter.length) + 1;
        int outCols = (input[0].length + filter[0].length) + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for(int i = -fRows + 1; i < inRows; i ++){

            outCol = 0;

            for(int j = -fCols + 1; j < inCols; j++){

                double sum = 0.0;

                // Apply filter around this position
                for(int x = 0; x < fRows; x++){
                    for(int y = 0; y < fCols; y++){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        if(inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inRows && inputColIndex < inCols){
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum+= value;
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;

        }

        return output;

    }

    @Override
    public int getOutputLength() {
        return _filters.size()*_inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows-_filterSize)/_stepsize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols-_filterSize)/_stepsize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCols()*getOutputRows()*getOutputLength();
    }
}