package layers;

import java.util.List;
import java.util.Random;

public class FullyConnected extends Layer{

    private final long SEED;

    private final double[][] _weights;
    private final int _inLength;
    private final int _outLength;
    private final double _learningRate;

    private double[] lastZ;
    private double[] lastX;


    public FullyConnected(int _inLength, int _outLength, long SEED, double learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this._learningRate = learningRate;

        _weights = new double[_inLength][_outLength];
        setRandomWeights();
    }

    /**
     * Performs a forward pass on a fully connected layer.
     * x -w-> z -f-> y <- dL/dy
     * @param input The input to the layer
     * @return The output of the layer
     */
    public double[] fullyConnectedForwardPass(double[] input){

        lastX = input;

        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        // For each input node, calculate the dot product with the
        // weights and apply activation function.
        for(int currentInputIndex = 0; currentInputIndex < _inLength; currentInputIndex++){
            for(int currentOutputIndex = 0; currentOutputIndex < _outLength; currentOutputIndex++){
                // Calculate the dot product of the input and the weights
                z[currentOutputIndex] += input[currentInputIndex] * _weights[currentInputIndex][currentOutputIndex];

                // Apply activation function
                out[currentOutputIndex] = leakyReLU(z[currentOutputIndex]);
            }
        }

        // Store the dot product for use in backpropagation
        lastZ = z;

        return out;

    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        return fullyConnectedForwardPass(input);
    }

    /**
     * Performs a backpropagation step on a fully connected layer.
     * @param dLdO The derivative of the loss with respect to the output of the layer
     */
    @Override
    public void backPropagation(double[] dLdO, int iteration) {

        double[] dLdX = new double[_inLength];

        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        // Calculate the adaptive learning rate based on the current iteration
        double alpha = 0.5 / (1 + iteration / 1000);
        double currentLearningRate = alpha * _learningRate;

        for(int k = 0; k < _inLength; k++){

            // Initialize a variable to store the sum of the gradients of the loss with respect
            // to the output of this layer (dLdO) times the derivatives of the output
            // with respect to the input (dOdz) times the weights (dzdw)
            double dLdX_sum = 0;

            for(int j = 0; j < _outLength; j++){

                // Calculate the derivative of the output of this layer with respect to the input (dOdz), the derivative of
                // the input of this layer with respect to the weight (dzdw), and the derivative of the output
                // of this layer with respect to the weight (dzdx)
                dOdz = derivativeLeakyReLU(lastZ[j]);
                dzdw = lastX[k];
                dzdx = _weights[k][j];

                // Calculate the gradient of the loss with respect to the weight (dLdw) using the chain rule
                dLdw = dLdO[j]*dOdz*dzdw;

                // Update the weight using the gradient descent algorithm with the calculated learning rate
                _weights[k][j] -= dLdw * currentLearningRate;

                // Add the gradient of the loss with respect to the output of this layer times the derivatives
                // of the output with respect to the input and the weight to the sum
                dLdX_sum += dLdO[j]*dOdz*dzdx;

            }

            // Store the sum of the gradients of the loss with respect to the output of this layer
            // times the derivatives of the output with respect to the input and the weight in the dLdX array
            dLdX[k] = dLdX_sum;
        }

        if (_previousLayer != null) _previousLayer.backPropagation(dLdX, iteration);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO, int iteration) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector, iteration);
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outLength;
    }

    public void setRandomWeights(){
        Random random = new Random(SEED);

        for(int i = 0; i < _inLength; i++){
            for(int j =0; j < _outLength; j++){
                _weights[i][j] = random.nextGaussian();
            }
        }
    }

    /**
     * Applies the leaky ReLU activation function to the input.
     * @param input The input to the activation function
     * @return The output of the activation function
     */
    public double leakyReLU(double input) {
        final double alpha = 0.01;
        if (input > 0) {
            return input;
        } else {
            return alpha * input;
        }
    }

    /**
     * Calculates the derivative of the leaky ReLU activation function.
     * @param input The input to the activation function
     * @return The output of the activation function
     */
    public double derivativeLeakyReLU(double input) {
        final double alpha = 0.01;
        if (input > 0) {
            return 1.0;
        } else {
            return alpha;
        }
    }

}
