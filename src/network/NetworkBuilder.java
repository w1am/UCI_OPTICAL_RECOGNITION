package network;

import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import layers.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

    private final int _inputRows;
    private final int _inputCols;
    private final double _scaleFactor;
    List<Layer> _layers;

    /**
     * Creates a new NetworkBuilder
     * @param _inputRows the number of rows in the input
     * @param _inputCols the number of columns in the input
     * @param _scaleFactor the scale factor to be used in the network
     */
    public NetworkBuilder(int _inputRows, int _inputCols, double _scaleFactor) {
        this._inputRows = _inputRows;
        this._inputCols = _inputCols;
        this._scaleFactor = _scaleFactor;
        _layers = new ArrayList<>();
    }

    /**
     * Adds a convolution layer to the network
     * @param numFilters the number of filters to be used in the convolution layer
     * @param filterSize the size of the filters to be used in the convolution layer
     * @param stepSize the step size to be used in the convolution layer
     * @param learningRate the learning rate to be used in the convolution layer
     * @param SEED the seed to be used in the convolution layer
     */
    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long SEED){
        if(_layers.isEmpty()){
            _layers.add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows, _inputCols, SEED, numFilters, learningRate));
        } else {
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED, numFilters, learningRate));
        }
    }

    /**
     * Adds a max pool layer to the network
     * @param windowSize the size of the window to be used in the max pool layer
     * @param stepSize the step size to be used in the max pool layer
     */
    public void addMaxPoolLayer(int windowSize, int stepSize){
        if(_layers.isEmpty()){
            _layers.add(new MaxPoolLayer(stepSize, windowSize, 1, _inputRows, _inputCols));
        } else {
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    /**
     * Adds a fully connected layer to the network
     * @param outLength the number of output elements to be used in the fully connected layer
     * @param learningRate the learning rate to be used in the fully connected layer
     * @param SEED the seed to be used in the fully connected layer
     */
    public void addFullyConnectedLayer(int outLength, double learningRate, long SEED){
        if(_layers.isEmpty()) {
            _layers.add(new FullyConnectedLayer(_inputCols * _inputRows, outLength, SEED, learningRate));
        } else {
            Layer prev = _layers.get(_layers.size() - 1);
            _layers.add(new FullyConnectedLayer(prev.getOutputElements(), outLength, SEED, learningRate));
        }

    }

    public NeuralNetwork build(){
        return new NeuralNetwork(_layers, _scaleFactor);
    }

}
