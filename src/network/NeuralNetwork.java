package network;

import data.Image;
import layers.Layer;

import java.util.ArrayList;
import java.util.List;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class NeuralNetwork {

    List<Layer> _layers;
    double scaleFactor;

    public NeuralNetwork(List<Layer> _layers, double scaleFactor) {
        this._layers = _layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers() {

        if(_layers.size() <= 1) return;

        for(int i = 0; i < _layers.size(); i++){
            if(i == 0){
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            } else if (i == _layers.size()-1){
                _layers.get(i).set_previousLayer(_layers.get(i-1));
            } else {
                _layers.get(i).set_previousLayer(_layers.get(i-1));
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            }
        }

    }

    /**
     * Get error for an image based on the output of the network and the correct answer
     * @param networkOutput the output of the network
     * @param correctAnswer the correct answer
     * @return the error
     */
    public double[] getErrors(double[] networkOutput, int correctAnswer){
        int numClasses = networkOutput.length;
        double[] expected = new double[numClasses];

        // Set the expected output to 1 for the correct answer
        expected[correctAnswer] = 1;

        // Return the difference between the expected and the actual output.
        return add(networkOutput, multiply(expected, -1));
    }

    private int getMaxIndex(double[] in){

        double max = 0;
        int index = 0;

        for(int i = 0; i < in.length; i++){
            if(in[i] >= max){
                max = in[i];
                index = i;
            }

        }

        return index;
    }

    public int guess(Image image){
        List<double[][]> inList = new ArrayList<>();

        inList.add(multiply(image.getData(), (1.0 / scaleFactor)));

        double[] out = _layers.get(0).getOutput(inList);

        return getMaxIndex(out);
    }

    public float test(List<Image> images) {
        int correct = 0;

        for (Image img: images) {
            int guess = guess(img);

            if (guess == img.getLabel()) {
                correct++;
            }
        }

        return((float) correct / images.size());
    }

    /**
     * Trains the network on a list of images
     * @param images the list of images to train on
     */
    public void train(List<Image> images) {

        for (Image img:images) {

            List<double[][]> inList = new ArrayList<>();

            // Scale the image down to avoid big numbers
            inList.add(multiply(img.getData(), (1.0 / scaleFactor)));

            // Get the output of the network
            double[] out = _layers.get(0).getOutput(inList);

            // Calculate the error based on the output and the label of the image
            double[] dldO = getErrors(out, img.getLabel());

            // Perform back propagation on fully connected layer
            _layers.get((_layers.size() - 1)).backPropagation(dldO);

        }

    }

}
