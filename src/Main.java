import data.DataReader;
import data.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;

import java.io.IOException;
import java.util.List;

import static java.util.Collections.shuffle;

public class Main {

    public static void main(String[] args) throws IOException {

        long SEED = 123;

        System.out.println("Starting data loading...");

        List<Image> imagesTrain = new DataReader().readData("src/data/train.csv");
        List<Image> imagesTest = new DataReader().readData("src/data/test.csv");

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test size: " + imagesTest.size());

//        NetworkBuilder builder = new NetworkBuilder(8,8,12*100);
//        builder.addConvolutionLayer(15, 4, 1, 0.4, SEED);
//        builder.addMaxPoolLayer(2,1);
//        builder.addFullyConnectedLayer(10, 0.4, SEED);

//        NetworkBuilder builder = new NetworkBuilder(8,8,12*100);
//        builder.addConvolutionLayer(14, 3, 1, 0.32, SEED);
//        builder.addMaxPoolLayer(2,1);
//        builder.addFullyConnectedLayer(10, 0.32, SEED);

//        NetworkBuilder builder = new NetworkBuilder(8,8,5*100);
//        builder.addConvolutionLayer(12, 4, 1, 0.43, SEED);
//        builder.addMaxPoolLayer(2,1);
//        builder.addFullyConnectedLayer(10, 0.43, SEED);

//        NetworkBuilder builder = new NetworkBuilder(8,8,100);
//        builder.addConvolutionLayer(20, 2, 1, 0.43, SEED);
//        builder.addMaxPoolLayer(2,1);
//        builder.addFullyConnectedLayer(10, 0.43, SEED);

        NetworkBuilder builder = new NetworkBuilder(8,8,100);
        builder.addConvolutionLayer(20, 2, 1, 0.41, SEED);
        builder.addMaxPoolLayer(2,1);
        builder.addFullyConnectedLayer(10, 0.41, SEED);

        NeuralNetwork net = builder.build();

        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + rate);

        int epochs = 200;

        for(int i = 0; i < epochs; i++){
            shuffle(imagesTrain);
            net.train(imagesTrain);
            rate = net.test(imagesTest);
            System.out.println("Success rate after round " + i + ": " + rate);
        }
    }
}
