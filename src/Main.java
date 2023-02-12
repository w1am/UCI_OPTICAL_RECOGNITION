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

        NetworkBuilder builder = new NetworkBuilder(8,8,100);
        builder.addConvolutionLayer(18, 2, 1, 0.35, SEED);
        builder.addMaxPoolLayer(2,1);
        builder.addFullyConnectedLayer(10, 0.35, SEED);

        NeuralNetwork net = builder.build();

        int epochs = 100;
        int wait = 10;
        double bestAccuracy = 0;
        int count = 0;

        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + rate);

        for(int i = 0; i < epochs; i++){
            shuffle(imagesTrain);
            net.train(imagesTrain);
            rate = net.test(imagesTest);

            if (rate > bestAccuracy) {
                bestAccuracy = rate;
                count = 0;
            } else {
                // Early stopping helps to prevent over-fitting by stopping the training
                // process when the validation loss stops improving.
                count++;
                if (count == wait) break;
            }

            System.out.println("Success rate after round " + i + ": " + rate);
        }
    }
}
