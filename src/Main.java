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

        double LEARNING_RATE = 0.43;

        NetworkBuilder builder = new NetworkBuilder(8,8,2*100);
        builder.addConvolutionLayer(13, 3, 1, LEARNING_RATE, SEED);
        builder.addMaxPoolLayer(2,1);
        builder.addFullyConnectedLayer(10, LEARNING_RATE, SEED);

        NeuralNetwork net = builder.build();

        int epochs = 200;
        int wait = 15;
        double bestAccuracy = 0;
        int count = 0;

        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + rate);

        // Early stopping helps to prevent over-fitting by stopping the training
        // process when the validation loss stops improving.
        for(int epochIndex = 0; epochIndex < epochs; epochIndex++){
            shuffle(imagesTrain);
            double averageCost = net.train(imagesTrain);
            rate = net.test(imagesTest);

            if (rate > bestAccuracy) {
                bestAccuracy = rate;
                count = 0;
            } else {
                count++;
                if (count == wait) break;
            }

            System.out.println("epoch: " + epochIndex + ", cost: " + averageCost + ", accuracy: " + rate);
        }
    }
}
