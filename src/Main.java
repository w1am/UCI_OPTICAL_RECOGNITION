import data.DataReader;
import data.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Scanner;

import static java.util.Collections.shuffle;

public class Main {

    public void twoFoldTest(String foldOne, String foldTwo) {
        long SEED = 123;

        List<Image> imagesTrain = new DataReader().readData(foldOne, true);
        List<Image> imagesTest = new DataReader().readData(foldTwo, false);

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test size: " + imagesTest.size());

        double LEARNING_RATE = 0.41;

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
            double averageCost = net.train(epochIndex, imagesTrain);
            rate = net.test(imagesTest);

            if (rate > bestAccuracy) {
                bestAccuracy = rate;
                count = 0;
            } else {
                count++;
                if (count == wait) break;
            }

            // Round average cost to two decimal places
            DecimalFormat df = new DecimalFormat("#.###");

            System.out.println("epoch: " + epochIndex + ", cost: " + df.format(averageCost) + ", accuracy: " + df.format(rate));
        }

    }

    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Run first fold? (y/n)");
        String input = scanner.nextLine();
        if (input.equals("y")) {
            Main main = new Main();
            main.twoFoldTest("src/data/train.csv", "src/data/test.csv");
        }

        System.out.println();

        System.out.println("Run second fold? (y/n)");
        input = scanner.nextLine();
        if (input.equals("y")) {
            Main main = new Main();
            main.twoFoldTest("src/data/test.csv", "src/data/train.csv");
        }
    }
}
