import helpers.DataLoader;
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
        List<Image> imagesTrain = new DataLoader().readData(foldOne, true);
        List<Image> imagesTest = new DataLoader().readData(foldTwo, false);

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test size: " + imagesTest.size());

        NetworkBuilder builder = new NetworkBuilder(Config.INPUT_ROWS, Config.INPUT_COLS, Config.SCALE_FACTOR);
        builder.addConvolutionLayer(Config.NUM_FILTERS, Config.FILTER_SIZE, Config.STEP_SIZE, Config.LEARNING_RATE, Config.SEED);
        builder.addMaxPoolLayer(Config.WINDOW_SIZE, Config.STEP_SIZE);
        builder.addFullyConnectedLayer(Config.OUTPUT_LENGTH, Config.LEARNING_RATE, Config.SEED);

        NeuralNetwork net = builder.build();

        double bestAccuracy = 0;
        int count = 0;

        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + rate);

        // Early stopping helps to prevent over-fitting by stopping the training
        // process when the validation loss stops improving.
        for(int epochIndex = 0; epochIndex < Config.EPOCHS; epochIndex++){
            shuffle(imagesTrain);
            double averageCost = net.train(epochIndex, imagesTrain);
            rate = net.test(imagesTest);

            if (rate > bestAccuracy) {
                bestAccuracy = rate;
                count = 0;
            } else {
                count++;
                if (count == Config.WAIT) break;
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
