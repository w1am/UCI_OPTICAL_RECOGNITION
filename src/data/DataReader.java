package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Z-score normalization: This method standardizes the values so that the mean is 0 and the standard deviation is 1.
 * The formula is x' = (x - mean) / standard deviation.
 */
public class DataReader {

    private double[][] rotateClockwise(double[][] data, int rows, int cols) {
        double[][] rotatedData = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                rotatedData[j][rows - 1 - i] = data[i][j];
            }
        }

        return rotatedData;
    }

    private double[][] rotateCounterClockwise(double[][] data, int rows, int cols) {
        double[][] rotatedData = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                rotatedData[cols - 1 - j][i] = data[i][j];
            }
        }

        return rotatedData;
    }

    private double[][] flipHorizontally(double[][] data, int rows, int cols) {
        double[][] flippedData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flippedData[i][cols - 1 - j] = data[i][j];
            }
        }

        return flippedData;
    }

    private double[][] flipVertically(double[][] data, int rows, int cols) {
        double[][] flippedData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, flippedData[rows - 1 - i], 0, cols);
        }

        return flippedData;
    }

    public List<Image> readData(String path) throws IOException {
        List<Image> images = new ArrayList<>();
        List<Double> allData = new ArrayList<>();
        Random random = new Random();

        try (BufferedReader dataReader = new BufferedReader(new FileReader(path))) {
            String line;
            int rows = 8;
            int cols = 8;
            int label;
            int i;

            while ((line = dataReader.readLine()) != null) {
                String[] lineItems = line.split(",");
                double[][] data = new double[rows][cols];
                label = Integer.parseInt(lineItems[lineItems.length - 1]);
                i = 0;

                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        data[row][col] = (double) Integer.parseInt(lineItems[i]);
                        allData.add(data[row][col]);
                        i++;
                    }
                }

                if (path.equals("src/data/train.csv")) {
                    int flip = random.nextInt(4) + 1;
                    double[][] augmentedData = new double[rows][cols];
                    switch (flip) {
                        case 1 -> augmentedData = flipHorizontally(data, rows, cols);
                        case 2 -> augmentedData = flipVertically(data, rows, cols);
                        case 3 -> augmentedData = rotateClockwise(data, rows, cols);
                        case 4 -> augmentedData = rotateCounterClockwise(data, rows, cols);
                    }

                    images.add(new Image(augmentedData, label));
                }

                images.add(new Image(data, label));
            }
        } catch (Exception e) {
            System.out.println(e);
            throw new IllegalArgumentException("File not found " + path);
        }

        // Calculate the mean and standard deviation of all the data values
        double mean = allData.stream().mapToDouble(Double::doubleValue).average().getAsDouble();
        double stdDev = Math.sqrt(allData.stream().mapToDouble(d -> Math.pow(d - mean, 2)).average().getAsDouble());

        // Normalize the data values
        for (Image image : images) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    image.getData()[i][j] = (image.getData()[i][j] - mean) / stdDev;
                }
            }
        }

        return images;
    }

}