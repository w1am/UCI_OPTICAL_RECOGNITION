package helpers;

import data.Image;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Z-score normalization: This method standardizes the values so that the mean is 0 and the standard deviation is 1.
 * The formula is x' = (x - mean) / standard deviation.
 */
public class DataLoader {

    private static final double DEGREE_OF_ROTATION = 0.14;

    public double[][] rotateRight(double[][] data, int rows, int cols) {
        double[][] rotatedData = new double[rows][cols];
        int angle = (int) (rows * cols * DEGREE_OF_ROTATION);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int newRow = (i + angle) % rows;
                int newCol = (j + angle) % cols;
                rotatedData[newRow][newCol] = data[i][j];
            }
        }

        return rotatedData;
    }

    public double[][] rotateLeft(double[][] data, int rows, int cols) {
        double[][] rotated = new double[rows][cols];
        int rotationAmount = (int) (cols * DEGREE_OF_ROTATION);

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int rotatedCol = (col + rotationAmount) % cols;
                rotated[row][rotatedCol] = data[row][col];
            }
        }

        return rotated;
    }

    /**
     * Reads the data from the file and returns a list of images
     * @param path the path to the file
     * @param isTrain whether the data is for training or testing (used for two-fold test)
     * @return a list of images
     */
    public List<Image> readData(String path, Boolean isTrain) {
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
                        data[row][col] = Integer.parseInt(lineItems[i]);
                        allData.add(data[row][col]);
                        i++;
                    }
                }

                // Augment the data by rotating it
                if (isTrain) {
                    int direction = random.nextInt(2) + 1;
                    double[][] augmentedData = new double[rows][cols];
                    switch (direction) {
                        case 1 -> augmentedData = rotateLeft(data, rows, cols);
                        case 2 -> augmentedData = rotateRight(data, rows, cols);
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