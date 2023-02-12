package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Z-score normalization: This method standardizes the values so that the mean is 0 and the standard deviation is 1.
 * The formula is x' = (x - mean) / standard deviation.
 */
public class DataReader {

    public List<Image> readData(String path) {
        List<Image> images = new ArrayList<>();
        List<Double> allData = new ArrayList<>();

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
