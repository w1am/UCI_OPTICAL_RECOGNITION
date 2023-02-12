package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {

    public List<Image> readData(String path){

        List<Image> images = new ArrayList<>();

        try (BufferedReader dataReader = new BufferedReader(new FileReader(path))){

            String line;

            while((line = dataReader.readLine()) != null){
                String[] lineItems = line.split(",");

                int rows = 8;
                int cols = 8;
                double[][] data = new double[rows][cols];
                int label = Integer.parseInt(lineItems[lineItems.length - 1]);

                int i = 0;

                for(int row = 0; row < rows; row++){
                    for(int col = 0; col < cols; col++){
                        data[row][col] = (double) Integer.parseInt(lineItems[i]);
                        i++;
                    }
                }

                images.add(new Image(data, label));

            }

        } catch (Exception e){
            System.out.println(e);
            throw new IllegalArgumentException("File not found " + path);
        }

        return images;

    }

}
