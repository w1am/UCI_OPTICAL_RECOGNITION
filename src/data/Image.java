package data;

public record Image(double[][] data, int label) {

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    /**
     * Returns a string representation of the image
     *
     * @return string representation
     */
    @Override
    public String toString() {

        StringBuilder s = new StringBuilder(label + ", \n");

        for (double[] datum : data) {
            for (int j = 0; j < data[0].length; j++) {
                s.append(datum[j]).append(", ");
            }
            s.append("\n");
        }

        return s.toString();
    }
}
