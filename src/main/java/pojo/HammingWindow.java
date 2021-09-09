package pojo;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author ljj
 */
public class HammingWindow {

    public static INDArray getHammingWindow(int window_size){
        INDArray n = Nd4j.arange(1-window_size, window_size, 2);
        return Nd4j.math.cos(n.mul(Math.PI).div(window_size-1)).mul(0.46).add(0.54);
    }
}
