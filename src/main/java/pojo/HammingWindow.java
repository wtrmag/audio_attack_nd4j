package pojo;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author ljj
 */
public class HammingWindow {
    public static double[] toHammingWindow(double[] values,int windowSize)
    {
        double[] hammingWindow=new double[windowSize];
        double[] newFrames=new double[values.length];
        for(int i=0;i<windowSize;i++)
        {
            hammingWindow[i]=0.54-0.46*Math.cos(2*Math.PI*i/(windowSize-1));
            //System.out.println("Hamming window "+i+": "+hammingWindow[i]);
        }
        double sumHamming=0;
        for(int i=0;i<windowSize;i++)
        {
            sumHamming+=hammingWindow[i];
        }
        //System.out.println(sumHamming);
        for(int i=0;i<windowSize;i++)
        {
            hammingWindow[i]=hammingWindow[i]/windowSize;
            //System.out.println("Hamming window "+i+": "+hammingWindow[i]);
        }
        for(int i=0;i<values.length;i++)
        {

            newFrames[i]=(values[i]*hammingWindow[i%windowSize]);
        }
        return newFrames;
    }
    public static double getHammingSquareSum(int windowSize)
    {
        double hammingSquareSum=0;
        for(int i=0;i<windowSize;i++)
        {
            hammingSquareSum+=Math.pow(0.54-0.46*Math.cos(2*Math.PI*i/(windowSize-1)),2);
            //System.out.println("Hamming window "+i+": "+hammingWindow[i]);
        }
        return hammingSquareSum;
    }


    public double[] toHammingWindow(int windowSize) {
        double[] hammingWindow=new double[windowSize];
    //    double[] newFrames=new double[values.length];
        for(int i=0;i<windowSize;i++)
        {
            hammingWindow[i]=0.54-0.46*Math.cos(2*Math.PI*i/(windowSize-1));
            //System.out.println("Hamming window "+i+": "+hammingWindow[i]);
        }
        double sumHamming=0;
        for(int i=0;i<windowSize;i++)
        {
            sumHamming+=hammingWindow[i];
        }
        //System.out.println(sumHamming);
        for(int i=0;i<windowSize;i++)
        {
            hammingWindow[i]=hammingWindow[i]/windowSize;
            //System.out.println("Hamming window "+i+": "+hammingWindow[i]);
        }

        return hammingWindow;
    }

    public static INDArray getHammingWindow(int window_size){
        INDArray n = Nd4j.arange(1-window_size, window_size, 2);
        return Nd4j.math.cos(n.mul(Math.PI).div(window_size-1)).mul(0.46).add(0.54);
    }
}
