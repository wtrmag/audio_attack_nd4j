package pojo;

import java.util.Arrays;

/**
 * Fast Fourier Transform by Danny Su and Hanns Holger Rutz (adjusted to match np.fft.rfft by GommeAntiLegit)<br>
 * <b>description:</b> FFT class for real signals. Upon entry, N contains the
 * numbers of points in the DFT, real[] and imaginary[] contain the real and
 * imaginary parts of the input. Upon return, real[] and imaginary[] contain the
 * DFT output. All signals run from 0 to N - 1<br>
 * <b>input:</b> speech signal<br>
 * <b>output:</b> real and imaginary part of DFT output
 *
 * @author Danny Su
 * @author Hanns Holger Rutz
 * @author GommeAntiLegit
 */
public class FFT {

    /**
     * Performs Fast Fourier Transformation (adjusted to match np.fft.rfft)
     * @see FFT
     */
    public static double[][] rfft(double[] signal, int numPoints) {
        // initialize real & imag array
        double[] real = Arrays.copyOf(signal, signal.length);
        double[] imag = new double[numPoints];

        // FFT using the real & imag array
        final double pi = (double) Math.PI;
        final int numStages = (int) (Math.log(numPoints) / Math.log(2));
        final int halfNumPoints = numPoints >> 1;
        int j = halfNumPoints;
        // FFT time domain decomposition carried out by "bit reversal sorting"
        // algorithm
        int k;
        for (int i = 1; i < numPoints - 2; i++) {
            if (i < j) {
                // swap
                double tempReal = real[j];
                double tempImag = imag[j];
                real[j] = real[i];
                imag[j] = imag[i];
                real[i] = tempReal;
                imag[i] = tempImag;
            }
            k = halfNumPoints;
            while (k <= j) {
                j -= k;
                k >>= 1;
            }
            j += k;
        }

        // loop for each stage
        for (int stage = 1; stage <= numStages; stage++) {
            int LE = 1;
            for (int i = 0; i < stage; i++) {
                LE <<= 1;
            }
            final int LE2 = LE >> 1;
            double UR = 1;
            double UI = 0;
            // calculate sine & cosine values
            final double SR = (double) Math.cos(pi / LE2);
            final double SI = (double) -Math.sin(pi / LE2);
            // loop for each sub DFT
            for (int subDFT = 1; subDFT <= LE2; subDFT++) {
                // loop for each butterfly
                for (int butterfly = subDFT - 1; butterfly <= numPoints - 1; butterfly += LE) {
                    int ip = butterfly + LE2;
                    // butterfly calculation
                    double tempReal = real[ip] * UR - imag[ip] * UI;
                    double tempImag = real[ip] * UI + imag[ip] * UR;
                    real[ip] = real[butterfly] - tempReal;
                    imag[ip] = imag[butterfly] - tempImag;
                    real[butterfly] += tempReal;
                    imag[butterfly] += tempImag;
                }

                double tempUR = UR;
                UR = tempUR * SR - UI * SI;
                UI = tempUR * SI + UI * SR;
            }
        }
        return new double[][]{Arrays.copyOfRange(real, 0, numPoints / 2 + 1), Arrays.copyOfRange(imag, 0, numPoints / 2 + 1)};
    }

    public static double[] AbsAndSquare(double[] re,double[] im){
        double[] result=new double[re.length];
        for(int i=0;i<re.length;i++)
        {
            result[i]=Math.pow(re[i],2)+Math.pow(im[i],2);
        }
        return result;
    }

}
