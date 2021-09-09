package utils;

import org.nd4j.linalg.factory.ops.NDLinalg;
import pojo.HammingWindow;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import pojo.Variables;
import pojo.FFT;
import me.gommeantilegit.sonopy.DCT;

import java.io.File;

public class Tf_logits {
    int batch_size;
    long batch_size_F;
    int batch_size_mfc;
    long batch_size_mfc_F;
    int size;
    long size_F;
    INDArray empty_context;
    INDArray new_input_to_mfcc;
    INDArray features;
    INDArray features_tem1;
    INDArray features_tem2;

    public Tf_logits() {
    }

    public INDArray get_logits(INDArray new_input, INDArray lengths) {
        batch_size_F = new_input.size(0);
        batch_size = (int) batch_size_F;

        // 1. Compute the MFCCs for the input audio
        // (this is differentable with our implementation above)
        empty_context = Nd4j.zeros(new int[]{batch_size, 9, 26}, DataType.FLOAT);
        new_input_to_mfcc = compute_mfcc(new_input);
        features = Nd4j.concat(1, empty_context, new_input_to_mfcc, empty_context);

        //# 2. We get to see 9 frames at a time to make our decision,
        //# so concatenate them together.
        features = features.reshape((int) new_input.size(0), -1);

        INDArray[] f = new INDArray[(int) ((features.shape()[1] - 493) / 26 + 1)];
        for (int i = 0; i < features.shape()[1] - 493; i = i + 26) {
            f[i / 26] = features.get(NDArrayIndex.all(), NDArrayIndex.interval(i, i + 494));
        }
        features = Nd4j.stack(1, f);

        features = features.reshape(batch_size, -1, 19, 26);

        //# 3. Finally we process it with DeepSpeech
        //# We need to init DeepSpeech the first time we're called
        return PyCall.CallDeepSpeech(features, lengths);
    }

    public INDArray compute_mfcc(INDArray audio) {

        INDArray feat;
        INDArray audio1;
        INDArray audio2;
        INDArray audio3;
        INDArray audio4;
        INDArray windowed1;
        INDArray windowed2;
        INDArray windowed;
        double[] window_tmp;
        INDArray window;

        INDArray ffted;
        INDArray window_op;

        INDArray energy;
        INDArray filters;

        NDLinalg linalg = new NDLinalg();

        batch_size_mfc_F = audio.size(0);
        size_F = audio.size(1);
        batch_size_mfc = (int) batch_size_mfc_F;
        size = (int) size_F;
        //audio = tf.cast(audio, tf.float32)
        //数据类型由传参阶段数据转换完成，audio在nd下已为浮点型

        //# 1. Pre-emphasizer, a high-pass filter
        audio1 = audio.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 1));
        audio2 = audio.get(NDArrayIndex.all(), NDArrayIndex.interval(1, (int) audio.size(1)));
        audio3 = audio.get(NDArrayIndex.all(), NDArrayIndex.interval(0, (int) audio.size(1) - 1));
        audio4 = Nd4j.zeros(batch_size_mfc, 512);
        audio3 = audio3.mul(0.97);
        audio = Nd4j.concat(1, audio1, audio2.sub(audio3), audio4);

        //# 2. windowing into frames of 512 samples, overlapping
        INDArray[] win = new INDArray[(size - 320) / 320 + 1];
        for (int i = 0; i < size - 320; i = i + 320) {
            win[i/320] = audio.get(NDArrayIndex.all(), NDArrayIndex.interval(i, i + 512));
        }
        windowed = Nd4j.stack(1, win);

        window = HammingWindow.getHammingWindow(512);
        windowed = Nd4j.math.mul(windowed, window.castTo(DataType.FLOAT));

        //# 3. Take the FFT to convert to frequency space
        DataBuffer windowed_tmp1 = windowed.data();
        int k = (int) windowed_tmp1.length();
        double[] windowed_tmp2 = new double[k];
        for (int i = 0; i < k; i++) {
            windowed_tmp2[i] = windowed_tmp1.getDouble(i);
        }

        double[][] ffted_tmp = new double[(int)windowed.size(1)][(int)windowed.size(2)];

        for(int i=0;i<(int)windowed.size(1);i++) {
            for (int j = 0; j < (int) windowed.size(2); j++) {
                ffted_tmp[i][j] = windowed_tmp2[i * j + j];
            }
        }
        double[][][] ffted_tmp_data=new double[(int)windowed.size(1)][2][(int)windowed.size(2)/2+1];
        for(int i=0;i<(int)windowed.size(1);i++) {
            ffted_tmp_data[i] = FFT.rfft(ffted_tmp[i], ffted_tmp[i].length);
        }
        double[][] ffted_tmp_final=new double[(int)windowed.size(1)][(int)windowed.size(2)/2+1];
        for(int i=0;i<(int)windowed.size(1);i++)
        {
            ffted_tmp_final[i]=FFT.AbsAndSquare(ffted_tmp_data[i][0],ffted_tmp_data[i][1]);
        }

        ffted = Nd4j.stack(0, Nd4j.create(ffted_tmp_final));
        ffted = ffted.mul(1.0 / 512);

        //# 4. Compute the Mel windowing of the FFT
        energy = Nd4j.math.asum(ffted, 2);
        File file_tmp = new File(Variables.NPY_PATH);
        filters = Nd4j.createFromNpyFile(file_tmp).transpose();

        INDArray[] t = new INDArray[batch_size];
        for (int i = 0; i < batch_size; i++) {
            t[i] = filters;
        }
        feat = Nd4j.stack(0, t);
        feat = linalg.mmul(ffted, feat);
        //feat=feat+eps
        //***调试增加eps值
        //***注意此处[filters]*batch_size他们的地址相同，更改一个同时更改所有

        //# 5. Take the DCT again, because why not
        feat = Nd4j.math.log(feat);
        DCT d = new DCT();

        int a, b, c;
        a = (int) feat.size(0);
        b = (int) feat.size(1);
        c = (int) feat.size(2);
        DataBuffer dct_tmp1 = feat.data();

        float[][][] dct_tmp2 = new float[a][b][c];
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                for (int kk = 0; kk < c; kk++) {
                    dct_tmp2[i][j][kk] = dct_tmp1.getFloat(i * b * c + j * c + kk);
                }
            }
        }
        float[][][] feat_tmp = new float[a][b][c];
        for (int i = 0; i < a; i++) {
            feat_tmp[i] = DCT.dct(dct_tmp2[i], true);

        }
        feat = Nd4j.create(feat_tmp).get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 26));

        //# 6. Amplify high frequencies for some reason
        int ncoeff = (int) feat.size(2);
        double[] n_tmp = new double[ncoeff];
        for (int i = 0; i < ncoeff - 1; i++) {
            n_tmp[i] = i;
        }
        INDArray n = Nd4j.create(n_tmp);
        INDArray lift = Nd4j.math.sin(n.mul(Math.PI / 22)).mul(22 / 2.0).add(1);
        feat = Nd4j.math.mul(lift.castTo(DataType.FLOAT), feat);
        int width = (int) feat.size(1);

        //# 7. And now stick the energy next to the features
        INDArray log_e = Nd4j.math.log(energy);
        INDArray e = log_e.dup().reshape(-1, width, 1).castTo(DataType.FLOAT);
        INDArray f = feat.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, feat.size(2)));
        feat = Nd4j.concat(2, e, f);

        return feat;

    }

}
