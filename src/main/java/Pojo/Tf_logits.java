
package Pojo;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Tf_logits {
    int batch_size;
    long batch_size_F;
    int batch_size_mfc;
    long batch_size_mfc_F;
    int size;
    long size_F;
    INDArray new_input_cov;
    INDArray empty_context;
    INDArray new_input_to_mfcc;
    INDArray features;
    INDArray features_tem1;
    INDArray features_tem2;
    ArrayList<Boolean> first_cov;


    public Tf_logits(){
    //new_input
        batch_size_F = new_input_cov.size(0);
        batch_size=(int)batch_size_F;

        // 1. Compute the MFCCs for the input audio
        // (this is differentable with our implementation above)
        empty_context=Nd4j.zeros(new int[]{batch_size,9,26}, DataType.FLOAT);
        new_input_to_mfcc=compute_mfcc(new_input_cov);
        features=Nd4j.concat(1,empty_context,new_input_to_mfcc,empty_context);

        //# 2. We get to see 9 frames at a time to make our decision,
        //# so concatenate them together.
        features=features.reshape((int)new_input_cov.size(0),-1);

        features_tem1=features.get(NDArrayIndex.all(),NDArrayIndex.interval(0,19*26));
        for(int i=26;i<=features.shape()[1]-19*26+1;i=i+26)
        {
            features_tem2=features.get(NDArrayIndex.all(),NDArrayIndex.interval(i,i+19*26));
            features_tem1=Nd4j.stack(1,features_tem1,features_tem2);
        }
        features=features_tem1;

        features=features.reshape(batch_size,-1,19,26);

        //# 3. Finally we process it with DeepSpeech
        //# We need to init DeepSpeech the first time we're called
        if(first_cov.isEmpty())
        {
            first_cov.add(false);

            deep
        }



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

        batch_size_mfc_F=audio.size(0);
        size_F= audio.size(1);
        batch_size_mfc=(int)batch_size_mfc_F;
        size=(int)size_F;
        //audio = tf.cast(audio, tf.float32)
        //数据类型由传参阶段数据转换完成，audio在nd下已为浮点型

        //# 1. Pre-emphasizer, a high-pass filter
        audio1=audio.get(NDArrayIndex.all(),NDArrayIndex.interval(0,1));
        audio2=audio.get(NDArrayIndex.all(),NDArrayIndex.interval(1,(int)audio.size(1)));
        audio3=audio.get(NDArrayIndex.all(),NDArrayIndex.interval(0,(int)audio.size(1)-1));
        audio4=Nd4j.zeros(batch_size_mfc,512);
        audio3=audio3.mul(0.97);
        audio=Nd4j.concat(1,audio1,audio2.sub(audio3),audio4);

        //# 2. windowing into frames of 512 samples, overlapping
       windowed1=audio.get(NDArrayIndex.all(),NDArrayIndex.interval(0,512));
        for(int i=320;i<=size-320;i=i+320)
        {
            windowed2=audio.get(NDArrayIndex.all(),NDArrayIndex.interval(i,i+512));
            windowed1=Nd4j.stack(1,windowed1,windowed2);
        }
        windowed=windowed1;
        HammingWindow window1=new HammingWindow();
        window_tmp=window1.toHammingWindow(512);
        window=Nd4j.create(window_tmp);
        windowed=windowed.mmul(window);

        //# 3. Take the FFT to convert to frequency space
        //ffted = tf.spectral.rfft(windowed, [512])
        //ffted = 1.0 / 512 * tf.square(tf.abs(ffted))
        DataBuffer windowed_tmp1=windowed.data();
        int k=windowed_tmp1.getElementSize();
        double[] windowed_tmp2=new double[k];
        for(int i=0;i<k;i++)
        {
            windowed_tmp2[i]=windowed_tmp1.getDouble(i);
        }
        Complex[] ffted_tmp=new Complex[(int)windowed.size(0)];
        for(int i=0;i<(int)windowed.size(0);i++)
        {
            ffted_tmp[i].re=windowed_tmp2[i];
            ffted_tmp[i].im=0;
        }
        double[] ffted_cur=new double[(int)windowed.size(0)];
        ffted_tmp=Complex.fft(ffted_tmp);
        ffted_cur=Complex.Covlex(ffted_tmp);
        ffted=Nd4j.create(ffted_cur);
        ffted=Nd4j.math.square(Nd4j.math.abs(ffted)).mul(1.0/512);
    //
    //    Graph graph=new Graph();
    //    Ops tf=Ops.create(graph);

    //    window_op=Conver_Op(windowed);
    //    ffted=tf.spectral.rfft(window_op,512);
    //    ffted=1.0/512*tf.square(tf.abs(ffted));
    //

        //# 4. Compute the Mel windowing of the FFT
        //energy = tf.reduce_sum(ffted,axis=2)+np.finfo(float).eps ;
        //filters = np.load("filterbanks.npy").T
        //feat = tf.matmul(ffted, np.array([filters]*batch_size,dtype=np.float32))+np.finfo(float).eps
        energy=Nd4j.math.asum(ffted,2);
        filters;
        feat=filters;
        for(int i=batch_size-1;i>0;i--)
        {
            feat=Nd4j.concat(0,feat,feat);
        }
        feat=ffted.mmul(feat);
        //feat=feat+eps
        //***调试增加eps值
        //***注意此处[filters]*batch_size他们的地址相同，更改一个同时更改所有

        //# 5. Take the DCT again, because why not
        //feat = tf.log(feat)
        //feat = tf.spectral.dct(feat, type=2, norm='ortho')[:,:,:26]
        feat=Nd4j.math.log(feat);
        DCT d=new DCT();

        int a,b,c;
        a=(int)feat.size(0);
        b=(int)feat.size(1);
        c=(int)feat.size(2);
        DataBuffer dct_tmp1=feat.data();
        //int tmp=dct_tmp1.getElementSize();
        double[][][] dct_tmp2=new double[a][b][c];
        for(int i=0;i<a;i++)
            for(int j=0;j<b;j++)
                for(int kk=0;kk<c;kk++)
                    dct_tmp2[i][j][kk]=dct_tmp1.getDouble(i*b*c+j*c+kk);

        //double[] feat_tmp=d.dct(dct_tmp2);
        double[][][] feat_tmp=new double[a][b][c];
        for(int i=0;i<a;i++)
            feat_tmp[i]=d.process(dct_tmp2[i]);
        feat=Nd4j.create(feat_tmp);

        //# 6. Amplify high frequencies for some reason
        //_,nframes,ncoeff = feat.get_shape().as_list()
        //n = np.arange(ncoeff)
        //lift = 1 + (22/2.)*np.sin(np.pi*n/22)
        //feat = lift*feat
        //width = feat.get_shape().as_list()[1]
        int nframes=(int)feat.size(1);
        int ncoeff=(int)feat.size(2);
        double[] n_tmp=new double[ncoeff];
        for(int i=0;i<ncoeff-1;i++)
            n_tmp[i]= i;
        INDArray n=Nd4j.create(n_tmp);
        INDArray lift=Nd4j.math.sin(n.mul(Math.PI/22)).mul(22/2.0).add(1);
        feat=lift.mmul(feat);
        int width=(int)feat.size(1);

        //# 7. And now stick the energy next to the features
        //feat = tf.concat((tf.reshape(tf.log(energy),(-1,width,1)), feat[:, :, 1:]), axis=2)
        feat=Nd4j.concat(2,Nd4j.math.log(energy).reshape(-1,width,1),feat.dup().get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(1,feat.size(2))));

        return feat;

    }

}


