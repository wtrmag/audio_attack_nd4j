package utils;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Operand;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

import java.io.*;
import java.util.List;

public class DataConvert {

    public static final String[] NPY_NAME = {"features.npy", "lengths.npy", "logits.npy"};

    /**
     * @author wtr
     * @return
     */
    public static boolean export(String path, INDArray features, INDArray lengths){
        boolean isSucess=false;

        try {
            File file = new File(path);
            File f1 = new File(file, NPY_NAME[0]);
            File f2 = new File(file, NPY_NAME[1]);
            if (!file.exists()){
                file.mkdirs();
            }
            if (!f1.exists()){
                f1.createNewFile();
            }
            if (!f2.exists()){
                f2.createNewFile();
            }
            Nd4j.writeAsNumpy(features, f1);
            Nd4j.writeAsNumpy(lengths, f2);
            isSucess=true;
        } catch (Exception e) {
            isSucess=false;
        }
        return isSucess;
    }

    public static Operand nd2tf(Ops tf, INDArray ndarray){
        return tf.constant(ndarray.toIntVector());
    }

    public static  <T extends TType> INDArray tf2nd(Ops tf, Operand<T> tensor){
        int size = (int) tensor.size();
        ByteDataBuffer dataBuffer = tensor.asTensor().asRawTensor().data();

        switch (tensor.rank()){
            case 1 :
                int[] array = new int[size];

                break;
            case 2 :
                break;
            case 3 :
                break;
            case 4 :
                break;
            default:
                System.err.println("Nd4j does not support!");
        }

        return null;
    }

}
