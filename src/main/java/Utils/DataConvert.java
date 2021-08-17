package Utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Operand;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.op.Operands;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.types.family.TType;

public class DataConvert {

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
