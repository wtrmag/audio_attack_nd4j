package utils;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.RawTensor;
import org.tensorflow.Session;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.op.Ops;

import java.io.File;

public class DataConvert {

//    public static final String[] NPY_NAME = {"features.npy", "lengths.npy", "logits.npy", "loss.npy", "var.npy", "delta.npy"};
    public static final String[] NPY_NAME = {"data1.npy", "data2.npy", "logit.npy" ,"result.npy"};

    /**
     * @author wtr
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

    public static Operand nd2tf(Ops tf, INDArray ndarray) {
        int rank = ndarray.rank();
        DataType type = ndarray.dataType();

        if (rank == 1){
            if (type.isIntType() || type == DataType.BOOL){
                return tf.constant(ndarray.toIntVector());
            }else if (type.isFPType()){
                return tf.constant(ndarray.toFloatVector());
            }
        }else if (rank == 2){
            if (type.isIntType() || type == DataType.BOOL){
                return tf.constant(ndarray.toIntMatrix());
            }else if (type.isFPType()){
                return tf.constant(ndarray.toFloatMatrix());
            }
        }else if(rank == 3){
            String data = ndarray.toStringFull();
            long[] shape = ndarray.shape();
            if (type.isIntType()){
                int[][][] array = strArraysToIntNum(data, (int) shape[0], (int) shape[1], (int) shape[2]);
                return tf.constant(array);
            }else if(type.isFPType()) {
                float[][][] array = strArraysToFloatNum(data, (int) shape[0], (int) shape[1], (int) shape[2]);
                return tf.constant(array);
            }
        }

        return null;
    }

    public static INDArray tf2nd(Session session, Ops tf, Output tensor) throws Exception {
        int[] INT_VALUE = new int[]{3, 5, 6,7, 9, 10};
        int[] FLOAT_VALUE = new int[]{1, 2};

        RawTensor rawTensor = session.runner().fetch(tensor).run().get(0).asRawTensor();
        int rank = rawTensor.rank();
        long[] shape = rawTensor.shape().asArray();
        int size = (int) rawTensor.size();
        ByteDataBuffer dataBuffer = rawTensor.data();
        int type_value = tensor.dataType().getNumber();

        int[] array1 = new int[size];
        float[] array2 = new float[size];
        boolean iorf = true;

        if (ArrayUtils.contains(INT_VALUE, type_value)){
            dataBuffer.asInts().read(array1);
        }else if (ArrayUtils.contains(FLOAT_VALUE, type_value)){
            dataBuffer.asFloats().read(array2);
            iorf = false;
        }

        if (iorf){
            return Nd4j.createFromArray(array1).reshape(shape);
        }else {
            return Nd4j.createFromArray(array2).reshape(shape);
        }
    }

    public static float[][][] strArraysToFloatNum(String str, int x, int y, int z) {
        char[] strc = str.toCharArray();
        int lenth = str.length();
        float[][][] result = new float[x][y][z];
        int a = 0;
        int b = 0;
        int c = 0;
        int num = 0;
        for (int i = 0; i < x * y * z; i++) {
            char[] tmp1 = new char[lenth];
            while ((strc[num] < 48 || strc[num] > 57) && strc[num] != 45 && strc[num] != 46) {
                num += 1;
            }
            int j = 0;
            while ((strc[num] >= 48 && strc[num] <= 57) || strc[num] == 45 || strc[num] == 46) {
                tmp1[j] = strc[num];
                j += 1;
                num += 1;
            }
            char[] tmp2 = new char[j];
            for (int k = 0; k < j; k++) {
                tmp2[k] = tmp1[k];
            }
            String res = String.valueOf(tmp2);
            result[a][b][c] = Float.parseFloat(res);

            c = c + 1;
            while (a >= x || b >= y || c >= z) {
                if (c >= z) {
                    c = c % z;
                    b = b + 1;
                }
                if (b >= y) {
                    b = b % y;
                    a = a + 1;
                }
                if (a >= x) {
                    a = a % x;
                }
            }
        }
        return result;
    }

    public static int[][][] strArraysToIntNum(String str, int x, int y, int z) {
        char[] strc = str.toCharArray();
        int lenth = str.length();
        int[][][] result = new int[x][y][z];
        int a = 0;
        int b = 0;
        int c = 0;
        int num = 0;
        for (int i = 0; i < x * y * z; i++) {
            char[] tmp1 = new char[lenth];
            while ((strc[num] < 48 || strc[num] > 57) && strc[num] != 45) {
                num += 1;
            }
            int j = 0;
            while ((strc[num] >= 48 && strc[num] <= 57) || strc[num] == 45) {
                tmp1[j] = strc[num];
                j += 1;
                num += 1;
            }
            char[] tmp2 = new char[j];
            for (int k = 0; k < j; k++) {
                tmp2[k] = tmp1[k];
            }
            String res = String.valueOf(tmp2);
            result[a][b][c] = Integer.parseInt(res);

            c = c + 1;
            while (a >= x || b >= y || c >= z) {
                if (c >= z) {
                    c = c % z;
                    b = b + 1;
                }
                if (b >= y) {
                    b = b % y;
                    a = a + 1;
                }
                if (a >= x) {
                    a = a % x;
                }
            }
        }
        return result;
    }

}
