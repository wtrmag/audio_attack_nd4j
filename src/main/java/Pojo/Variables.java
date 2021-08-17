package Pojo;

import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.framework.*;
import org.tensorflow.op.Ops;

import java.util.ArrayList;


/**
 * @author wtr
 */
public class Variables {

    public static final String TOKENS = " abcdefghijklmnopqrstuvwxyz'-";

    public static final String RESULTS = "src/main/resources/results";

    public static final String TEMP = "sec/main/resources/temp";

    public static final String FFMPEG_PATH = "D:\\ffmpeg-n4.4-18-gc813f5e343-win64-gpl-4.4\\bin\\ffmpeg.exe";
}
