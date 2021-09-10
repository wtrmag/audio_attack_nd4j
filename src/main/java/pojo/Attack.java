package pojo;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.AdamUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.nn.CtcBeamSearchDecoder;
import org.tensorflow.op.nn.CtcLoss;
import org.tensorflow.op.train.ApplyAdam;
import org.tensorflow.op.train.Restore;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.family.TType;
import utils.*;
import utils.waveaccess.WaveFileWriter;

import java.lang.reflect.Method;
import java.nio.charset.Charset;
import java.util.*;
import java.util.stream.Collectors;

public class Attack {

    private String loss_fn;

    private int phrase_length;

    private int max_audio_len;

    private int learning_rate;

    private int num_iterations;

    private long batch_size;

    private boolean mp3;

    private float l2penalty = Float.POSITIVE_INFINITY;

    private String restore_path;

    public INDArray delta;

    public INDArray mask;

    public INDArray cw_mask;

    public INDArray original;

    public INDArray lengths;

    public INDArray importance;

    public INDArray target_phrase;

    public INDArray target_phrase_lengths;

    public INDArray rescale;

    public INDArray apply_delta;

    public INDArray new_input;

    public INDArray logits;

    public INDArray loss;

    public INDArray ctc_loss;

    public INDArray expanded_loss;

    public ApplyAdam train;

    public INDArray grad;

    public INDArray var;

    public SparseTensor decoded;

    public Attack(String loss_fn, int phrase_length, int max_audio_len, int learning_rate, int num_iterations, long batch_size, boolean mp3, float l2penalty, String restore_path) {

        this.loss_fn = loss_fn;
        this.phrase_length = phrase_length;
        this.max_audio_len = max_audio_len;
        this.learning_rate = learning_rate;
        this.num_iterations = num_iterations;
        this.batch_size = batch_size;
        this.mp3 = mp3;
        this.l2penalty = l2penalty;
        this.restore_path = restore_path;

    }

    public double[][] do_attack(int[][] audio, int[] length, int[][] target, int[][] finetune) throws Exception {
        this.original = Nd4j.create(audio);
        INDArray nd = Nd4j.createFromArray(length).sub(1).div(320);
        this.lengths = nd.dup();
        int[][] temp1 = new int[length.length][this.getMax_audio_len()];
        for (int i = 0; i < length.length; i++) {
            for (int j = 0; j < this.getMax_audio_len(); j++) {
                if (j < length[i]){
                    temp1[i][j] = 1;
                }else {
                    temp1[i][j] = 0;
                }
            }
        }
        this.mask = Nd4j.create(temp1);

        int[][] temp2 = new int[length.length][(int) nd.length()];
        for (int i = 0; i < length.length; i++) {
            for (int j = 0; j < this.getPhrase_length(); j++) {
                if (j < nd.toIntVector()[i]){
                    temp1[i][j] = 1;
                }else {
                    temp1[i][j] = 0;
                }
            }
        }
        this.cw_mask = Nd4j.create(temp2);

        this.target_phrase_lengths = Nd4j.create(Arrays.stream(target).map(x -> x.length).collect(Collectors.toList()));

        int[][] temp3 = new int[target.length][this.phrase_length];
        List ls = Arrays.stream(target).map(x -> {
            int[] t = new int[this.getPhrase_length()];
            for (int i = 0; i < t.length; i++) {
                if (i < x.length){
                    t[i] = x[i];
                }else {
                    t[i] = 0;
                }
            }
            return t;
        }).collect(Collectors.toList());
        int k = 0;
        for (Object l: ls){
            temp3[k++] = (int[]) l;
        }
        this.target_phrase = Nd4j.create(temp3);

        this.importance = Nd4j.ones(this.getBatch_size(), this.getPhrase_length());
        this.rescale = Nd4j.ones(this.getBatch_size(), 1);

        this.delta = Nd4j.zeros(this.batch_size, this.max_audio_len);
        boolean t = finetune.length != 0 && (finetune.length != 1 || finetune[0].length !=0);
        if (t){
            INDArray tune = Nd4j.create(finetune);
            this.delta.assign(Nd4j.math.div(tune, this.original.dup()));
        }
        this.apply_delta = Nd4j.math.mul(Nd4j.math.clipByValue(this.delta.dup(), -2000, 2000), this.rescale);

        this.new_input = Nd4j.math.add(Nd4j.math.mul(this.apply_delta, this.mask.dup()), this.original.dup());

        INDArray noise = Nd4j.random.normal(0.0, 2.0, DataType.FLOAT, this.new_input.shape());

        INDArray pass_in = Nd4j.math.clipByValue(Nd4j.math.add(this.new_input.dup(), noise), Math.pow(-2, 15), Math.pow(2, 15) - 1);

        this.logits  = new Tf_logits().get_logits(pass_in, this.lengths);

        if ("CTC".equals(loss_fn)) {
            List list = PyCall.UpdateDelta(this.target_phrase, this.target_phrase_lengths.castTo(DataType.INT32), this.learning_rate, length[0]);
            this.delta =  list.get(0)!= null ? (INDArray) list.get(0) : this.delta;
            this.ctc_loss = Nd4j.createFromArray((double) list.get(1));

            if (l2penalty != Float.POSITIVE_INFINITY){
                loss = Nd4j.math().add(Nd4j.mean(Nd4j.math().pow(Nd4j.math().sub(this.new_input, this.original),
                        2), 0), Nd4j.math().mul(ctc_loss, this.l2penalty));
            } else {
                loss = this.ctc_loss;
            }
            this.expanded_loss = Nd4j.createFromArray(0);
        }else {
            throw new Exception("unfinished");
        }
        this.loss = loss;

        CtcBeamSearchDecoder decode;
        try(Graph graph = new Graph()) {
            Ops tf =  Ops.create(graph);
            Operand inputs = DataConvert.nd2tf(tf, this.logits.dup());
            Operand sequenceLength = DataConvert.nd2tf(tf, this.lengths.dup());

            decode = tf.nn.ctcBeamSearchDecoder(inputs, sequenceLength, 100L, 1L);
            Output va = (Output) decode.decodedValues().get(0);
            Output i = (Output) decode.decodedIndices().get(0);
            Output s = (Output) decode.decodedShape().get(0);

            try(Session session = new Session(graph)) {
                session.run(tf.init());
                int[][]  t_i = DataConvert.tf2nd(session, tf, i).toIntMatrix();
                int[] t_v = DataConvert.tf2nd(session, tf, va).toIntVector();
                int remove_indice_len;
                if (t_i.length % 2 == 0){
                    remove_indice_len = t_i.length / 2;
                }else {
                    remove_indice_len = t_i.length / 2 + 1;
                }
                Object[] t_in = Arrays.stream(t_i).sorted(Comparator.comparing(ints -> ints[0])).skip(remove_indice_len).toArray();
                int[][] indices = new int[t_in.length][2];
                for (int j = 0; j < t_in.length; j++) {
                    indices[j] = (int[]) t_in[j];
                }
                int[] values = new int[t_v.length - remove_indice_len];
                for (int j = 0; j < values.length; j++) {
                    values[j] = t_v[j*2];
                }
                int[] shape = new int[]{1, t_i.length/2};
                this.decoded = new SparseTensor(Nd4j.createFromArray(indices), Nd4j.createFromArray(values), Nd4j.createFromArray(shape));
            }

        }

        double[][] final_deltas = new double[((int) this.batch_size)][];

        //Todo finetune

        Date date = new Date();
        for (int i = 0; i < this.getNum_iterations(); i++){
            long now = date.getTime();

            INDArray res = Nd4j.zeros(this.decoded.getDenseShape().toIntVector());
            //todo print信息
            if (i % 10 == 0) {
                Map<SparseTensor, INDArray> lst = new HashMap<>();
                lst.put(this.decoded, this.logits);

                //Todo this.mp3
                for (Map.Entry<SparseTensor, INDArray> entry : lst.entrySet()){
                    SparseTensor s = entry.getKey();
                    res = res.add(Variables.TOKENS.length() - 1);
                    for (int j = 0; j < s.getValues().length(); j++) {
                        int y = s.getIndices().get(NDArrayIndex.point(j), NDArrayIndex.point(0)).getInt(0);
                        int x = s.getIndices().get(NDArrayIndex.point(j), NDArrayIndex.point(1)).getInt(0);

                        res.put(x, y, s.getValues().get(NDArrayIndex.point(j)));
                        //todo print
                    }
                }
            }
            //Todo this.mp3
            //todo print

            StringBuilder str_res = new StringBuilder();
            int[][] t_res = res.toIntMatrix();
            for (int j = 0; j < t_res.length; j++) {
                for (int r : t_res[j]) {
                    str_res.append(Variables.TOKENS.charAt(r));
                }
            }
            str_res = new StringBuilder(String.join("", str_res).replace("-", ""));
            INDArray logits = Nd4j.argMax(this.logits.dup(), 2).transpose();
            for (int j = 0; j < this.getBatch_size(); j++) {
                StringBuilder builder = new StringBuilder();
                for (int x: target[j]){
                    char c = Variables.TOKENS.charAt(x);
                    builder.append(c);
                }
                boolean b = (StringUtils.equals("CTC", this.loss_fn) && i % 10 == 0 && StringUtils.equals(str_res.toString(), String.join("", builder)))
                        || (i == this.getNum_iterations() - 1);
                if (b) {
                    INDArray rescale = this.rescale.dup();
                    if (rescale.getDouble(j) * 2000 > this.delta.dup().amaxNumber().doubleValue()){
                        double v = this.delta.dup().get(NDArrayIndex.point(j)).amaxNumber().doubleValue() / 2000.0;
                        System.out.println("It's way over"+v);
                        rescale.put(j, Nd4j.create(new double[]{v}));
                    }
                    rescale.put(j, Nd4j.create(new double[]{rescale.getDouble(j) * 0.8}));

                    final_deltas[j] = this.new_input.toDoubleMatrix()[j];

                    INDArray round = Nd4j.math.round(this.new_input.dup().get(NDArrayIndex.point(j)));
                    INDArray result = Nd4j.math.clipByValue(round, Math.pow(-2, 15), Math.pow(2, 15)-1);
                    WaveFileWriter writer = new WaveFileWriter(Variables.TEMP+"adv.wav", Nd4j.expandDims(result, 0).toIntMatrix(), 16000);
                    writer.close();
                }
            }
        }

        return final_deltas;
    }

    public String getLoss_fn() {
        return loss_fn;
    }

    public int getPhrase_length() {
        return phrase_length;
    }

    public int getMax_audio_len() {
        return max_audio_len;
    }

    public int getLearning_rate() {
        return learning_rate;
    }

    public int getNum_iterations() {
        return num_iterations;
    }

    public long getBatch_size() {
        return batch_size;
    }

    public boolean isMp3() {
        return mp3;
    }

    public float getL2penalty() {
        return l2penalty;
    }

}
