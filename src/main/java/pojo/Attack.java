package pojo;

import utils.CTC;
import utils.DataConvert;
import utils.Tf_logits;
import utils.waveaccess.WaveFileWriter;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.tensorflow.EagerSession;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.op.Ops;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Attack {

    private String loss_fn;

    private int phrase_length;

    private int max_audio_len;

    private int learning_rate;

    private int num_iterations;

    private long batch_size;

    private boolean mp3;

    private float l2penalty = Float.POSITIVE_INFINITY;

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

    public Operand train;

    public SparseTensor decode;

    public Attack(String loss_fn, int phrase_length, int max_audio_len, int learning_rate, int num_iterations, long batch_size, boolean mp3, float l2penalty) {

        this.loss_fn = loss_fn;
        this.phrase_length = phrase_length;
        this.max_audio_len = max_audio_len;
        this.learning_rate = learning_rate;
        this.num_iterations = num_iterations;
        this.batch_size = batch_size;
        this.mp3 = mp3;
        this.l2penalty = l2penalty;

//        long[] shape1 = Nd4j.zeros(batch_size, max_audio_len).shape();
//        long[] shape2 = Nd4j.zeros(batch_size, phrase_length).shape();
//        long[] shape3 = Nd4j.zeros(batch_size).shape();
//        long[] shape4 = Nd4j.zeros(batch_size, 1).shape();
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

        this.loss_fn = loss_fn;
        INDArray ctc_loss = null;
        INDArray loss = null;
        if ("CTC".equals(loss_fn)) {
            SparseTensor labels = CTC.ctc_label_dense_to_sparse(target_phrase, target_phrase_lengths);

            ctc_loss = Nd4j.loss.ctcLoss(labels.getValues().dup(), this.logits.dup(), labels.getIndices().dup(), this.lengths.dup());

            if (l2penalty != Float.POSITIVE_INFINITY){
                loss = Nd4j.math().add(Nd4j.mean(Nd4j.math().pow(Nd4j.math().sub(this.new_input, this.original),
                        2), 0), Nd4j.math().mul(ctc_loss, this.l2penalty));
            } else {
                loss = ctc_loss;
            }
            this.expanded_loss = Nd4j.arange(0);
        }else {
            throw new Exception("unfinished");
        }

        this.ctc_loss = ctc_loss;
        this.loss = loss;

        try(EagerSession session = EagerSession.create();
            Graph graph = new Graph()) {
            Ops tf =  Ops.create(session);
//            tf.nn.ctcLoss()
            Adam optimzer = new Adam(graph, learning_rate);
            Operand l = DataConvert.nd2tf(tf, this.loss.dup());

            this.train = l;
            List<Optimizer.GradAndVar<?>> list = optimzer.computeGradients(l);
//            optimzer.applyGradients(, "optimzer");

//            tf.nn.ctcBeamSearchDecoder()
        }

        double[][] final_deltas = new double[((int) this.batch_size)][];

        //Todo finetune
//        if finetune is not None and len(finetune) > 0:
//        sess.run(self.delta.assign(finetune - audio))

        Date date = new Date();
        for (int i = 0; i < this.getNum_iterations(); i++){
            long now = date.getTime();

            INDArray res = Nd4j.zeros(this.decode.getDenseShape().shape());
            //todo print信息
            if (i % 10 == 0) {
                Map<SparseTensor, INDArray> lst = new HashMap<>();
                lst.put(this.decode, this.logits);
//                if(this.mp3){
//                    this.do_something();//Todo
//                }
                for (Map.Entry<SparseTensor, INDArray> entry : lst.entrySet()){
                    SparseTensor s = entry.getKey();
                    res = res.add(Variables.TOKENS.length() - 1);
                    for (int j = 0; j < s.getValues().length(); j++) {
                        int x = s.getIndices().get(NDArrayIndex.point(j), NDArrayIndex.point(0)).getInt(0);
                        int y = s.getIndices().get(NDArrayIndex.point(j), NDArrayIndex.point(1)).getInt(0);

                        res.put(x, y, s.getValues().get(NDArrayIndex.point(j)));
                        //todo print
                    }
                }
            }

//            if (this.mp3){
//                this.do_something();//Todo
//            }else {
//
//            }

            //Todo
//            print("%.3f" % np.mean(cl), "\t", "\t".join("%.3f" % x for x in cl))

            INDArray logits = Nd4j.argMax(this.logits.dup(), 2).transpose();
            for (int j = 0; j < this.getBatch_size(); j++) {
                StringBuilder builder = new StringBuilder();
                for (int x: target[j]){
                    char c = Variables.TOKENS.charAt(x);
                    builder.append(c);
                }
                boolean b = (StringUtils.equals("CTC", this.loss_fn) && i % 10 == 0 && StringUtils.equals(res.getString(j), String.join("", builder)))
                        || (i == this.getNum_iterations());
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
                    WaveFileWriter writer = new WaveFileWriter(Variables.TEMP+"adv.wav", result.toIntMatrix(), 16000);
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
