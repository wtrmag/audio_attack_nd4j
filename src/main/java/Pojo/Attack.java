package Pojo;

import Lib.SparseTensor;
import Utils.CTC;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDLinalg;
import org.tensorflow.*;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ClipByValue;
import org.tensorflow.op.core.Print;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.nn.CtcLoss;
import org.tensorflow.op.random.RandomStandardNormal;
import org.tensorflow.types.TFloat32;

import java.util.*;
import java.util.stream.Collectors;

public class Attack {

    public Graph graph;

    public Ops tf;

    private String loss_fn;

    private int phrase_length;

    private int max_audio_len;

    private int learning_rate;

    private int num_iterations;

    private long batch_size;

    private boolean mp3;

    private float l2penalty = Float.MAX_VALUE;

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

    public SparseTensor decode;

    //Todo session待定
    public Attack(String loss_fn, int phrase_length, int max_audio_len, int learning_rate, int num_iterations, long batch_size, boolean mp3, float l2penalty, String restore_path) throws Exception {

        this.loss_fn = loss_fn;
        this.phrase_length = phrase_length;
        this.max_audio_len = max_audio_len;
        this.learning_rate = learning_rate;
        this.num_iterations = num_iterations;
        this.batch_size = batch_size;
        this.mp3 = mp3;
        this.l2penalty = l2penalty;
        this.restore_path = restore_path;
        this.graph = graph;
        this.tf = tf;

        long[] shape1 = Nd4j.zeros(batch_size, max_audio_len).shape();
        long[] shape2 = Nd4j.zeros(batch_size, phrase_length).shape();
        long[] shape3 = Nd4j.zeros(batch_size).shape();
        long[] shape4 = Nd4j.zeros(batch_size, 1).shape();

        this.delta = Nd4j.create(shape1);
        this.mask = Nd4j.create(shape1);
        this.cw_mask = Nd4j.create(shape2);
        this.original = Nd4j.create(shape1);
        this.lengths = Nd4j.create(shape3);
        this.importance = Nd4j.create(shape2);
        this.target_phrase = Nd4j.create(shape2);
        this.target_phrase_lengths = Nd4j.create(shape3);
        this.rescale = Nd4j.create(shape4);

        this.apply_delta = Nd4j.math.mul(Nd4j.math.clipByValue(this.delta, -2000, 2000), this.rescale);

        this.new_input = Nd4j.math.add(Nd4j.math.mul(this.apply_delta, this.mask), this.original);

        INDArray noise = Nd4j.random.normal(0.0, 2.0, DataType.FLOAT, this.new_input.shape());

        INDArray pass_in = Nd4j.math.clipByValue(Nd4j.math.add(this.new_input, noise), Math.pow(-2, 15), Math.pow(2, 15) - 1);

        //Todo get_logits()待完成
//        self.logits = logits = get_logits(pass_in, lengths)

       //Todo 重写tf.global_variables()
//        Save saver = tf.train.save();
//        saver.restore(sess, restore_path);

        this.loss_fn = loss_fn;
        INDArray ctc_loss = null;
        INDArray loss = null;
        if ("CTC".equals(loss_fn)) {
//            SparseTensor target = CTC.ctc_label_dense_to_sparse(target_phrase, target_phrase_lengths);
            //todo 待验证参数
            ctc_loss = Nd4j.loss.ctcLoss(target_phrase, this.logits, target_phrase_lengths, this.lengths);

            if (l2penalty != Float.MAX_VALUE){
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
            Adam optimzer = new Adam(graph, learning_rate);
//            optimzer.minimize(loss);

        }
//        Adam optimizer = new Adam(graph, learning_rate);
//        this.train = optimizer.minimize(loss);

//        tf.nn.ctcBeamSearchDecoder(logits, lengths, 100L, 1L);
//        Ops.create().nn.ctcBeamSearchDecoder()

    }

    public void do_attack(int[][] audio, int[] lengths, int[][] target, int[][] finetune) {


        this.original.assign(Nd4j.create(audio));
        INDArray nd = Nd4j.create(lengths).sub(1).div(320);
        this.lengths.assign(nd);

        int[][] temp1 = new int[lengths.length][this.getMax_audio_len()];
        for (int i : lengths) {
            for (int j = 0; j < this.getMax_audio_len(); j++) {
                if (j < i){
                    temp1[i][j] = 1;
                }else {
                    temp1[i][j] = 0;
                }
            }
        }
        this.mask.assign(Nd4j.create(temp1));

        int[][] temp2 = new int[lengths.length][(int) nd.length()];
        for (int i : nd.toIntVector()) {
            for (int j = 0; j < this.getPhrase_length(); j++) {
                if (j < i){
                    temp1[i][j] = 1;
                }else {
                    temp1[i][j] = 0;
                }
            }
        }
        this.cw_mask.assign(Nd4j.create(temp2));

        this.target_phrase_lengths.assign(Nd4j.create(Arrays.stream(target).map(x -> {
            return x.length;
        }).collect(Collectors.toList())));

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
        this.target_phrase.assign(Nd4j.create(ls));

        this.importance.assign(Nd4j.ones(this.getBatch_size(), this.getPhrase_length()));
        this.rescale.assign(Nd4j.ones(this.getBatch_size(), 1));


        int[] final_deltas = new int[((int) this.batch_size)];

        //Todo finetune
//        if finetune is not None and len(finetune) > 0:
//        sess.run(self.delta.assign(finetune - audio))

        Date date = new Date();
        for (int i = 0; i < this.getNum_iterations(); i++){
            long now = date.getTime();

            //todo print信息
//            if (i % 10 == 0) {
//                Map<INDArray, INDArray> lst = new HashMap<>();
//                lst.put(this.decode, this.logits);
//                if(this.mp3){
//                    this.do_something();//Todo
//                }
//                for (Map.Entry<INDArray, INDArray> entry : lst.entrySet()){
//                    //Todo
//                }
//            }

            if (this.mp3){
                this.do_something();//Todo
            }else {

            }

            //todo
//            print("%.3f" % np.mean(cl), "\t", "\t".join("%.3f" % x for x in cl))

            INDArray logits = Nd4j.argMax(this.logits, 2).transpose();
            for (int j = 0; j < this.getBatch_size(); j++) {
//                if (StringUtils.equals("CTC", this.loss_fn) && i % 10 == 0 &&)
            }
        }

    }

    public void do_something(){

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

    public String getRestore_path() {
        return restore_path;
    }
}
