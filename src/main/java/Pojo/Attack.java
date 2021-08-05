package Pojo;

import Lib.SparseTensor;
import Utils.CTC;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.*;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ClipByValue;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.nn.CtcLoss;
import org.tensorflow.op.random.RandomStandardNormal;
import org.tensorflow.types.TFloat32;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;

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

    public Operand delta;

    public Operand mask;

    public Operand cw_mask;

    public Operand original;

    public Operand lengths;

    public Operand importance;

    public Operand target_phrase;

    public Operand target_phrase_lengths;

    public Operand rescale;

    public Operand apply_delta;

    public Operand new_input;

    public Operand logits;

    public Operand loss;

    public Operand ctc_loss;

    public Operand expanded_loss;

    public Session session;

    public Op train;

    //Todo unsure
    public int[] decode;

    //Todo session待定
    public Attack(Graph graph, Ops tf, Session session, String loss_fn, int phrase_length, int max_audio_len, int learning_rate, int num_iterations, long batch_size, boolean mp3, float l2penalty, String restore_path) {

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

        // 通过反射获取shape构造函数
        Shape shape1 = null;
        Shape shape2 = null;
        Shape shape3 = null;
        Shape shape4 = null;
        Variable.Options options = null;
        HashMap<String, Variable.Options> hashMap = new HashMap<>();
        try {
            Class shape_class =  Class.forName("org.tensorflow.ndarray.Shape");
            Constructor<Shape> ShapeConstructor = shape_class.getDeclaredConstructor(long[].class);
            shape1 = ShapeConstructor.newInstance(Nd4j.zeros(batch_size, max_audio_len).shape());
            shape2 = ShapeConstructor.newInstance(Nd4j.zeros(batch_size, phrase_length).shape());
            shape3 = ShapeConstructor.newInstance(Nd4j.zeros(batch_size).shape());
            shape4 = ShapeConstructor.newInstance(Nd4j.zeros(batch_size, 1).shape());

            Class options_class = Class.forName("org.tensorflow.op.core.Variable$Options");
            Constructor<Variable.Options> optionsConstructor = options_class.getDeclaredConstructor();

            for (options_sharedname opt: options_sharedname.values()){
                options = optionsConstructor.newInstance();
                Field sharedName = options_class.getDeclaredField("sharedName");
                sharedName.setAccessible(true);
                sharedName.set(options, opt.getName());
                hashMap.put(opt.getName(), options);
            }
        } catch (ClassNotFoundException exception){
            System.err.println("找不到所在的包");
        } catch (NoSuchMethodException exception){
            System.err.println("无此方法");
        } catch (Exception exception){
            System.err.println("生成对象出错");
        }

        Operand delta = tf.variable(shape1, TFloat32.class, hashMap.get("qq_delta"));
        Operand mask = tf.variable(shape1, TFloat32.class, hashMap.get("qq_mask"));
        Operand cw_mask = tf.variable(shape2, TFloat32.class, hashMap.get("qq_cwmask"));
        Operand original = tf.variable(shape1, TFloat32.class, hashMap.get("qq_original"));
        Operand lengths = tf.variable(shape3, TFloat32.class, hashMap.get("qq_lengths"));
        Operand importance = tf.variable(shape2, TFloat32.class, hashMap.get("qq_importance"));
        Operand target_phrase = tf.variable(shape2, TFloat32.class, hashMap.get("qq_phrase"));
        Operand target_phrase_lengths = tf.variable(shape3, TFloat32.class, hashMap.get("qq_phrase_lengths"));
        Operand rescale = tf.variable(shape4, TFloat32.class, hashMap.get("qq_phrase_lengths"));

        ArrayList list = Variables.global_variables(delta, mask, cw_mask, original, lengths, importance, target_phrase, target_phrase_lengths, rescale);

        Mul apply_data = tf.math.mul(tf.clipByValue(delta, tf.constant((float) -2000), tf.constant((float) 2000)), rescale);
        Add new_input = tf.math.add(tf.math.mul(apply_data, mask), original);

        //RandomNormal noise = new RandomNormal(tf, RandomNormal.MEAN_DEFAULT, 2.0D, tf.shape(new_input).shape().size());
        RandomStandardNormal random = tf.random.randomStandardNormal(tf.shape(new_input), TFloat32.class);
        Mul noise = tf.math.mul(random, tf.constant(2));

        ClipByValue pass_in = tf.clipByValue(tf.math.add(new_input, noise), tf.constant((float) Math.pow(-2, 15)), tf.constant((float) Math.pow(22, 15) - 1));

        //Todo get_logits()待完成
//        self.logits = logits = get_logits(pass_in, lengths)

       //Todo 重写tf.global_variables()
//        Save saver = tf.train.save();
//        saver.restore(sess, restore_path);

        CtcLoss ctc_loss = null;
        Operand loss = null;
        this.loss_fn = loss_fn;
        if ("CTC".equals(loss_fn)) {
            SparseTensor target = CTC.ctc_label_dense_to_sparse(tf, target_phrase, target_phrase_lengths);

            ctc_loss = tf.nn.ctcLoss(logits, target.getIndices(), target.getValues(), lengths);

            if (l2penalty != Float.MAX_VALUE){
                loss = tf.math.add(tf.math.mean(tf.math.pow(tf.math.sub(new_input, original), tf.constant(2)), tf.constant(1)),
                        tf.math.mul(tf.constant(l2penalty), ctc_loss.loss()));
            } else {
                loss = ctc_loss.loss();
            }
            this.expanded_loss = tf.constant(0);
        }

        this.ctc_loss = ctc_loss.loss();
        this.loss = loss;

        Adam optimizer = new Adam(graph, learning_rate);
        this.train = optimizer.minimize(loss);
//        this(delta, mask, cw_mask, original, lengths, importance, target_phrase, target_phrase_lengths, rescale);


        tf.nn.ctcBeamSearchDecoder(logits, lengths, 100L, 1L);
    }

    public Attack(Operand delta, Operand mask, Operand cw_mask, Operand original, Operand lengths, Operand importance, Operand target_phrase, Operand target_phrase_lengths, Operand rescale) {
        this.delta = delta;
        this.mask = mask;
        this.cw_mask = cw_mask;
        this.original = original;
        this.lengths = lengths;
        this.importance = importance;
        this.target_phrase = target_phrase;
        this.target_phrase_lengths = target_phrase_lengths;
        this.rescale = rescale;
    }

    public void do_attack(int[][] audio, int[][] finetune) {
        Session session = this.session;
        Ops tf = this.tf;

        //Todo
//        session.run(tf.varIsInitializedOp(this.delta));
//        session.run(tf.assignVariableOp(this.original, );
//        session.run();
//        session.run(tf.assignVariableOp(attack.lengths, (Operand<? extends TType>) NdArrays.ofLongs(audio.shape())));
//        session.run(tf.assignVariableOp(attack.mask, NdArrays.));
//        session.run(tf.assignVariableOp(attack.cw_mask, ));
//        session.run(tf.assignVariableOp(attack.target_phrase, ));
//        session.run(tf.assignVariableOp(attack.target_phrase_lengths, ));

        int[] final_deltas = new int[((int) this.batch_size)];

        //Todo finetune
//        if ()

        Date date = new Date();
        for (int i = 0; i < this.getNum_iterations(); i++){
            long now = date.getTime();

            if (i % 10 == 0) {
                //Todo
//                new, delta, r_out, r_logits = sess.run((self.new_input, self.delta, self.decoded, self.logits))
//                lst = [(r_out, r_logits)]
            }
        }

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
