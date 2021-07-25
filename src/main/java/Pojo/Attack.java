package Pojo;

import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.ClipByValue;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.random.RandomStandardNormal;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.family.TNumber;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.HashMap;

public class Attack {

    private String loss_fn;

    private int phrase_length;

    private int max_audio_len;

    private int learning_rate;

    private int num_iterations;

    private int batch_size;

    private boolean mp3;

    //默认值为正无穷
    private float l2penalty;

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

    //Todo unsure
    public Operation train;

    //Todo unsure
    public int[] decode;

    //Todo session待定
    public Attack(String loss_fn, int phrase_length, int max_audio_len, int learning_rate, int num_iterations, int batch_size, boolean mp3, float l2penalty, String restore_path) {

        this.loss_fn = loss_fn;
        this.phrase_length = phrase_length;
        this.max_audio_len = max_audio_len;
        this.learning_rate = learning_rate;
        this.num_iterations = num_iterations;
        this.batch_size = batch_size;
        this.mp3 = mp3;
        this.l2penalty = l2penalty;
        this.restore_path = restore_path;

        Graph graph = new Graph();
        Ops tf = Ops.create(graph);
        Scope scope = new Scope(graph);

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

        Operand<TFloat32> delta = tf.variable(shape1, TFloat32.class, hashMap.get("qq_delta"));
        Operand<TFloat32> mask = tf.variable(shape1, TFloat32.class, hashMap.get("qq_mask"));
        Operand<TFloat32> cwmask = tf.variable(shape2, TFloat32.class, hashMap.get("qq_cwmask"));
        Operand<TFloat32> original = tf.variable(shape1, TFloat32.class, hashMap.get("qq_original"));
        Operand<TFloat32> lengths = tf.variable(shape3, TFloat32.class, hashMap.get("qq_lengths"));
        Operand<TFloat32> importance = tf.variable(shape2, TFloat32.class, hashMap.get("qq_importance"));
        Operand<TFloat32> target_phrase = tf.variable(shape2, TFloat32.class, hashMap.get("qq_phrase"));
        Operand<TFloat32> target_phrasew_lengths = tf.variable(shape3, TFloat32.class, hashMap.get("qq_phrase_lengths"));
        Operand<TFloat32> rescale = tf.variable(shape4, TFloat32.class, hashMap.get("qq_phrase_lengths"));

        float min = -2000;
        float max = 2000;
        Mul apply_data = tf.math.mul(tf.clipByValue(delta, tf.constant((float) -2000), tf.constant((float) 2000)), rescale);
        Add new_input = tf.math.add(tf.math.mul(apply_data, mask), original);

        //Todo 待修改
        RandomStandardNormal noise = rand_normal(scope, tf.shape.size(tf.shape(new_input), TFloat32.class), tf.constant((float) 2.0), TFloat32.class);
        ClipByValue pass_in = tf.clipByValue(tf.math.add(new_input, noise), tf.constant((float) Math.pow(-2, 15)), tf.constant((float) Math.pow(22, 15) - 1));

        //Todo get_logits()待完成
//        self.logits = logits = get_logits(pass_in, lengths)

       //Todo 重写tf.global_variables()
//        Save saver = tf.train.save();
//        saver.restore(sess, restore_path);

        //Todo 重写ctc_label_dense_to_sparse
        if ("CTC".equals(loss_fn)) {

        }






//        super(delta, mask, cw_mask, original, lengths, importance, target_phrase, target_phrase_lengths, rescale, apply_delta, new_input, logits, loss, ctc_loss, train, decode);
    }

    public<U extends TNumber, T extends TNumber> RandomStandardNormal<U> rand_normal(Scope scope, Operand<T> shape, Operand<T> stddev, Class<U> dtype){
        OperationBuilder opBuilder = scope.env().opBuilder("RandomStandardNormal", scope.makeOpName("RandomStandardNormal"));
        opBuilder.addInput(shape.asOutput());
        opBuilder.addInput(stddev.asOutput());
        opBuilder = scope.applyControlDependencies(opBuilder);
//        opBuilder.setAttr("dtype", dtype);

        RandomStandardNormal normal = null;
        try {
            Class cls =  Class.forName("org.tensorflow.op.random.RandomStandardNormal");
            Constructor<RandomStandardNormal> Constructor = cls.getDeclaredConstructor(Operation.class);
            normal = Constructor.newInstance(opBuilder.build());
        } catch (ClassNotFoundException exception){
            System.err.println("找不到所在的包");
        } catch (NoSuchMethodException exception){
            System.err.println("无此方法");
        } catch (Exception exception){
            System.err.println("生成对象出错");
        }
        return normal;
    }

}
