package Pojo;

import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Graph;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;


public class Attack extends Variables {
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

    //Todo session待定
    public Attack(String loss_fn, int phrase_length, int max_audio_len, int learning_rate, int num_iterations, int batch_size, boolean mp3, float l2penalty, String restore_path) {
        super();

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
        Scope scope = new Scope(graph);

        // 通过反射获取shape构造函数
        Shape shape = null;
        Variable.Options options = null;
        try {
            Class shape_class =  Class.forName("org.tensorflow.ndarray.Shape");
            Constructor<Shape> ShapeConstructor = shape_class.getDeclaredConstructor(long[].class);
            shape = ShapeConstructor.newInstance(Nd4j.zeros(batch_size, max_audio_len).shape());

            Class options_class = Class.forName("org.tensorflow.op.core.Variable$Options");
            Constructor<Variable.Options> optionsConstructor = options_class.getDeclaredConstructor();
            options = optionsConstructor.newInstance();

            Field sharedName = options_class.getDeclaredField("sharedName");
            sharedName.set(options, "qq_delta");

        } catch (ClassNotFoundException exception){
            System.err.println("找不到Shape所在的包");
        } catch (NoSuchMethodException exception){
            System.err.println("Shape中无此方法");
        } catch (Exception exception){
            System.err.println("生成Shape对象出错");
        }


        String sharedName = "";
        Variable<TFloat32> delta = Variable.create(scope, shape, TFloat32.DTYPE, options);


//        super(delta, mask, cw_mask, original, lengths, importance, target_phrase, target_phrase_lengths, rescale, apply_delta, new_input, logits, loss, ctc_loss, train, decode);
    }

}
