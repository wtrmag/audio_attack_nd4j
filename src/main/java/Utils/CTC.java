package Utils;

import Lib.SparseTensor;
import Pojo.Variables;
import org.nd4j.common.io.CollectionUtils;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.op.Ops;
import org.tensorflow.op.RawOp;
import org.tensorflow.op.core.*;
import org.tensorflow.op.dtypes.Cast;
import org.tensorflow.op.linalg.Transpose;
import org.tensorflow.op.math.Less;
import org.tensorflow.op.sparse.SparseSplit;
import org.tensorflow.op.sparse.SparseTensorDenseAdd;
import org.tensorflow.types.TFloat32;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * @author wtr
 */
public class CTC extends RawOp {

    /**
     * Mike Henry's implementation, with some minor modifications.
     * @param labels
     * @param label_lengths
     * @return
     */
    public static SparseTensor ctc_label_dense_to_sparse(Ops tf, Operand labels, Operand label_lengths) {

        Shape label_shape = tf.shape(labels);
        float f0 = (float) label_shape.shape().asArray()[1];
        float f1 = (float) label_shape.shape().asArray()[1];
        Iterable i1 = Collections.singletonList(label_shape.shape().asArray()[0]);
        Iterable i2 = Collections.singletonList(label_shape.shape().asArray()[1]);
        Stack num_batch_tns = tf.stack(i1);
        Stack max_num_labels_tns = tf.stack(i2);

        Cast init = tf.dtypes.cast(tf.fill(max_num_labels_tns, tf.constant(1)), TFloat32.class);
        ExpandDims init_expanded = tf.expandDims(init, tf.constant(0));

        Range range = tf.range(tf.constant((float) 0),tf.constant(f1), tf.constant((float) 1));
        ExpandDims temp = tf.expandDims(range, tf.constant(0));

        Less dense_mask = tf.math.less(temp, label_lengths);
        long[] longs = tf.shape(dense_mask).shape().asArray();
        long[] begin = new long[3];
        long len1 = tf.shape(dense_mask).shape().asArray()[0];
        long len3 = tf.shape(dense_mask).shape().asArray()[2];
        long[] size = {len1 ,1 ,len3};
        Slice dense_mask_slice = tf.slice(dense_mask, tf.constant(begin), tf.constant(size));

        Reshape label_array = tf.reshape(tf.tile(tf.range(tf.constant((float) 0), tf.constant(f1), tf.constant((float) 1)), num_batch_tns), label_shape);
        Operand label_ind = tf.booleanMask(label_array, dense_mask_slice);

        Transpose batch_array = tf.linalg.transpose(tf.reshape(tf.tile(tf.range(tf.constant((float) 0), tf.constant(f0), tf.constant((float) 1)), max_num_labels_tns), tf.reverse(label_shape, tf.constant((float) 0))), tf.constant(Variables.perm));
        Operand batch_ind = tf.booleanMask(batch_array, dense_mask_slice);

        ArrayList list = new ArrayList();
        list.add(batch_array);
        list.add(label_ind);
        Transpose indices = tf.linalg.transpose(tf.reshape(tf.concat(list, tf.constant((float) 0)), tf.constant(new float[] {2, -1})), tf.constant(Variables.perm));
        GatherNd vals_sparse = tf.gatherNd(labels, indices);

        return new SparseTensor(indices, vals_sparse, label_shape);
    }

    /**
     * Constructor.
     *
     * @param operation the underlying operation
     */
    protected CTC(Operation operation) {
        super(operation);
    }
}
