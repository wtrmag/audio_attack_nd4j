package utils;

import pojo.SparseTensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDBase;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

/**
 * @author wtr
 */
public class CTC {

    /**
     *
     * @param labels
     * @param label_lengths
     * @return
     */
    public static SparseTensor ctc_label_dense_to_sparse(INDArray labels, INDArray label_lengths) {

        long[] s = labels.shape();
        INDArray shape = Nd4j.createFromArray(s);
        NDBase base_ops = new NDBase();

        INDArray num_batches_tns = Nd4j.stack(0, shape.getScalar(0));
        INDArray max_num_labels_tns = Nd4j.stack(0, shape.getScalar(1));

        INDArray init = Nd4j.zeros(1, s[1]).castTo(DataType.BOOL);
        BiFunction<INDArray, INDArray, INDArray> range_less_than = (old_input, current_input) -> {
            INDArray old = Nd4j.expandDims(Nd4j.arange(old_input.shape()[1]), 0);
            return old.lt(current_input);
        };
        INDArray dense_mask = scan4j(range_less_than, label_lengths, init);
//        INDArray dense_mask = r.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all());

        INDArray label_array = base_ops.reshape(base_ops.tile(Nd4j.arange(0, s[1]).castTo(DataType.INT64)
                , num_batches_tns), s);
        AtomicInteger index1 = new AtomicInteger(0);
        long[] array1 = Arrays.stream(label_array.toLongVector()).filter(o ->
                dense_mask.toIntVector()[index1.getAndIncrement()] == 1).toArray();
        INDArray label_ind = Nd4j.createFromArray(array1);

        INDArray t = base_ops.reshape(base_ops.tile(Nd4j.arange(0, s[0]).castTo(DataType.INT64), max_num_labels_tns),
                base_ops.reverse(shape, 0));
        INDArray batch_array = base_ops.transpose(t);
        AtomicInteger index2 = new AtomicInteger(0);
        long[] array2 = Arrays.stream(label_array.toLongVector()).filter(o ->
                dense_mask.toIntVector()[index2.getAndIncrement()] == 1).toArray();
        INDArray batch_ind = Nd4j.createFromArray(array2);

        INDArray indices = base_ops.transpose(base_ops.reshape(Nd4j.concat(0, batch_ind, label_ind), 2, -1));
        INDArray vals_sparse =  base_ops.gatherNd(labels, indices);
        return new SparseTensor(indices, vals_sparse, shape);
    }

    public static INDArray scan4j(BiFunction<INDArray, INDArray, INDArray> range_less_than, INDArray label_lengths, INDArray init) {
        long temp = label_lengths.shape()[0] - 1;
        return func(range_less_than, init, label_lengths, temp);
    }

    public static INDArray func(BiFunction<INDArray, INDArray, INDArray> range_less_than, INDArray a, INDArray b, long i){
        if(i == 0){
            return range_less_than.apply(a, b.getScalar(i));
        }else {
            return range_less_than.apply(func(range_less_than, a, b, i - 1), b.getScalar(i));
        }
    }

    public static List beam_search_decoder(INDArray z, int beam_width){
        long[] shape = z.shape();
        INDArray log_z = Nd4j.math.log(Nd4j.math.abs(z.dup()));

        ArrayList list = new ArrayList();
        ArrayList<Data> beam = new ArrayList();
        beam.add(new Data(list, Nd4j.zeros(1)));
        List u = null;
        for (int i = 0; i < shape[0]; i++) {
            ArrayList new_beam = new ArrayList<>();

//            System.out.println(0);
            for (Data temp : beam){
//                System.out.println(1);
                for (int j = 0; j < shape[1]; j++) {
//                    System.out.println(j);
                    list.add(j);
                    temp.setArray(list);
                    temp.setNum(log_z.dup().get(NDArrayIndex.point(i), NDArrayIndex.point(j)));
                    temp = new Data(list, log_z.dup().get(NDArrayIndex.point(i), NDArrayIndex.point(j)));
                    new_beam.add(temp);
                }
            }
            u = (List) new_beam.stream().sorted(Comparator.comparing(Data::getNum, (x , y)->{
                if (x.dup().toIntVector()[0] > y.dup().toIntVector()[0]){
                    return 1;
                }else {
                    return 0;
                }
            })).limit(beam_width).collect(Collectors.toList());
        }
        return u;
    }

    public static class Data{
        public ArrayList array;

        public INDArray num;

        public Data(ArrayList array, INDArray num){
            this.array = array;
            this.num = num;
        }

        public List getArray() {
            return array;
        }

        public void setArray(ArrayList array) {
            this.array = array;
        }

        public INDArray getNum() {
            return num;
        }

        public void setNum(INDArray num) {
            this.num = num;
        }
    }
}
