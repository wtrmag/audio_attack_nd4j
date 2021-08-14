package Lib;

import org.nd4j.linalg.api.ndarray.INDArray;


public class SparseTensor {
    private final INDArray indices;
    private final INDArray values;
    private final INDArray denseShape;

    public SparseTensor (INDArray indices, INDArray values, INDArray denseShape) {
        this.indices =  indices;
        this.values =  values;
        this.denseShape = denseShape;
    }

    public INDArray getIndices() {
        return indices;
    }

    public INDArray getValues() {
        return values;
    }

    public INDArray getDenseShape() {
        return denseShape;
    }
}
