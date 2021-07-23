package Pojo;

import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Operation;
import org.tensorflow.framework.*;


/**
 * @author wtr
 */
public class Variables {

    public static final float infinity = (float) (1 / 0.0);

    public static final long[] perm = {0, 1, 2};

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

    //Todo unsure
    public Operation train;

    //Todo unsure
    public int[] decode;

    public Variables(INDArray delta, INDArray mask, INDArray cw_mask, INDArray original, INDArray lengths, INDArray importance, INDArray target_phrase, INDArray target_phrase_lengths, INDArray rescale, INDArray apply_delta, INDArray new_input, INDArray logits, INDArray loss, INDArray ctc_loss, Operation train, int[] decode) {
        this.delta = delta;
        this.mask = mask;
        this.cw_mask = cw_mask;
        this.original = original;
        this.lengths = lengths;
        this.importance = importance;
        this.target_phrase = target_phrase;
        this.target_phrase_lengths = target_phrase_lengths;
        this.rescale = rescale;
        this.apply_delta = apply_delta;
        this.new_input = new_input;
        this.logits = logits;
        this.loss = loss;
        this.ctc_loss = ctc_loss;
        this.train = train;
        this.decode = decode;
    }

    public INDArray getDelta() {
        return delta;
    }

    public void setDelta(INDArray delta) {
        this.delta = delta;
    }

    public INDArray getMask() {
        return mask;
    }

    public void setMask(INDArray mask) {
        this.mask = mask;
    }

    public INDArray getCw_mask() {
        return cw_mask;
    }

    public void setCw_mask(INDArray cw_mask) {
        this.cw_mask = cw_mask;
    }

    public INDArray getOriginal() {
        return original;
    }

    public void setOriginal(INDArray original) {
        this.original = original;
    }

    public INDArray getLengths() {
        return lengths;
    }

    public void setLengths(INDArray lengths) {
        this.lengths = lengths;
    }

    public INDArray getImportance() {
        return importance;
    }

    public void setImportance(INDArray importance) {
        this.importance = importance;
    }

    public INDArray getTarget_phrase() {
        return target_phrase;
    }

    public void setTarget_phrase(INDArray target_phrase) {
        this.target_phrase = target_phrase;
    }

    public INDArray getTarget_phrase_lengths() {
        return target_phrase_lengths;
    }

    public void setTarget_phrase_lengths(INDArray target_phrase_lengths) {
        this.target_phrase_lengths = target_phrase_lengths;
    }

    public INDArray getRescale() {
        return rescale;
    }

    public void setRescale(INDArray rescale) {
        this.rescale = rescale;
    }

    public INDArray getApply_delta() {
        return apply_delta;
    }

    public void setApply_delta(INDArray apply_delta) {
        this.apply_delta = apply_delta;
    }

    public INDArray getNew_input() {
        return new_input;
    }

    public void setNew_input(INDArray new_input) {
        this.new_input = new_input;
    }

    public INDArray getLogits() {
        return logits;
    }

    public void setLogits(INDArray logits) {
        this.logits = logits;
    }

    public INDArray getLoss() {
        return loss;
    }

    public void setLoss(INDArray loss) {
        this.loss = loss;
    }

    public INDArray getCtc_loss() {
        return ctc_loss;
    }

    public void setCtc_loss(INDArray ctc_loss) {
        this.ctc_loss = ctc_loss;
    }

    public Operation getTrain() {
        return train;
    }

    public void setTrain(Operation train) {
        this.train = train;
    }

    public int[] getDecode() {
        return decode;
    }

    public void setDecode(int[] decode) {
        this.decode = decode;
    }
}
