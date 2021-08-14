import Pojo.Attack;
import Pojo.Variables;
import Utils.waveaccess.WaveFileReader;
import Utils.waveaccess.WaveFileWriter;
import com.github.javaparser.metamodel.OptionalProperty;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.OptionDef;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Print;
import org.tensorflow.proto.framework.NoneValue;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Audio_Attack {

    @Option(name = "-i", aliases = "--in", usage = "path of input audio", required = true)
    private String input;

    @Option(name = "-t", aliases = "--target", required = true)
    private String target;

    @Option(name = "-o", aliases = "--out")
    private String out;

    @Option(name = "-p", aliases = "--prefix")
    private String out_prefix;

    @Option(name = "-f", aliases = "--finetune")
    private String finetune;

    @Option(name = "-l", aliases = "--lr", usage = "learing rate of optimization")
    private int learning_rate = 100;

    @Option(name = "-it", aliases = "--iterations")
    private int iterations = 1000;

    @Option(name = "-l2", aliases = "--l2penalty")
    private float l2penalty = Float.POSITIVE_INFINITY;

    @Option(name = "-m", aliases = "--mp3")
    private boolean mp3 = false;

    @Option(name = "-r", aliases = "restore_path",usage = "path of the DeepSpeech checkpoint", required = true)
    private String restore_path;

    public void do_main(String[] args) throws Exception {
        CmdLineParser parser = new CmdLineParser(this);
        if (args.length < 1 ) {
            parser.printUsage(System.out);
            System.exit(-1);
        }

        parser.parseArgument(args);

        if (StringUtils.equals(this.out, "")) {
            assert !StringUtils.equals(this.out_prefix, "");
        }else {
            assert StringUtils.equals(this.out_prefix, "");
            assert this.input.length() == this.out.length();
        }
        if (!StringUtils.equals(this.finetune, "")) {
            assert this.input.length() == this.finetune.length();
        }

        //"src/main/resources/sample-000000.wav"
        WaveFileReader reader = new WaveFileReader(this.input);
        assert reader.getSampleRate() == 16000;
        //todo print dB
//        reader.getData()


        WaveFileReader reader2 = null;
        if (!StringUtils.equals(this.finetune, "")){
            reader2 = new WaveFileReader(this.input);
        }

        int maxlen = Arrays.stream(reader.getData()).map(n -> n.length).collect(Collectors.toList()).stream().max(Integer::compareTo).get();
//        INDArray audios = Nd4j.create(Arrays.stream(reader.getData()).map(n -> {
//            int[] temp = new int[maxlen-n.length];
//            return ArrayUtils.addAll(n, temp);
//        }).collect(Collectors.toList()).toArray(new int[][]{}));
        int[][] audios = Arrays.stream(reader.getData()).map(n -> {
            int[] temp = new int[maxlen-n.length];
            return ArrayUtils.addAll(n, temp);
        }).collect(Collectors.toList()).toArray(new int[][]{});
//        INDArray finetune = Nd4j.create(Arrays.stream(reader2.getData()).map(n -> {
//            int[] temp = new int[maxlen-n.length];
//            return ArrayUtils.addAll(n, temp);
//        }).collect(Collectors.toList()).toArray(new int[][]{}));
        int[][] finetune = Arrays.stream(reader2.getData()).map(n -> {
            int[] temp = new int[maxlen-n.length];
            return ArrayUtils.addAll(n, temp);
        }).collect(Collectors.toList()).toArray(new int[][]{});

        Graph graph = new Graph();
        Ops tf = Ops.create(graph);
        Session session = new Session(graph);
        Attack attack = new Attack("CTC", this.target.length(), maxlen, this.learning_rate,
                this.iterations, audios.length, this.mp3, this.l2penalty, this.restore_path);
        //todo do_attack方法待完成
//        attack.do_attack();

        String path = null;
        if (this.mp3) {
            //Todo convert_mp3
            throw new Exception("unfinshed");
        }else {
            if (StringUtils.equals("", this.out)) {
                path = this.out;
            }else {
                path = this.out_prefix + ".wav";
            }
            File file = new File(Variables.RESULTS);
            if (!file.exists()){
                file.mkdirs();
            }else {
                //Todo 输出对抗样本，需要do_attack返回值
//                INDArray out = Nd4j
//                WaveFileWriter writer = new WaveFileWriter(Variables.RESULTS+path, , 16000);

            }

        }




    }

    public static void main(String[] args){
        Audio_Attack instance = new Audio_Attack();
        try {
            instance.do_main(args);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
