import pojo.Attack;
import pojo.Variables;
import utils.waveaccess.WaveFileReader;
import utils.waveaccess.WaveFileWriter;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;
import java.util.stream.Collectors;

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

    @Option(name = "-r", aliases = "restore_path",usage = "path of the DeepSpeech checkpoint")
    private String restore_path;

    public void do_main(String[] args) throws Exception {
        CmdLineParser parser = new CmdLineParser(this);
        if (args.length < 1 ) {
            parser.printUsage(System.out);
            System.exit(-1);
        }

        parser.parseArgument(args);

        if (StringUtils.equals(this.out, null)) {
            assert !StringUtils.equals(this.out_prefix, null);
        }else {
            assert StringUtils.equals(this.out_prefix, null);
            assert this.input.length() == this.out.length();
        }
        if (!StringUtils.equals(this.finetune, null)) {
            assert this.input.length() == this.finetune.length();
        }

        //"src/main/resources/sample-000000.wav"
        WaveFileReader reader = new WaveFileReader(this.input);
        assert reader.getSampleRate() == 16000;

        int maxlen = Arrays.stream(reader.getData()).map(n -> n.length).collect(Collectors.toList()).stream().max(Integer::compareTo).get();

        //todo print dB
//        reader.getData()

        WaveFileReader reader2 = null;
        int[][] finetune = new int[][]{};
        if (!StringUtils.equals(this.finetune, null)){
            reader2 = new WaveFileReader(this.input);
            finetune = Arrays.stream(reader2.getData()).map(n -> {
                int[] temp = new int[maxlen-n.length];
                return ArrayUtils.addAll(n, temp);
            }).collect(Collectors.toList()).toArray(new int[][]{});
        }

        int[][] audios = Arrays.stream(reader.getData()).map(n -> {
            int[] temp = new int[maxlen-n.length];
            return ArrayUtils.addAll(n, temp);
        }).collect(Collectors.toList()).toArray(new int[][]{});

        int[] lengths = Arrays.stream(reader.getData()).mapToInt(n -> n.length).toArray();

        int[][] index = new int[audios.length][];
        for (int i = 0; i < audios.length; i++) {
            index[i] = new int[this.target.length()];
            for (int j = 0; j < this.target.length(); j++) {
                char c = this.target.charAt(j);
                index[i][j] = Variables.TOKENS.indexOf(c);
            }
        }

        Attack attack = new Attack("CTC", this.target.length(), maxlen, this.learning_rate,
                this.iterations, audios.length, this.mp3, this.l2penalty, this.restore_path);

        double[][] deltas = attack.do_attack(audios, lengths, index, finetune);

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
                INDArray out = Nd4j.create(Arrays.stream(deltas[0]).limit(lengths[0]).toArray());
                int[][] r = Nd4j.math.clipByValue(Nd4j.math.round(out), Math.pow(-2, 15), Math.pow(2, 15)-1).toIntMatrix();
                WaveFileWriter writer = new WaveFileWriter(Variables.RESULTS+path, r, 16000);
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
