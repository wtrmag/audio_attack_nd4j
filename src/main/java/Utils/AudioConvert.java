package Utils;

import Pojo.Variables;
import Utils.waveaccess.WaveFileWriter;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import ws.schild.jave.*;
import ws.schild.jave.encode.AudioAttributes;
import ws.schild.jave.encode.EncodingAttributes;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.spi.AudioFileReader;
import java.io.File;
import java.util.Arrays;

public class AudioConvert {

    public static double[] convert_mp3(int[][] data, int length) throws Exception {
        INDArray array = Nd4j.create(Arrays.stream(data[0]).limit(length).toArray());
        int[][] out = Nd4j.math().clipByValue(Nd4j.math().round(array), Math.pow(-2, 15), Math.pow(2, 15) - 1).toIntMatrix();

        File file = new File(Variables.TEMP);
        if (!file.exists()){
            file.mkdirs();
        }else {
            WaveFileWriter writer = new WaveFileWriter(Variables.TEMP+"load.wav", out, 16000);
        }
        File wav = new File(Variables.TEMP+"load.wav");
        File target = new File(Variables.TEMP+"saved.mp3");
        if (!file.exists()){
            throw new Exception("生成wav文件失败");
        }else {
            trans(wav, target);
        }


        return new double[0];
    }

    public static void trans(File source, File target) {
        try {
            //Audio Attributes
            AudioAttributes audio = new AudioAttributes();
            audio.setCodec("libmp3lame");
            audio.setBitRate(16000);
            audio.setChannels(2);
            audio.setSamplingRate(44100);

            //Encoding attributes
            EncodingAttributes attrs = new EncodingAttributes();
            attrs.setOutputFormat("mp3");
            attrs.setAudioAttributes(audio);

            //Encode
            Encoder encoder = new Encoder();
            encoder.encode(new MultimediaObject(source), target, attrs);
            System.out.println("transport success");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

//    public static void main(String[] args){
//        File s = new File("src/main/resources/sample-000000.wav");
//        s.setReadable(true);
//        File t = new File("src/main/resources/test.mp3");
//        t.setWritable(true);
//
//        trans(s, t);
//    }
}
