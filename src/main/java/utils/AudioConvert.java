package utils;

import pojo.Variables;
import utils.waveaccess.WaveFileWriter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import ws.schild.jave.Encoder;
import ws.schild.jave.MultimediaObject;
import ws.schild.jave.encode.AudioAttributes;
import ws.schild.jave.encode.EncodingAttributes;

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

        //Todo


        return new double[0];
    }

    /**
     *  wav 转 MP3
     * @param source
     * @param target
     */
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
}
