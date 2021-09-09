package utils;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import pojo.Variables;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * @author wtr
 */
public class PyCall {

    public static INDArray CallDeepSpeech(INDArray features, INDArray lengths){
        INDArray logits = null;
        try {
            DataConvert.export(Variables.TEMP, features, lengths);
            String[] command = new String[]{Variables.PYTHON_PATH, "src/main/Python/call.py", "-f"+Variables.TEMP+DataConvert.NPY_NAME[0], "-l"+Variables.TEMP+DataConvert.NPY_NAME[1]};
            Process p = Runtime.getRuntime().exec(command);
            // 防止缓冲区满, 导致卡住
            new Thread() {
                @Override
                public void run() {
                    super.run();
                    String line;
                    try {
                        BufferedReader stderr = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                        while ((line = stderr.readLine()) != null) {
                            System.out.println(line);
                        }
                    }
                    catch (Exception e) {

                    }

                }
            }.start();


            new Thread() {
                @Override
                public void run() {
                    super.run();
                    String line;
                    try {
                        BufferedReader stdout = new BufferedReader(new InputStreamReader(p.getInputStream()));
                        while ((line = stdout.readLine()) != null) {
                            System.out.println(line);
                        }
                    }
                    catch (Exception e) {

                    }
                }
            }.start();

            int exitVal = p.waitFor();
            if (0 != exitVal) {
                throw new Exception("fail to call DeepSpeech");
            }else {
                System.out.println("success");
                logits = Nd4j.createFromNpyFile(new File(Variables.TEMP+DataConvert.NPY_NAME[2]));
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        return logits;
    }

    public static INDArray UpdateDelta(INDArray loss, INDArray delta){
        INDArray delta_new = null;
        try {
            DataConvert.export(Variables.TEMP, loss, delta);
            String[] command = new String[]{Variables.PYTHON_PATH, "src/main/Python/update.py", "-t"+Variables.TEMP+DataConvert.NPY_NAME[0], "-l"+Variables.TEMP+DataConvert.NPY_NAME[1]};
            Process p = Runtime.getRuntime().exec(command);
            // 防止缓冲区满, 导致卡住
            new Thread() {
                @Override
                public void run() {
                    super.run();
                    String line;
                    try {
                        BufferedReader stderr = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                        while ((line = stderr.readLine()) != null) {
                            System.out.println(line);
                        }
                    }
                    catch (Exception e) {

                    }

                }
            }.start();


            new Thread() {
                @Override
                public void run() {
                    super.run();
                    String line;
                    try {
                        BufferedReader stdout = new BufferedReader(new InputStreamReader(p.getInputStream()));
                        while ((line = stdout.readLine()) != null) {
                            System.out.println(line);
                        }
                    }
                    catch (Exception e) {

                    }
                }
            }.start();

            int exitVal = p.waitFor();
            if (0 != exitVal) {
                throw new Exception("fail to Update delta");
            }else {
                System.out.println("success");
                delta_new = Nd4j.createFromNpyFile(new File(Variables.TEMP+DataConvert.NPY_NAME[3]));
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        return delta_new;
    }
}
