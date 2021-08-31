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

    public static INDArray CallDeepSpeech(INDArray fetures, INDArray lengths){
        INDArray logits = null;
        try {
            DataConvert.export(Variables.TEMP, fetures, lengths);

            String[] command = new String[]{Variables.PYTHON_PATH, "src/main/call.py", "-f"+Variables.TEMP+DataConvert.NPY_NAME[0], "-l"+Variables.TEMP+DataConvert.NPY_NAME[1]};
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
                throw new Exception("fail to call py");
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

    public static void main(String[] args){
        double[][][] p = new double[][][]{
                {{-9.14318,-3.39702,-5.01707,-7.49291,-7.13737,-6.35238,-11.64279,-6.48578,-7.41815,-2.76987,-15.96768,-10.36601,-11.34054,-7.93783,-8.40006,-6.57689,-5.70179,-21.74776,-14.55497,-7.42479,-3.18959,-5.39941,-15.00021,-9.48480,-21.50571,-5.40170,-20.05874,-9.13122,13.64532}},
                {{-8.77652,-3.83748,-3.05141,-7.20004,-11.05040,-6.47998,-13.28422,-9.59072,-5.04566,-2.74999,-16.88125,-13.27481,-11.48200,-8.76435,-10.50619,-6.81913,-4.86955,-22.99290,-13.31697,-8.97902,-3.15393,-4.67254,-15.92128,-5.63666,-24.50668,-5.94451,-22.35293,-11.52456,17.04568}}
        };
        double[] l = new double[]{51072};
        INDArray x = Nd4j.createFromArray(p);
        INDArray y = Nd4j.createFromArray(l);

        List<String> list = new ArrayList<>();
        list.add(x.toString());
        list.add(y.toString());

//        DataConvert.exportCsv(new File(Variables.TEMP + "data.csv"), list);
        System.out.println(x);
        System.out.println(x.toString());
        System.out.println(x.toStringFull());
//        CallDeepSpeech(x , y);
//        System.out.println(x.toString());

//        PythonInterpreter interpreter = new PythonInterpreter();
//        interpreter.execfile("src/main/call.py");
//
//        PyFunction pyFunction = interpreter.get("toStr", PyFunction.class);
//        PyString str = new PyString("what");
//        PyObject pyObject = pyFunction.__call__(str); // 调用函数
//
//        System.out.println(pyObject);


    }
}
