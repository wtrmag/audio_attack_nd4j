package pojo;

import java.util.Objects;

public class Complex {
    public double re; // the real part
    public double im; // the imaginary part

    // create a new object with the given real and imaginary parts
    public Complex(double real, double imag) {
        re = real;
        im = imag;
    }

    // return a string representation of the invoking Complex object
    @Override
    public String toString() {
        if (im == 0) {
            return re + "";
        }
        if (re == 0) {
            return im + "i";
        }
        if (im < 0) {
            return re + " - " + (-im) + "i";
        }
        return re + " + " + im + "i";
    }

    // return abs/modulus/magnitude
    public double abs() {
        return Math.hypot(re, im);
    }

    // return angle/phase/argument, normalized to be between -pi and pi
    public double phase() {
        return Math.atan2(im, re);
    }

    // return a new Complex object whose value is (this + b)
    public Complex plus(Complex b) {
        Complex a = this; // invoking object
        double real = a.re + b.re;
        double imag = a.im + b.im;
        return new Complex(real, imag);
    }

    // return a new Complex object whose value is (this - b)
    public Complex minus(Complex b) {
        Complex a = this;
        double real = a.re - b.re;
        double imag = a.im - b.im;
        return new Complex(real, imag);
    }

    // return a new Complex object whose value is (this * b)
    public Complex multiple(Complex b) {
        Complex a = this;
        double real = a.re * b.re - a.im * b.im;
        double imag = a.re * b.im + a.im * b.re;
        return new Complex(real, imag);
    }

    // scalar multiplication
    // return a new object whose value is (this * alpha)
    public Complex multiple(double alpha) {
        return new Complex(alpha * re, alpha * im);
    }

    // return a new object whose value is (this * alpha)
    public Complex scale(double alpha) {
        return new Complex(alpha * re, alpha * im);
    }

    // return a new Complex object whose value is the conjugate of this
    public Complex conjugate() {
        return new Complex(re, -im);
    }

    // return a new Complex object whose value is the reciprocal of this
    public Complex reciprocal() {
        double scale = re * re + im * im;
        return new Complex(re / scale, -im / scale);
    }

    // return the real or imaginary part
    public double re() {
        return re;
    }

    public double im() {
        return im;
    }

    // return a / b
    public Complex divides(Complex b) {
        Complex a = this;
        return a.multiple(b.reciprocal());
    }

    // return a new Complex object whose value is the complex exponential of
    // this
    public Complex exp() {
        return new Complex(Math.exp(re) * Math.cos(im), Math.exp(re) * Math.sin(im));
    }

    // return a new Complex object whose value is the complex sine of this
    public Complex sin() {
        return new Complex(Math.sin(re) * Math.cosh(im), Math.cos(re) * Math.sinh(im));
    }

    // return a new Complex object whose value is the complex cosine of this
    public Complex cos() {
        return new Complex(Math.cos(re) * Math.cosh(im), -Math.sin(re) * Math.sinh(im));
    }

    // return a new Complex object whose value is the complex tangent of this
    public Complex tan() {
        return sin().divides(cos());
    }

    // a static version of plus
    public static Complex plus(Complex a, Complex b) {
        double real = a.re + b.re;
        double imag = a.im + b.im;
        Complex sum = new Complex(real, imag);
        return sum;
    }

    // See Section 3.3.
    @Override
    public boolean equals(Object x) {
        if (x == null) {
            return false;
        }
        if (this.getClass() != x.getClass()) {
            return false;
        }
        Complex that = (Complex) x;
        return (this.re == that.re) && (this.im == that.im);
    }

    // See Section 3.3.
    @Override
    public int hashCode() {
        return Objects.hash(re, im);
    }




    public static Complex[] fft(Complex[] x) {
        int n = x.length;

        // 因为exp(-2i*n*PI)=1，n=1时递归原点
        if (n == 1){
            //  这里和B博客中有一点变化
            return new Complex[]{x[0]};
        }

        // 如果信号数为奇数，使用dft计算
        if (n % 2 != 0) {
            return dft(x);
        }

        // 提取下标为偶数的原始信号值进行递归fft计算
        Complex[] even = new Complex[n / 2];
        for (int k = 0; k < n / 2; k++) {
            even[k] = x[2 * k];
        }
        Complex[] evenValue = fft(even);

        // 提取下标为奇数的原始信号值进行fft计算
        // 节约内存
        Complex[] odd = even;
        for (int k = 0; k < n / 2; k++) {
            odd[k] = x[2 * k + 1];
        }
        Complex[] oddValue = fft(odd);

        // 偶数+奇数
        Complex[] result = new Complex[n];
        for (int k = 0; k < n / 2; k++) {
            // 使用欧拉公式e^(-i*2pi*k/N) = cos(-2pi*k/N) + i*sin(-2pi*k/N)
            double p = -2 * k * Math.PI / n;
            Complex m = new Complex(Math.cos(p), Math.sin(p));
            result[k] = evenValue[k].plus(m.multiple(oddValue[k]));
            // exp(-2*(k+n/2)*PI/n) 相当于 -exp(-2*k*PI/n)，其中exp(-n*PI)=-1(欧拉公式);
            result[k + n / 2] = evenValue[k].minus(m.multiple(oddValue[k]));
        }
        return result;
    }

    public static Complex[] dft(Complex[] x) {
        int n = x.length;

        // exp(-2i*n*PI)=cos(-2*n*PI)+i*sin(-2*n*PI)=1
        if (n == 1) {
            return new Complex[]{x[0]};
        }

        Complex[] result = new Complex[n];
        for (int i = 0; i < n; i++) {
            result[i] = new Complex(0, 0);
            for (int k = 0; k < n; k++) {
                //使用欧拉公式e^(-i*2pi*k/N) = cos(-2pi*k/N) + i*sin(-2pi*k/N)
                double p = -2 * i * k* Math.PI / n;
                Complex m = new Complex(Math.cos(p), Math.sin(p));
                result[i] = result[i].plus(x[k].multiple(m));
            }
        }
        return result;
    }

    public static double[] Covlex(Complex[] x){
        int j;
        int i=x.length;
        double[] result=new  double[i];
        for(j=0;j<i;j++)
        {
            result[j]=x[j].re;
        }
        return result;

    }
}



