public class Attack implements Variables {
    private String loss_fn;

    private int phrase_length;

    private int max_audio_len;

    private int learning_rate;

    private int num_iterations;

    private int batch_size;

    private boolean mp3;

    //默认值为正无穷
    private float l2penalty;

    private String restore_path;

    //Todo session待定
    public Attack(String loss_fn, int phrase_length, int max_audio_len, int learning_rate, int num_iterations, int batch_size, boolean mp3, float l2penalty, String restore_path) {
        this.loss_fn = loss_fn;
        this.phrase_length = phrase_length;
        this.max_audio_len = max_audio_len;
        this.learning_rate = learning_rate;
        this.num_iterations = num_iterations;
        this.batch_size = batch_size;
        this.mp3 = mp3;
        this.l2penalty = l2penalty;
        this.restore_path = restore_path;
    }

}
