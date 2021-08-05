package Pojo;


/**
 * @author wtr
 */
public enum options_sharedname {
    delta("qq_delta"),
    mask("qq_mask"),
    cwmask("qq_cwmask"),
    original("qq_original"),
    lengths("qq_lengths"),
    importance("qq_importance"),
    phrase("qq_phrase"),
    phrase_lengths("qq_phrase_lengths"),
    rescale("qq_rescale");

    private String name;

    /**
     * 私有构造,防止被外部调用
     * @param s
     */
    private options_sharedname(String s) {
        this.name=s;
    }

    public String getName() {
        return this.name;
    }
}
