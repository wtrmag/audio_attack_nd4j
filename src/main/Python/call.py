import argparse
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append("src/main/DeepSpeech")
import DeepSpeech

if __name__ == '__main__':
    with tf.Session() as sess:
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', type=str, required=True, dest="features")
        parser.add_argument('-l', type=str, required=True, dest="lengths")
        args = parser.parse_args()
        while len(sys.argv) > 1:
            sys.argv.pop()

        features = np.load(args.features)
        lengths = np.load(args.lengths)

        DeepSpeech.create_flags()
        tf.app.flags.FLAGS.alphabet_config_path = "src/main/DeepSpeech/data/alphabet.txt"
        DeepSpeech.initialize_globals()

        logits, _ = DeepSpeech.BiRNN(features, lengths, [0] * 10)
        sess.run(tf.global_variables_initializer())
        l = sess.run(logits)

        file = "src/main/resources/temp/"
        if not os.path.exists(file):
            os.mkdir(file)
        np.save(os.path.join(file, "logit.npy"), l)