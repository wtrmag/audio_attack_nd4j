import argparse
import tensorflow as tf
import sys

sys.path.append("DeepSpeech")
import DeepSpeech

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=list, required=True, dest="features")
parser.add_argument('-l', type=list, required=True, dest="lengths")
args = parser.parse_args()
while len(sys.argv) > 1:
    sys.argv.pop()

with tf.Session() as sess:
    DeepSpeech.create_flags()
    tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
    DeepSpeech.initialize_globals()
    logits, _ = DeepSpeech.BiRNN(args.features, args.lengths, [0] * 10)
    a = sess.run(logits)
    print(a)
