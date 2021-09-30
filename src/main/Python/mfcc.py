from scipy.io import wavfile as wav
from spafe.utils import vis
from spafe.features.mfcc import mfcc, imfcc
from matplotlib import pyplot as plt

# init input vars
num_ceps = 13
low_freq = 0
high_freq = 2000
nfilts = 24
nfft = 512
lifter = 5
normalize = False

# read wav
fs, sig = wav.read("../resources/sample-000000.wav")

# compute features
mfccs = mfcc(sig=sig,
             fs=fs,
             num_ceps=num_ceps,
             nfilts=nfilts,
             nfft=nfft,
             low_freq=low_freq,
             high_freq=high_freq,
             lifter=lifter,
             normalize=normalize)

# visualize spectogram
# vis.spectogram(sig, fs)
# plt.specgram(sig, NFFT=1024, Fs=fs)
# plt.ylabel("Frequency (kHz)")
# plt.xlabel("Time (s)")
# plt.show(block=False)
# visualize features
# vis.visualize_features(mfccs, 'MFCC Index', 'Frame Index')
plt.imshow(mfccs.T,
           origin='lower',
           aspect='auto',
           cmap='viridis',
           interpolation='nearest')
plt.ylabel('MFCC Index')
plt.xlabel('Frame Index')
plt.show(block=False)

# compute features
# imfccs = imfcc(sig=sig,
#                fs=fs,
#                num_ceps=num_ceps,
#                nfilts=nfilts,
#                nfft=nfft,
#                low_freq=low_freq,
#                high_freq=high_freq,
#                lifter=lifter,
#                normalize=normalize)
#
# # visualize features
# vis.visualize_features(imfccs, 'IMFCC Index', 'Frame Index')
input()