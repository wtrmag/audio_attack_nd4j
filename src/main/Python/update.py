import argparse
import numpy as np
import sys
import os

import torch
from torch.autograd import Variable
from torch.optim import Adam

# from beam_search import beam_decode
sys.path.append("src/main/DeepSpeech")
import DeepSpeech

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, required=True, dest="target")
    parser.add_argument('-l', type=str, required=True, dest="target_length")
    parser.add_argument('-r', type=int, required=True, dest="lr")
    parser.add_argument('-s', type=int, required=True, dest="lengths")
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()

    target = torch.from_numpy(np.load(args.target))
    target_length = torch.from_numpy(np.load(args.target_length))
    logit = torch.from_numpy(np.load("src/main/resources/temp/logit.npy")).log_softmax(2)
    length = torch.tensor([logit.shape[0]])

    ctcloss = torch.nn.CTCLoss()
    loss = ctcloss(logit, target, length, target_length).requires_grad_()

    print(loss)

    delta = []
    for i in range(args.lengths):
        a = Variable(torch.zeros([1, 51072]), requires_grad=True)
        optimizer = Adam([a, loss], lr=args.lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        delta.append(loss.item())


    file = "src/main/resources/temp/"
    if not os.path.exists(file):
        os.mkdir(file)
    np.save(os.path.join(file, "result.npy"), [delta])
