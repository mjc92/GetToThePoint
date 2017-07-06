import argparse
import numpy as np
from packages.vocab import Vocab
from packages.batch import Batch
from model import Model
from packages.functions import to_cuda
import os
import random

parser = argparse.ArgumentParser()

# arguments related to the dataset
parser.add_argument("--data_dir",type=str, default='/home/mjc/datasets/CNN_DailyMail/cnn/stories_merged_100/',
                    help='directory where data files are located')
parser.add_argument("--word2idx",type=str, default='word2idx.npy', help='file name for word2idx file')
parser.add_argument("--idx2word",type=str, default='idx2word.npy', help='file name for idx2word file')
parser.add_argument("--max_enc",type=int, default=400, help='max length of encoder sequence')
parser.add_argument("--max_dec",type=int, default=100, help='max length of decoder sequence')
parser.add_argument("--min_dec",type=int, default=35, help='min length of decoder sequence')
parser.add_argument("--vocab_size",type=int, default=50000, help='vocabulary size')
parser.add_argument("--max_oovs",type=int, default=20, help='max number of OOVs to accept in a sample')


# arguments related to model training and inference
parser.add_argument("--train",type=bool, default=True, help='train/test model. Set by default to True(=train)')
parser.add_argument("--epochs",type=int, default=20, help='Number of epochs. Set by default to 20')
parser.add_argument("--load_model",type=str, default='', help='input model name to start from a pretrained model')
parser.add_argument("--hidden",type=int, default=256, help='size of hidden dimension')
parser.add_argument("--embed",type=int, default=128, help='size of embedded word dimension')
parser.add_argument("--lr",type=float, default=0.15, help='learning rate')
parser.add_argument("--cov_lambda",type=float, default=1.0, help='lambda for coverage loss')
parser.add_argument("--beam",type=int, default=4, help='beam size')
parser.add_argument("--cuda",type=bool, default=True, help='whether to use GPU')

args = parser.parse_args()

def main(args):
    # obtain vocabulary
    vocab = Vocab(args.vocab_size)
    vocab.w2i = np.load(args.word2idx).item()
    vocab.i2w = np.load(args.idx2word).item()
    vocab.count = len(vocab.w2i)

    # obtain dataset in batches
    file_list = os.listdir(args.data_dir)
    batch = Batch(file_list, args.max_enc, args.max_dec)

    # load model
    if args.load_model != '':
        model = torch.load(args.load_model)
    else:
        model = Model(args)
    model = to_cuda(model)

    # computation for each epoch
    epoch = 1
    while (epoch<=args.epochs):
        random.shuffle(file_list)
        for file in file_list:
            with open(os.path.join(args.data_dir,file)) as f:
                minibatch = f.read()
            stories,summaries = batch.process_minibatch(minibatch,vocab)
            print(stories)
            print(summaries)

if __name__ == "__main__":
    main(args)


