import numpy as np
import spacy
import os
from collections import Counter
import torch
import glob
from spacy import attrs

vocab_size = 50000
batch_size = 1000

nlp = spacy.load('en') # loads default English object
cnn_dir = '/home/mjc/datasets/CNN_DailyMail/cnn/stories/'
cnn_pre_dir = '/home/mjc/datasets/CNN_DailyMail/cnn/preprocessed_stories/'
# cnn_dir = '/home/mjc/datasets/CNN_DailyMail/dailymail/stories/'

file_list = [os.path.join(cnn_dir,file) for file in os.listdir(cnn_dir)]
total_files = len(file_list)
files_read = 0

counter = Counter()

while (files_read<total_files):
    word_list = []
    batch_files = file_list[files_read:min(files_read+1000,total_files)]
    for file_name in batch_files:
        with open(file_name) as f:
            text = f.read()
            text = text.lower()
            text = text.replace('\n\n',' ')
            text = text.replace('(cnn)','')
            text = text.split("@highlight")
            body = text[0]
            doc = list(nlp(body))
            word_list.extend([x.text for x in doc])

    counter = counter + Counter(word_list)
    files_read+=len(batch_files)
    print("%d files read so far..." % files_read)
    word2idx = {tup[0]: i for i,tup in enumerate(counter.most_common(vocab_size))}
    np.save('word2idx.npy',word2idx)
print("All merged!")
word2idx = {tup[0]: i for i,tup in enumerate(counter.most_common(vocab_size))}
np.save('word2idx.npy',word2idx)