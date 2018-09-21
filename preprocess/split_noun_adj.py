# Split the data into train and test
import os
import numpy as np
from shutil import copyfile

train_src_dir = "../datasets/sentibank_flickr/regular_256/train"
test_src_dir = "../datasets/sentibank_flickr/regular_256/test"

train_noun_base_dir = "../datasets/sentibank_flickr/regular_256/train_noun"
test_noun_base_dir = "../datasets/sentibank_flickr/regular_256/test_noun"

train_adj_base_dir = "../datasets/sentibank_flickr/regular_256/train_adj"
test_adj_base_dir = "../datasets/sentibank_flickr/regular_256/test_adj"

for subdir, dirs, files in os.walk(train_src_dir):
    if subdir == train_src_dir:
        continue
    # create directories
    segs = subdir.split('/')
    ANP = segs[-1]
    segs = ANP.split('_')
    adj = segs[0]
    noun = segs[1]

    train_noun_dir = os.path.join(train_noun_base_dir, noun)
    if not os.path.exists(train_noun_dir):
        os.mkdir(train_noun_dir)
    train_adj_dir = os.path.join(train_adj_base_dir, adj)
    if not os.path.exists(train_adj_dir):
        os.mkdir(train_adj_dir)

    # split files
    for f in files:
        src_fp = os.path.join(subdir, f)
        noun_dst_fp = os.path.join(train_noun_dir, f)
        adj_dst_fp = os.path.join(train_adj_dir, f)
        copyfile(src_fp, noun_dst_fp)
        copyfile(src_fp, adj_dst_fp)

for subdir, dirs, files in os.walk(test_src_dir):
    if subdir == test_src_dir:
        continue
    # create directories
    segs = subdir.split('/')
    ANP = segs[-1]
    segs = ANP.split('_')
    adj = segs[0]
    noun = segs[1]

    test_noun_dir = os.path.join(test_noun_base_dir, noun)
    if not os.path.exists(test_noun_dir):
        os.mkdir(test_noun_dir)
    test_adj_dir = os.path.join(test_adj_base_dir, adj)
    if not os.path.exists(test_adj_dir):
        os.mkdir(test_adj_dir)

    # split files
    for f in files:
        src_fp = os.path.join(subdir, f)
        noun_dst_fp = os.path.join(test_noun_dir, f)
        adj_dst_fp = os.path.join(test_adj_dir, f)
        copyfile(src_fp, noun_dst_fp)
        copyfile(src_fp, adj_dst_fp)
