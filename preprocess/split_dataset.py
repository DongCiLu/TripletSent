# Split the data into train and test
import os
import numpy as np
from shutil import copyfile

'''
src_dir = "../datasets/sentibank_flickr/preprocessed_256"
train_base_dir = "../datasets/sentibank_flickr/regular_256/train"
test_base_dir = "../datasets/sentibank_flickr/regular_256/test"
'''
src_dir = "../datasets/google/preprocessed"
train_base_dir = "../datasets/google/regular/train"
test_base_dir = "../datasets/google/regular/test"

split_ratio = 0.7 
# hard_cap = 20

for subdir, dirs, files in os.walk(src_dir):
    if src_dir == subdir:
        continue
    # create directories
    segs = subdir.split('/')
    ANP = segs[-1]
    train_dir = os.path.join(train_base_dir, ANP)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    test_dir = os.path.join(test_base_dir, ANP)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    train_cnt = 0
    test_cnt = 0
    total_cnt = 0
    # split files
    for f in files:
        total_cnt += 1
        src_fp = os.path.join(subdir, f)
        r = np.random.uniform()
        if r < split_ratio:
        # if r < split_ratio or test_cnt >= hard_cap:
            dst_fp = os.path.join(train_dir, f)
            train_cnt += 1
        else:
            dst_fp = os.path.join(test_dir, f)
            test_cnt += 1
        copyfile(src_fp, dst_fp)

    print ("{} has {} images, {} moved to training dataset, {} moved to testing dataset.".format(ANP, total_cnt, train_cnt, test_cnt))
