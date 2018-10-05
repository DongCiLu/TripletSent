# Split the data into train and test
import os
import six
import numpy as np
from PIL import Image

import imagehash
hashfunc = imagehash.average_hash

src_dir = "../datasets/google/preprocessed"

total_cnt = 0
existing_img = {}
duplicate_img = {}
for subdir, dirs, files in os.walk(src_dir):
    for f in files:
        total_cnt += 1
        src_fp = os.path.join(subdir, f)
        try:
            h = hashfunc(Image.open(src_fp))
        except IOError:
            if src_fp not in duplicate_img:
                duplicate_img[src_fp] = 0
            continue
        if h in existing_img:
            # print ("{} already exists as {}".format(src_fp, existing_img[h]))
            duplicate_img[src_fp] = h
            if existing_img[h] not in duplicate_img:
                duplicate_img[existing_img[h]] = h
        existing_img[h] = src_fp

for fn in duplicate_img:
    print fn
