# Remove invalid image and normalize image size
import os
import sys
import math
import operator
from PIL import Image
from shutil import copyfile

dataset_dir = "../datasets/google/images"
dst_base_dir = "../datasets/google/preprocessed"
predef_size = 256
ratio_threshold = 2

if not os.path.exists(dst_base_dir):
    os.mkdir(dst_base_dir)
invalid_count = 0
invalid_fn_count = 0
img_count = 0
for subdir, dirs, files in os.walk(dataset_dir):
    invalid_count_dir = 0
    invalid_fn_count_dir = 0
    img_count_dir = 0
    if dataset_dir == subdir:
        continue
    segs = subdir.split('/')
    ANP = segs[-1]
    dst_dir = os.path.join(dst_base_dir, ANP)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for f in files:
        img_count += 1
        img_count_dir += 1
        fp = os.path.join(subdir, f)
        try:
            im = Image.open(fp)
            # check size and ratio
            width, height = im.size
            if width < predef_size or height < predef_size:
                invalid_count += 1
                invalid_count_dir += 1
                continue
            if float(width) / height > ratio_threshold or \
                    float(height) / width > ratio_threshold:
                invalid_count += 1
                invalid_count_dir += 1
                continue

            # create new image for the valid file
            new_fn = "{}.jpg".format(img_count_dir)
            dst_fp = os.path.join(dst_dir, new_fn)
            new_size = min(width, height)
            left = (width - new_size) / 2
            top = (height - new_size) / 2
            right = (width + new_size) / 2
            bottom = (height + new_size) / 2
            im = im.crop((left, top, right, bottom))
            im = im.resize((predef_size, predef_size), Image.ANTIALIAS)
            im.save(dst_fp)
        except IOError as e:
            invalid_fn_count += 1
            invalid_fn_count_dir += 1
            continue

    sys.stdout.write("{}: There is {} images, {} have invalid size, {} have invalid filename!\n".format(ANP, img_count_dir, invalid_count_dir, invalid_fn_count_dir))
    sys.stdout.flush()

sys.stdout.write("In total, there is {} images, {} have invalid size, {} have invalid filename!\n".format(img_count, invalid_count, invalid_fn_count))
