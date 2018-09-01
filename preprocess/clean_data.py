import os
import sys
import math
import operator
from PIL import Image
from shutil import copyfile

dataset_dir = "../datasets/sentibank_flickr/image"
# dataset_dir = "../datasets/sentibank_flickr/test"
dst_base_dir = "../datasets/sentibank_flickr/preprocessed"
invalid_img = "../datasets/sentibank_flickr/invalid.jpg"
predef_size = 128
ratio_threshold = 2

def compare(h1, h2):
    rms = math.sqrt(reduce(operator.add, 
        map(lambda a,b: (a-b)**2, h1, h2))/len(h1))
    return rms

iim = Image.open(invalid_img)
iih = iim.histogram()
invalid_count = 0
img_count = 0
for subdir, dirs, files in os.walk(dataset_dir):
    invalid_count_dir = 0
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
        # same to the invalid image
        ih = im.histogram()
        if len(iih) == len(ih):
            rms = compare(iih, ih)
            if rms < 1:
                invalid_count += 1
                invalid_count_dir += 1
                continue

        # create new image for the valid file
        dst_fp = os.path.join(dst_dir, f)
        # copyfile(fp, dst_fp)
        new_size = min(width, height)
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        im = im.crop((left, top, right, bottom))
        im = im.resize((predef_size, predef_size), Image.ANTIALIAS)
        im.save(dst_fp)

    sys.stdout.write("{}: {} out of {} images are invalid!\n".format(
        ANP, invalid_count_dir, img_count_dir))
    sys.stdout.flush()

print("In total, {} out of {} images are invalid!".format(
    invalid_count, img_count))
