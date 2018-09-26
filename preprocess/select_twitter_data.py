# Select qualified images from twitter dataset
import os
import re
import sys
from shutil import copyfile

dataset_dir = "/local_scratch/Datasets/twitterpic_data/image/all"
dest_dir = "/local_scratch/Datasets/twitterpic_data/image/selected"
label_file = open("label_list_256.txt", 'r')
labels = {}

for ANP in label_file:
    segs = re.split("_|\n", ANP)
    reformed_ANP = segs[0] + ' ' + segs[1]
    ANP_fn = segs[0] + '_' + segs[1]
    labels[reformed_ANP] = 0
    ANP_path = os.path.join(dest_dir, ANP_fn)
    os.mkdir(ANP_path)

cnt = 0
for subdir, dirs, files in os.walk(dataset_dir):
    print("select qualified files from {}".format(subdir))
    for f in files:
        segs = f.split('.')
        if segs[1] == 'txt':
            with open(os.path.join(subdir, f), 'r') as description:
                pre = ''
                hit_word = ''
                hit_count = 0
                for line in description:
                    for word in line.split():
                    # for word in re.split('[^a-zA-Z]', line):
                        if pre != '':
                            ANP = pre + ' ' + word
                            if ANP in labels:
                                if hit_word != ANP:
                                    hit_word = ANP
                                    hit_count += 1
                        pre = word

                if hit_count == 1:
                    labels[hit_word] += 1
                    ANP_dir = os.path.join(dest_dir, hit_word.replace(' ', '_'))
                    img_fn = segs[0] + '.jpg'
                    src_fp = os.path.join(subdir, img_fn)
                    dst_fp = os.path.join(ANP_dir, img_fn)
                    copyfile(src_fp, dst_fp)
                    # print("copying {} to {}".format(src_fp, dst_fp))

        cnt += 1
        if cnt % 1000 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
print (" ")

for ANP in labels:
    print("Found {} images for {}".format(labels[ANP], ANP))
