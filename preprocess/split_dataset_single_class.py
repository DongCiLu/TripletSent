# Split the data into train and test
import os
import argparse
import numpy as np
from shutil import copyfile

src_dir_pattern = "../datasets/google/regular/{}"
dst_dir_pattern = "../datasets/google/regular/single/{}_{}"
tfrecord_dir = "../datasets/google/regular/tfrecord/single"
labels_file = "label_list_filtered.txt"
build_script_pattern = "python build_image_data_simple.py --train_directory {} --validation_directory {} --output_directory {} --labels_file {}"
tfrecord_pattern = "ts-{}_{}_{}.tfrecord"
dataset_type = 'anp'

def generate_dir_structure(split, target_class):
    # copy the empty directory structure of all class
    src_dir = src_dir_pattern.format(split)
    dst_dir = dst_dir_pattern.format(split, target_class)
    os.mkdir(dst_dir)
    for subdir, dirs, files in os.walk(src_dir):
        if subdir == src_dir:
            continue
        segs = subdir.split('/')
        ANP_class = segs[-1]
        dst_ANP_dir = os.path.join(dst_dir, ANP_class)
        os.mkdir(dst_ANP_dir)

        # copy files in the target class
        if ANP_class == target_class:
            for f in files:
                src_fp = os.path.join(subdir, f)
                dst_fp = os.path.join(dst_ANP_dir, f)
                copyfile(src_fp, dst_fp)

def generate_tfrecord(target_class):
    # generate tfrecord using build script
    train_dir = dst_dir_pattern.format('train', target_class)
    validation_dir = dst_dir_pattern.format('test', target_class)
    build_script_command = build_script_pattern.format(
            train_dir, validation_dir, tfrecord_dir, labels_file)
    os.system(build_script_command)

    # rename tfrecord file
    src_train_tfrecord = os.path.join(tfrecord_dir, "train-00000-of-00001")
    src_test_tfrecord = os.path.join(tfrecord_dir, "validation-00000-of-00001")
    dst_train_tfrecord = os.path.join(tfrecord_dir, 
            tfrecord_pattern.format('train', dataset_type, target_class))
    dst_test_tfrecord = os.path.join(tfrecord_dir, 
            tfrecord_pattern.format('test', dataset_type, target_class))
    os.rename(src_train_tfrecord, dst_train_tfrecord)
    os.rename(src_test_tfrecord, dst_test_tfrecord)

def read_labels():
    with open(labels_file, 'r') as f:
        class_list = [line.rstrip() for line in f]
    return class_list

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('target_class', type=str)
    ap.add_argument('--gpu', type=str, default='0', required=False)
    args = ap.parse_args()
    print(args.target_class, args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.target_class == 'all':
        class_list = read_labels()
        for target_class in class_list:
            print("Processing {}".format(target_class))
            generate_dir_structure('train', target_class)
            generate_dir_structure('test', target_class)
            generate_tfrecord(target_class)
    else:
        print("Processing {}".format(args.target_class))
        generate_dir_structure('train', args.target_class)
        generate_dir_structure('test', args.target_class)
        generate_tfrecord(args.target_class)
        

