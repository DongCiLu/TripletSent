# Create label file from the dataset

import os

# dataset_dir = "../datasets/sentibank_flickr/preprocessed_256"
dataset_dir = "../datasets/google/preprocessed"
outlist = open("label_list.txt", 'w')

buffer_list = []
noun_list = {}
for subdir, dirs, files in os.walk(dataset_dir):
    segs = subdir.split('/')
    if len(segs) < 5:
        print "head"
        continue
    
    buffer_list.append(segs[-1])

    segs = segs[-1].split('_')
    noun = segs[1]
    adj = segs[0]
    if noun not in noun_list:
        noun_list[noun] = []
    noun_list[noun].append(adj)

for noun in noun_list:
    if len(noun_list[noun]) == 1:
        print("Single ANP noun: {} {}".format(noun_list[noun][0], noun))

for ANP in sorted(buffer_list):
    outlist.write("{}\n".format(ANP))
outlist.close()
