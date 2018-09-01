import os

dataset_dir = "../datasets/sentibank_flickr/image"
outlist = open("label_list.txt", 'w')

for subdir, dirs, files in os.walk(dataset_dir):
    segs = subdir.split('/')
    if len(segs) < 5:
        print "head"
        continue

    outlist.write("{}\n".format(segs[-1]))
