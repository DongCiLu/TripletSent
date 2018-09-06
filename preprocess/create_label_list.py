import os

dataset_dir = "../datasets/sentibank_flickr/preprocessed"
outlist = open("label_list.txt", 'w')

buffer_list = []
for subdir, dirs, files in os.walk(dataset_dir):
    segs = subdir.split('/')
    if len(segs) < 5:
        print "head"
        continue

    buffer_list.append(segs[-1])

for ANP in sorted(buffer_list):
    outlist.write("{}\n".format(ANP))
outlist.close()
