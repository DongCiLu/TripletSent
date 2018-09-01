import os
from shutil import move

# dataset_dir = "../datasets/sentibank_flickr/image"
dataset_dir = "../datasets/sentibank_flickr/preprocessed"
dump_dir = "../datasets/sentibank_flickr/low_nf_ANPs"
threshold = 100

high_count = 0
low_count = 0
lowest_nf = 1000
ANP_count = 0
adj_table = {}
noun_table = {}
for subdir, dirs, files in os.walk(dataset_dir):
    if dataset_dir == subdir:
        print("ignore root")
        continue

    # Get stat of ANP
    segs = subdir.split('/')
    ANP = segs[-1]
    segs = ANP.split('_')
    adj = segs[0]
    noun = segs[1]

    ANP_count += 1
    if adj not in adj_table:
        adj_table[adj] = 0
    adj_table[adj] += 1
    noun_table[noun] = 1

    # Get stat of number of images in each ANP
    nf = len([f for f in os.listdir(subdir) 
        if os.path.isfile(os.path.join(subdir, f))])

    if nf < lowest_nf:
        lowest_nf = nf

    if nf >= threshold:
        high_count += 1
    else:
        low_count += 1
        # move(subdir, dump_dir)

# Print stats
print("{} folders with more than {} files and {} folders with less than {} files.".format(high_count, threshold, low_count, threshold))

print("The smallest folder has {} files.".format(lowest_nf))

print("There is {} ANP, {} adj and {} noun.".format(ANP_count, len(adj_table), len(noun_table)))
