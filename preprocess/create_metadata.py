# create look up tables for noun and adj from label file
import os
import pickle

# label_file = 'label_list_256.txt'
dataset_dir = '../datasets/google/regular/train'
label_file = 'label_list_filtered.txt'
label_file_noun = 'label_list_noun_filtered.txt'
label_file_adj = 'label_list_adj_filtered.txt'
metadata_file = 'metadata.p'

noun_dict = {}
adj_dict = {}
sample_cnt_list = []
total_cnt = 0
label_index = 0
with open(label_file) as f:
    for ANP in f:
        ANP = ANP.rstrip()
        segs = ANP.split('_')
        adj = segs[0]
        noun = segs[1]

        if noun not in noun_dict:
            noun_dict[noun] = {}
        noun_dict[noun][label_index] = 0

        if adj not in adj_dict:
            adj_dict[adj] = {}
        adj_dict[adj][label_index] = 0

        ANP_dir = os.path.join(dataset_dir, ANP)
        sample_cnt = len([name for name in os.listdir(ANP_dir)
                if os.path.isfile(os.path.join(ANP_dir, name))])
        sample_cnt_list.append(sample_cnt)
        total_cnt += sample_cnt
        print ANP, sample_cnt

        label_index += 1

NUM_ANP = label_index

print("There are {} noun and {} adj and {} samples".format(
    len(noun_dict), len(adj_dict), total_cnt))

noun_list = [{} for i in range(NUM_ANP)]
adj_list = [{} for i in range(NUM_ANP)]

nf = open(label_file_noun, 'w')
af = open(label_file_adj, 'w')

for noun in sorted(noun_dict):
    nf.write("{}\n".format(noun))
    same_noun_id = noun_dict[noun]
    for idx in same_noun_id:
        noun_list[idx] = same_noun_id

for adj in sorted(adj_dict):
    af.write("{}\n".format(adj))
    same_adj_id = adj_dict[adj]
    for idx in same_adj_id:
        adj_list[idx] = same_adj_id

nf.close()
af.close()

pf = open(metadata_file, 'wb')
pickle.dump(noun_list, pf)
pickle.dump(adj_list, pf)
pickle.dump(sample_cnt_list, pf)
pf.close()
