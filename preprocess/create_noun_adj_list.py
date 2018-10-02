# create look up tables for noun and adj from label file
import pickle

# NUM_ANP = 910
NUM_ANP = 351

# label_file = 'label_list_256.txt'
label_file = 'label_list_filtered.txt'
label_file_noun = 'label_list_noun_filtered.txt'
label_file_adj = 'label_list_adj_filtered.txt'

noun_dict = {}
adj_dict = {}
with open(label_file) as f:
    label_index = 0
    for ANP in f:
        print ANP
        segs = ANP.split('_')
        adj = segs[0]
        noun = segs[1].split('\n')[0]

        if noun not in noun_dict:
            noun_dict[noun] = {}
        noun_dict[noun][label_index] = 0

        if adj not in adj_dict:
            adj_dict[adj] = {}
        adj_dict[adj][label_index] = 0

        label_index += 1

print("There are {} noun and {} adj".format(len(noun_dict), len(adj_dict)))

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

print(noun_list[-1])
print("**************************************")
print(adj_list[-1])

pf = open('noun_adj_list.p', 'wb')
pickle.dump(noun_list, pf)
pickle.dump(adj_list, pf)
pf.close()
