import pickle

NUM_ANP = 914

label_file = 'label_list_{}.txt'.format(NUM_ANP)

noun_dict = {}
adj_dict = {}
with open(label_file) as f:
    label_index = 0
    for ANP in f:
        segs = ANP.split('_')
        adj = segs[0]
        noun = segs[1]

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

for noun in noun_dict:
    same_noun_id = noun_dict[noun]
    for idx in same_noun_id:
        noun_list[idx] = same_noun_id

for adj in adj_dict:
    same_adj_id = adj_dict[adj]
    for idx in same_adj_id:
        adj_list[idx] = same_adj_id


print(noun_list[-1])
print("**************************************")
print(adj_list[-1])

pf = open('noun_adj_list.p', 'wb')
pickle.dump(noun_list, pf)
pickle.dump(adj_list, pf)
pf.close()
