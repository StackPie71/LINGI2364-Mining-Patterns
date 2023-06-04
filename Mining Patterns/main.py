from prefixspan_nils import *
import numpy as np

positive, negative = open_file("Datasets/Test/positive.txt", "Datasets/Test/negative.txt")
itemset_neg = get_itemset(negative)
itemset_pos = get_itemset(positive)
seq_neg = seq_database(negative)
seq_pos = seq_database(positive)
k = 5

pos_neg = list(np.zeros(len(seq_neg), dtype=int))
init_dico_neg = init_dico(seq_neg, itemset_neg)
# print(seq_neg)
# print(itemset_neg)
# print(init_dico_neg)

dico = prefixSpan(seq_neg, itemset_neg, init_dico_neg)
