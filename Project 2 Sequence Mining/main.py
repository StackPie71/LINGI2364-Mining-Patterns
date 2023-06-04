from prefixspan_nils import *
from q2 import *
from q3_ok import *
from q4 import *
from q4_IG import *
from utility import *

# import numpy as np
# import time

# positive, negative = open_file("Datasets/Reuters/earn - Copy.txt", "Datasets/Reuters/acq - Copy.txt")
positive, negative = open_file("Datasets/Protein/SRC1521.txt", "Datasets/Protein/PKA_group15.txt")
# positive, negative = open_file("Datasets/Test/positive.txt", "Datasets/Test/negative.txt")
itemset, seq_neg, seq_pos = get_seq_itemset2(negative, positive)
k = 20

# dico = prefixSpan3(seq_neg, seq_pos, itemset, 6)
q1_print(positive, negative, k)
q2_print(positive, negative, k)
q3_print_nils(positive, negative, k)
q4_print(positive, negative, k)
# q4_IG_print(positive, negative, 6)
# false, unfound = test_sol("Datasets/Sols/test_infogain_6.txt", "q4_IG.txt")
# print("Fasle:", false)
# print("Unfound:", unfound)
