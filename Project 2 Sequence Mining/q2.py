import sys
import time
import numpy as np


# Utility functions
# ==================================================================================
def open_file(path1, path2):
    """ This function permit to open txt files

    Arguments :
        * path1 : The path to go the positive file
        * path2 : The path to go the negative file

    """
    po = open(path1, "r")
    positive = po.readlines()  # .read()

    ne = open(path2, "r")
    negative = ne.readlines()  # .read()

    return positive, negative


def get_seq_itemset2(data1, data2):
    seq_data1 = []
    seq_data2 = []
    tmp1 = []
    tmp2 = []
    itemset = []
    # seq_data1
    for line in data1:
        if (len(line) <= 1) or (line[0] == ' '):
            if len(tmp1) != 0:
                seq_data1.append(tmp1)
                tmp1 = []
        else:
            tmp1.append(line[0])
            if line[0] not in itemset:
                itemset.append(line[0])
    # seq_data2
    for line in data2:
        if (len(line) <= 1) or (line[0] == ' '):
            if len(tmp2) != 0:
                seq_data2.append(tmp2)
                tmp2 = []
        else:
            tmp2.append(line[0])
            if line[0] not in itemset:
                itemset.append(line[0])

    itemset = sorted(itemset)
    return itemset, seq_data1, seq_data2


def to_print(seq):
    """
    Convert a list of characters to a string
    Args:
        seq: ['a', 'b', 'c']

    Returns:
        string: [a, b, c]

    """
    string = '['
    for char in seq:
        string += char
        string += ', '
    string = string[:-2]
    string += ']'
    return string


def to_string(seq):
    string = ''
    for item in seq:
        string += item
    return string


def prefixSpan_Recursive2(dataset1, dataset2, itemset, dico, branch, P, N, min_diff):
    # Recursion
    for item in itemset:
        branch_copy = branch.copy()
        branch_copy.append(item)
        sequence = to_string(branch_copy)

        # Positions of the "mother" sequence
        start_pos = dico[sequence[:-1]][0]
        # print(start_pos)
        x1_start = start_pos[0]
        y1_start = start_pos[1]
        x2_start = start_pos[2]
        y2_start = start_pos[3]

        # Positions and frequency of the new sequence
        x1_pos = []
        y1_pos = []
        x2_pos = []
        y2_pos = []
        freq1 = 0
        freq2 = 0
        # print(x1_start, y1_start)
        # Occurences of the sequence in the dataset 1
        for x in range(len(dataset1)):
            if x in x1_start:
                for y in range(y1_start[x1_start.index(x)] + 1, len(dataset1[x])):
                    if item == dataset1[x][y]:
                        x1_pos.append(x)
                        y1_pos.append(y)
                        freq1 += 1
                        break

        # Occurences of the sequence in the dataset 2
        for x in range(len(dataset2)):
            if x in x2_start:
                for y in range(y2_start[x2_start.index(x)] + 1, len(dataset2[x])):
                    if item == dataset2[x][y]:
                        x2_pos.append(x)
                        y2_pos.append(y)
                        freq2 += 1
                        break

        if (freq1 > 0) or (freq2 > 0):
            # We look if the number of occurences of the sequence is in the k-best frequencies
            k_best = dico['k_first']

            coef = (P / (P + N)) * (N / (P + N))
            Wracc = coef * ((freq1 / P) - (freq2 / N))
            Wracc = round(Wracc, 5)
            freq = Wracc
            # freq = freq1 + freq2

            born = 1
            if min_diff != {}:
                born = min_diff[min(min_diff.keys())]

            # We continue to dig only if the actual sequence is frequent enough
            if freq in k_best:
                dico[sequence] = [[x1_pos, y1_pos, x2_pos, y2_pos], [freq1, freq2, freq]]
                dico = prefixSpan_Recursive2(dataset1, dataset2, itemset, dico, branch_copy, P, N, min_diff)
            elif freq > k_best[0]:
                if k_best[0] == -1 or len(min_diff) < len(k_best):
                    # We create our array of minimal born
                    min_diff[freq] = freq1 - freq2
                else:
                    # If our array is correctly created, we change the minimal born
                    min_diff.pop(k_best[0])
                    min_diff[freq] = freq1 - freq2
                # Update ou k_best
                k_best[0] = freq
                # Sorted k_best and min_diff
                k_best.sort()
                dico['k_first'] = k_best
                # for k in sorted(min_diff.keys()):
                #    min_diff[k]= min_diff[k] 

                dico[sequence] = [[x1_pos, y1_pos, x2_pos, y2_pos], [freq1, freq2, freq]]
                dico = prefixSpan_Recursive2(dataset1, dataset2, itemset, dico, branch_copy, P, N, min_diff)

            elif (freq1 >= born) and (freq2 != 0):
                dico[sequence] = [[x1_pos, y1_pos, x2_pos, y2_pos], [freq1, freq2, freq]]
                dico = prefixSpan_Recursive2(dataset1, dataset2, itemset, dico, branch_copy, P, N, min_diff)
            # else :
            #    print(k_best, min_diff)
            #    print("min_diff =", born)
            #    print("Branch =", branch_copy, " freq1 =", freq1, " freq 2 =", freq2, " Wracc = ", Wracc )

    return dico


def prefixSpan2(dataset1, dataset2, itemset, k):
    # Initialisation
    solution = {}
    x1_init = []
    x2_init = []
    P = len(dataset1)
    N = len(dataset2)
    for i in range(P):
        x1_init.append(i)
    for i in range(N):
        x2_init.append(i)
    y1_init = [-1] * P
    y2_init = [-1] * N
    # print(x1_init, y1_init)
    k_best = [-1] * k  # Min freq = -1
    dico = {'': [[x1_init, y1_init, x2_init, y2_init], [0, 0, 0]], 'k_first': k_best}
    branch = []
    diff_k_best = {}  # [1]*k

    dico = prefixSpan_Recursive2(dataset1, dataset2, itemset, dico, branch, P, N, diff_k_best)
    dico.pop('')
    k_best = dico['k_first']
    dico.pop('k_first')
    for sequence, frequency in dico.items():
        if frequency[1][2] in k_best:
            solution[sequence] = frequency[1]

    return solution


def q2_print(positive, negative, k):
    """
    Function that print the solution of prefixSpan
    :param negative: negative dataset
    :param positive: positive dataset
    :param k: k_best frequencies
    """
    itemset, seq_pos, seq_neg = get_seq_itemset2(positive, negative)
    dico = prefixSpan2(seq_neg, seq_pos, itemset, k)
    f = open('q2_large.txt', 'w')
    for seq, freq in dico.items():
        string = to_print(seq)
        string += ' '
        string += str(freq[0])
        string += ' '
        string += str(freq[1])
        string += ' '
        string += str(freq[2])
        print(string)
        f.write(string + "\n")
    f.close()
    # pass


# positive, negative = open_file("Datasets/Test/positive.txt", "Datasets/Test/negative.txt")
# # positive, negative = open_file("Datasets/Protein/SRC1521.txt", "Datasets/Protein/PKA_group15.txt")
#
# start = time.time()
# q1_print(negative, positive, 6)
# end = time.time()
# print("TIME : ", end - start)


# K = 7
# [D, Y] 174 25 0.121
# [E, Y] 190 28 0.13167
# [P, Y] 174 22 0.12295
# [Y] 314 99 0.18332
# [Y, D] 159 21 0.11176
# [Y, G] 175 22 0.12373
# [Y, V] 160 21 0.11255
# TIME ::  233.10732221603394

# TIME : 223.47623205184937
# [D, Y] 174 25 0.121
# [E, Y] 190 28 0.13167
# [P, Y] 174 22 0.12295
# [Y] 314 99 0.18332
# [Y, G] 175 22 0.12373
# [Y, V] 160 21 0.11255


# def main():
#     pos_filepath = sys.argv[1]  # filepath to positive class file
#     neg_filepath = sys.argv[2]  # filepath to negative class file
#     k = int(sys.argv[3])
#     positive, negative = open_file(pos_filepath, neg_filepath)
#     q1_print(negative, positive, k)
#
#
# #
# #
# if __name__ == "__main__":
#     main()
