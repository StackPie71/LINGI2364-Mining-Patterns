import sys
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


def sort_seq(all_seq):
    sorted_seqs = []
    longest = []
    for seq in all_seq:
        if len(seq) not in longest:
            longest.append(len(seq))
    longest.sort(reverse=True)
    for length in longest:
        tmp = []
        for seq in all_seq:
            if len(seq) == length:
                tmp.append(seq)
        tmp = sorted(tmp)
        for to_add in tmp:
            sorted_seqs.append(to_add)

    return sorted_seqs


def is_in_seq(seq1, seq2):
    # Input : strings !
    # Return True if all element of the the seq1 is in seq2
    list1 = list(seq1)
    list2 = list(seq2)
    counter = 0
    for char in list1:
        for y in range(len(list2)):
            if char == list2[y]:
                list2 = list2[y + 1:]
                counter += 1
                break
    if counter == len(list1):
        return True
    else:
        return False


def entropy(x):
    result = 0
    if x != 1 and x != 0:
        term1 = -x * np.log2(x)
        term2 = (1 - x) * np.log2(1 - x)
        result = term1 - term2
    return result


def information_gain(P, N, p, n):
    result = 0
    term1 = entropy(P / (P + N))
    term2 = ((p + n) / (P + N)) * entropy(p / (p + n))
    if (P + N - p - n) == 0:
        return 0
    else:
        term3 = ((P + N - p - n) / (P + N)) * entropy((P - p) / (P + N - p - n))
    result = term1 - term2 - term3
    # print(result)
    return result


# Algorithms
# ==================================================================================

def wracc_IG_recursive(dataset1, dataset2, itemset, dico, branch, P, N, min_diff):
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

            # coef = (P / (P + N)) * (N / (P + N))
            Wracc = information_gain(P, N, freq1, freq2)
            Wracc = round(Wracc, 5)
            freq = Wracc
            # print(sequence, freq)
            # freq = freq1 + freq2

            born = 1
            if min_diff != {}:
                born = min_diff[min(min_diff.keys())]

            # We continue to dig only if the actual sequence is frequent enough
            if freq in k_best:
                dico[sequence] = [[x1_pos, y1_pos, x2_pos, y2_pos], [freq1, freq2, freq]]
                dico = wracc_IG_recursive(dataset1, dataset2, itemset, dico, branch_copy, P, N, min_diff)
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
                dico = wracc_IG_recursive(dataset1, dataset2, itemset, dico, branch_copy, P, N, min_diff)

            elif (freq1 >= born) and (freq2 != 0):
                dico[sequence] = [[x1_pos, y1_pos, x2_pos, y2_pos], [freq1, freq2, freq]]
                dico = wracc_IG_recursive(dataset1, dataset2, itemset, dico, branch_copy, P, N, min_diff)
            # else :
            #    print(k_best, min_diff)
            #    print("min_diff =", born)
            #    print("Branch =", branch_copy, " freq1 =", freq1, " freq 2 =", freq2, " Wracc = ", Wracc )

    return dico


def wracc_IG(dataset1, dataset2, itemset, k):
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
    k_best = [0] * k  # Min freq = -1
    dico = {'': [[x1_init, y1_init, x2_init, y2_init], [0, 0, 0]], 'k_first': k_best}
    branch = []
    diff_k_best = {}

    dico = wracc_IG_recursive(dataset1, dataset2, itemset, dico, branch, P, N, diff_k_best)
    dico.pop('')
    k_best = dico['k_first']
    dico.pop('k_first')
    for sequence, frequency in dico.items():
        if frequency[1][2] in k_best:
            solution[sequence] = frequency[1]

    return solution


def keep_closed(dico):
    ordered_seq = sort_seq(dico.keys())
    for big_seq in ordered_seq:
        dico_keys = list(dico.keys())
        for seq in dico_keys:
            if len(seq) < len(big_seq):
                if is_in_seq(seq, big_seq) and dico[seq][0] == dico[big_seq][0] and dico[seq][1] == dico[big_seq][1]:
                    dico.pop(seq)
                    ordered_seq.remove(seq)

    return dico


#

# Print Solutions
# ==================================================================================

def q4_IG_print(positive, negative, k):
    """
    Function that print the solution of prefixSpan
    :param negative: negative dataset
    :param positive: positive dataset
    :param k: k_best frequencies
    """
    itemset, seq_pos, seq_neg = get_seq_itemset2(positive, negative)
    dico = wracc_IG(seq_pos, seq_neg, itemset, k)
    dico = keep_closed(dico)
    f = open('q4_IG_large.txt', 'w')
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


def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])
    positive, negative = open_file(pos_filepath, neg_filepath)
    q4_IG_print(positive, negative, k)


if __name__ == "__main__":
    main()
