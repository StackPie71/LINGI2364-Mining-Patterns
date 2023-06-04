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
    f1 = open(path1, "r")
    data1 = f1.readlines()  # .read()

    f2 = open(path2, "r")
    data2 = f2.readlines()  # .read()

    return data1, data2


def get_seq_itemset(datas):
    seq_data = []
    tmp = []
    itemset = []
    for line in datas:
        if (len(line) <= 1) or (line[0] == ' '):
            if len(tmp) != 0:
                seq_data.append(tmp)
                tmp = []
        else:
            if line[1] == ' ':
                tmp.append(line[0])
                print(line[0])
                if line[0] not in itemset:
                    itemset.append(line[0])
            else:
                word = ''
                letter = line[0]
                i = 0
                while letter != ' ':
                    word += letter
                    i += 1
                    letter = line[i]
                if word not in itemset:
                    itemset.append(word)

    itemset = sorted(itemset)
    return itemset, seq_data


def get_seq_itemset2(data1, data2):
    seq_data1 = []
    seq_data2 = []
    tmp1 = []
    tmp2 = []
    itemset = []
    # seq_data1
    for line in data1:
        # If the line is empty, add the formed sequence
        if (len(line) <= 1) or (line[0] == ' '):
            if len(tmp1) != 0:
                seq_data1.append(tmp1)
                tmp1 = []
        else:
            # If we have a letter, add the letter to the itemset
            if line[1] == ' ':
                tmp1.append(line[0])
                if line[0] not in itemset:
                    itemset.append(line[0])
            else:
                word = ''
                letter = line[0]
                i = 0
                while letter != ' ':
                    word += letter
                    i += 1
                    letter = line[i]
                tmp1.append(word)
                if word not in itemset:
                    itemset.append(word)
    # seq_data2
    for line in data2:
        # If the line is empty, add the formed sequence
        if (len(line) <= 1) or (line[0] == ' '):
            if len(tmp2) != 0:
                seq_data2.append(tmp2)
                tmp2 = []
        else:
            # If we have a letter, add the letter to the itemset
            if line[1] == ' ':
                tmp2.append(line[0])
                if line[0] not in itemset:
                    itemset.append(line[0])
            else:
                word = ''
                letter = line[0]
                i = 0
                while letter != ' ':
                    word += letter
                    i += 1
                    letter = line[i]
                tmp2.append(word)
                if word not in itemset:
                    itemset.append(word)

    # itemset = sorted(itemset)
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


def string_to_list(string):
    print("String:", string)
    seq = []
    word = ''
    if len(string) == 1:
        print("one letter")
        seq.append(string)
    else:
        for letter in string:
            if letter != ' ':
                word += letter
            else:
                seq.append(word)
                word = ''
        seq.append(word)
    print("Seq:", seq)
    return seq


def to_string(seq):
    string = ''
    for item in seq:
        string += item
    return string


def to_string_word(seq):
    string = ''
    for item in seq:
        string += item
        string += ' '
    string = string[:-1]
    return string


# Algorithms
# ==================================================================================

def prefixSpan_Recursive(dataset, itemset, dico, branch):
    # Recursion
    branch_copy = branch.copy()
    for item in itemset:
        branch.append(item)
        sequence = to_string(branch)

        # Positions of the "mother" sequence
        start_pos = dico[sequence[:-1]][0]
        x_start = start_pos[0]
        y_start = start_pos[1]

        # Positions and frequency of the new sequence
        x_pos = []
        y_pos = []
        frequency_tmp = 0

        for x in range(len(dataset)):
            if x in x_start:
                for y in range(y_start[x_start.index(x)] + 1, len(dataset[x])):
                    if item == dataset[x][y]:
                        x_pos.append(x)
                        y_pos.append(y)
                        frequency_tmp += 1
                        break

        # We continue to dig only if the actual sequence is present in the dataset
        if frequency_tmp > 0:
            dico[sequence] = [[x_pos, y_pos], frequency_tmp]
            dico = prefixSpan_Recursive(dataset, itemset, dico, branch)
        branch = branch_copy.copy()

    return dico


def prefixSpan(dataset, itemset, dico=None, branch=None):
    # Initialisation
    solution = {}
    if dico is None:
        x_init = []
        for i in range(len(dataset)):
            x_init.append(i)
        y_init = [-1] * len(dataset)
        dico = {'': [[x_init, y_init], 0]}
    if branch is None:
        branch = []

    dico = prefixSpan_Recursive(dataset, itemset, dico, branch)
    for sequence, frequency in dico.items():
        solution[sequence] = frequency[1]
    solution.pop('')

    return solution


def prefixSpan_Recursive2(dataset1, dataset2, itemset, dico, branch):
    """
    Implementation of prefixSpan for letters sequences
    Args:
        dataset1: First dataset (neg or pos)
        dataset2: Second dataset (pos or neg)
        itemset: The itemset for both datasets
        dico: Dico where all the sequences are with their position in each dataset and their frequency
        branch: List which is the current sequence of interest

    Returns:
        dico : Dico where all the sequences are with their position in each dataset and their frequency.
    """
    # Recursion
    branch_copy = branch.copy()
    for item in itemset:
        branch.append(item)
        sequence = to_string(branch)

        # Positions of the "mother" sequence
        start_pos = dico[sequence[:-1]][0]
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

        # We look if the number of occurences of the sequence is in the k-best frequencies
        k_best = dico['k_first']
        k_best.sort()
        freq = freq1 + freq2
        # We continue to dig only if the actual sequence is frequent enough
        if freq > k_best[0] and freq not in k_best:
            k_best[0] = freq
            dico['k_first'] = k_best
            dico[sequence] = [[x1_pos, y1_pos, x2_pos, y2_pos], [freq1, freq2, freq]]
            dico = prefixSpan_Recursive2(dataset1, dataset2, itemset, dico, branch)
        elif freq in k_best:
            dico[sequence] = [[x1_pos, y1_pos, x2_pos, y2_pos], [freq1, freq2, freq]]
            dico = prefixSpan_Recursive2(dataset1, dataset2, itemset, dico, branch)

        branch = branch_copy.copy()

    return dico


def prefixSpan2(dataset1, dataset2, itemset, k, dico=None, branch=None):
    """
    Call of prefixSpan for letters sequences
    Args:
        dataset1: First dataset (neg or pos)
        dataset2: Second dataset (pos or neg)
        itemset: The itemset for both datasets
        k: k most frequent sequences
        dico: Empty dico where the solution will be stored in
        branch: Empty list which is the current sequence

    Returns:
        solution : a dictionary with all the sequences and their frequency in each dataset.
    """
    # Initialisation
    solution = {}
    if dico is None:
        x1_init = []
        x2_init = []
        for i in range(len(dataset1)):
            x1_init.append(i)
        for i in range(len(dataset2)):
            x2_init.append(i)
        y1_init = [-1] * len(dataset1)
        y2_init = [-1] * len(dataset2)
        k_best = [1] * k  # Min freq = 1
        dico = {'': [[x1_init, y1_init, x2_init, y2_init], [0, 0, 0]], 'k_first': k_best}
    if branch is None:
        branch = []

    dico = prefixSpan_Recursive2(dataset1, dataset2, itemset, dico, branch)
    dico.pop('')
    k_best = dico['k_first']
    dico.pop('k_first')
    for sequence, frequency in dico.items():
        if frequency[1][2] in k_best:
            solution[sequence] = frequency[1]

    return solution


def prefixSpan_Recursive3(dataset1, dataset2, itemset, dico, branch):
    """
    Implementation of prefixSpan for word sequences
    Args:
        dataset1: First dataset (neg or pos)
        dataset2: Second dataset (pos or neg)
        itemset: The itemset for both datasets
        dico: Dico where all the sequences are with their position in each dataset and their frequency
        branch: List which is the current sequence of interest

    Returns:
        dico : Dico where all the sequences are with their position in each dataset and their frequency.
    """
    # Recursion
    branch_copy = branch.copy()
    for item in itemset:
        mother_seq = to_string_word(branch)
        branch.append(item)
        sequence = to_string_word(branch)
        # Positions of the "mother" sequence
        start_pos = dico[mother_seq][0]
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

        # We look if the number of occurences of the sequence is in the k-best frequencies
        k_best = dico['k_first']
        k_best.sort()
        freq = freq1 + freq2
        # We continue to dig only if the actual sequence is frequent enough
        if freq > k_best[0] and freq not in k_best:
            k_best[0] = freq
            dico['k_first'] = k_best
            dico[sequence] = [[x1_pos, y1_pos, x2_pos, y2_pos], [freq1, freq2, freq]]
            dico = prefixSpan_Recursive3(dataset1, dataset2, itemset, dico, branch)
        elif freq in k_best:
            dico[sequence] = [[x1_pos, y1_pos, x2_pos, y2_pos], [freq1, freq2, freq]]
            dico = prefixSpan_Recursive3(dataset1, dataset2, itemset, dico, branch)

        branch = branch_copy.copy()

    return dico


def prefixSpan3(dataset1, dataset2, itemset, k, dico=None, branch=None):
    """
    Call of prefixSpan for word sequences
    Args:
        dataset1: First dataset (neg or pos)
        dataset2: Second dataset (pos or neg)
        itemset: The itemset for both datasets
        k: k most frequent sequences
        dico: Empty dico where the solution will be stored in
        branch: Empty list which is the current sequence

    Returns:
        solution : a dictionary with all the sequences and their frequency in each dataset.
    """
    # Initialisation
    solution = {}
    if dico is None:
        x1_init = []
        x2_init = []
        for i in range(len(dataset1)):
            x1_init.append(i)
        for i in range(len(dataset2)):
            x2_init.append(i)
        y1_init = [-1] * len(dataset1)
        y2_init = [-1] * len(dataset2)
        k_best = [1] * k  # Min freq = 1
        dico = {'': [[x1_init, y1_init, x2_init, y2_init], [0, 0, 0]], 'k_first': k_best}
    if branch is None:
        branch = []

    dico = prefixSpan_Recursive3(dataset1, dataset2, itemset, dico, branch)
    dico.pop('')
    k_best = dico['k_first']
    dico.pop('k_first')
    for sequence, frequency in dico.items():
        if frequency[1][2] in k_best:
            solution[sequence] = frequency[1]

    return solution


# Print Solutions
# ==================================================================================
def q1_print(positive, negative, k):
    """
    Function that print the solution of prefixSpan
    :param negative: negative dataset
    :param positive: positive dataset
    :param k: k_best frequencies
    """
    itemset, seq_pos, seq_neg = get_seq_itemset2(positive, negative)
    dico = prefixSpan2(seq_neg, seq_pos, itemset, k)
    f = open('q1_print_large.txt', 'w')
    for seq, freq in dico.items():
        # string = to_print(string_to_list(seq))
        string = to_print(list(seq))
        string += ' '
        string += str(freq[0])
        string += ' '
        string += str(freq[1])
        string += ' '
        string += str(freq[2])
        print(string)
        f.write(string + "\n")
    f.close()
    pass


def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])
    positive, negative = open_file(pos_filepath, neg_filepath)
    q1_print(positive, negative, k)


if __name__ == "__main__":
    main()
