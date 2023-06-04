import sys


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


def get_itemset(datas):
    """ This function permit to obtain all items present in the data file.

    Argmuents :
        * datas : All the datas from the file considered

    Return :
        * itemset : The set of all items

    """
    itemset = []
    for line in datas:
        # index = -1
        if len(line) > 1:
            if line[0] not in itemset:
                itemset.append(line[0])
    itemset = sorted(itemset)
    return itemset


def to_print(seq):
    """
    Convert a list of characters to a string
    Args:
        seq: ['a', 'b', 'c']

    Returns:
        string: abc

    """
    string = '['
    for char in seq:
        string += char
        string += ', '
    string = string[:-2]
    string += ']'
    return string


def seq_database(datas):
    """

    Args:
        datas:

    Returns:

    """
    seq_data = []
    tmp = []
    for line in datas:
        if (len(line) <= 1) or (line[0] == ' '):
            if len(tmp) != 0:
                seq_data.append(tmp)
                tmp = []
        else:
            tmp.append(line[0])
    return seq_data


def to_string(seq):
    string = ''
    for item in seq:
        string += item
    return string


def init_dico(dataset, itemset):
    """
    Create the intial dictionary for the call of prefixSpan
    Args:
        dataset: seq_database
        itemset: itemset of the dataset
    Returns:
        dico: the initial dictionary for prefixSpan
    """
    dico = {}
    for item in itemset:
        x_pos = []
        y_pos = []
        frequency_tmp = 0
        for x in range(len(dataset)):
            for y in range(len(dataset[x])):
                if dataset[x][y] == item:
                    x_pos.append(x)
                    y_pos.append(y)
                    frequency_tmp += 1
                    break
        dico[item] = [[x_pos, y_pos], frequency_tmp]
    return dico


# ==================================================================================

def prefixSpan(dataset, itemset, dico, branch=None):
    if branch is None:
        branch = []
    for item in itemset:
        # If the sequence is empty, then we start with the next item
        if not branch:
            branch = [item]

        branch.append(item)
        sequence = to_string(branch)
        print(sequence)
        # Positions of the "mother" sequence
        start_pos = dico[sequence[:-1]][0]
        x_start = start_pos[0]
        y_start = start_pos[1]
        # Positions of the new sequence
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
        if frequency_tmp != 0:
            dico[sequence] = [[x_pos, y_pos], frequency_tmp]
            print("Ajout d'une nouvelle séquence:", sequence)
            print(dico)
            dico = prefixSpan(dataset, itemset, dico, branch)
        branch = branch[:-1]

    return dico
