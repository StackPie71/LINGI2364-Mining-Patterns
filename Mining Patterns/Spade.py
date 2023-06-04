import numpy as np


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


def get_verticalReprensentation(datas, itemset):
    """ This function permit to have vertical representation. 

    Arguments : 
        * datas : All the datas from the file considered
        * itemset : The set of all items from the file considered

    Return : 
        * The vertical representation

    """
    vertical_rep = []
    i = 0
    while i < len(itemset):
        vertical_rep.append([])
        i = i + 1

    transaction = 0

    for line in datas:
        if len(line) <= 1:
            transaction = transaction + 1
        else:
            for i in range(len(itemset)):
                if itemset[i] == line[0]:
                    tmp = [transaction, int(line[2])]
                    vertical_rep[i].append(tmp)
    return vertical_rep


def seq_to_str(seq):
    str = ''
    for char in seq:
        str += char
    return str


def prefixSpan(seq_data, itemset, pat, place, ret, dico):
    # if max(place) == -2 :
    # print("RETURN")
    #    return

    for item in itemset:
        # print("ITEM : ", item)
        count = 0
        new_place = place.copy()
        # print("New place = ", new_place)
        for i in range(len(seq_data)):
            if new_place[i] == -2:
                continue
            tmp = 0
            for j in range(new_place[i] + 1, len(seq_data[i])):
                if item == seq_data[i][j]:
                    count += 1
                    new_place[i] = j
                    tmp = 1
                    break
            if tmp == 0:
                # If we don't have this item in the transaction
                new_place[i] = -2
        # print("new_place :", new_place)
        # print("place : ", place)
        # else :
        if count > 0:
            pat2 = pat.copy()
            pat2.append(item)
            tmp2 = (pat2, count)
            ret2 = ret.copy()
            ret2.append(tmp2)
            # print("ret = ", ret2)
            seq = []
            for tup in ret2:
                seq.append(tup[0][0])
            dico[seq_to_str(seq)] = count
            prefixSpan(seq_data, itemset, pat, new_place, ret2, dico)
    return dico
    # prefixSpan(seq_data, itemset, pat, new_place, ret)


def q1(itemset, neg, pos, k):
    dic_neg = prefixSpan(neg, itemset, [], [-1, -1, -1], [], {})
    dic_pos = prefixSpan(pos, itemset, [], [-1, -1, -1, -1], [], {})
    both_dic = [dic_neg, dic_pos]

    neg_seq = list(dic_neg)
    neg_freq = list(dic_neg.values())

    pos_seq = list(dic_pos)
    pos_freq = list(dic_pos.values())

    classement_neg = []
    classement_pos = []
    for val in neg_freq:
        if val not in classement_neg:
            classement_neg.append(val)
    for val in pos_freq:
        if val not in classement_pos:
            classement_pos.append(val)
    classement_pos.sort()
    classement_neg.sort()
    best_neg = classement_neg[-k:]
    best_pos = classement_pos[-k:]

    print("best_neg: ", best_neg)
    print("best_pos: ", best_pos)
    k_best_neg = []
    k_best_pos = []

    for seq, freq in list(dic_neg.items()):
        if freq in best_neg:
            k_best_neg.append(seq)
    for seq, freq in list(dic_pos.items()):
        if freq in best_pos:
            k_best_pos.append(seq)

    print('Positive: ', dic_pos)
    print(k_best_pos)
    print('Negative: ', dic_neg)
    print(k_best_neg)



def seq_database(datas):
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


"""
Run Part
"""

positive, negative = open_file("Datasets/Test/positive.txt", "Datasets/Test/negative.txt")
itemseok = get_itemset(negative)
# print(itemset)

# for line in positive :
#   print(line)

seq_neg = seq_database(negative)
seq_pos = seq_database(positive)
print(seq_neg)
# seq_pos = seq_database(positive)
# print(seq_pos)

q_test = prefixSpan(seq_neg, itemseok, [], [-1, -1, -1], [], {})
print(q_test)

vertical_rep_negative = get_verticalReprensentation(negative, itemseok)
# print("vertical_rep_negative : ", vertical_rep_negative)

vertical_rep_positive = get_verticalReprensentation(positive, itemseok)
# print("vertical_rep_positive : ",vertical_rep_positive)

q1(itemseok, seq_neg, seq_pos, 1)
