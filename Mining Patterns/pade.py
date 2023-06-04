
import numpy as np

def open_file(path1, path2) :
    """ This function permit to open txt files

    Arguments :
        * path1 : The path to go the positive file
        * path2 : The path to go the negative file

    """
    po = open(path1, "r")
    positive = po.readlines() #.read()

    ne = open(path2, "r")
    negative = ne.readlines() #.read()

    return positive, negative

def get_itemset(datas) : 
    """ This function permit to obtain all items present in the data file. 

    Argmuents : 
        * datas : All the datas from the file considered

    Return : 
        * itemset : The set of all items

    """
    itemset = []
    for line in datas :
        #index = -1
        if len(line) > 1 :
            if line[0] not in itemset :
                itemset.append(line[0])
    itemset = sorted(itemset)
    return itemset

def get_verticalReprensentation(datas, itemset) :
    """ This function permit to have vertical representation. 

    Arguments : 
        * datas : All the datas from the file considered
        * itemset : The set of all items from the file considered

    Return : 
        * The vertical representation

    """
    vertical_rep = [] 
    i = 0
    while i < len(itemset) :
        vertical_rep.append([])
        i = i + 1

    transaction = 0

    for line in datas : 
        if len(line) <= 1 :
            transaction = transaction + 1
        else :
            for i in range(len(itemset)) :
                if itemset[i] == line[0] :
                    tmp = [transaction, int(line[2])]
                    vertical_rep[i].append(tmp)
    return vertical_rep

def prefixSpan(seq_data, itemset, pat, place) :
    count = 0
    
    for item in itemset : 
        for i in range(len(seq_data)) :
            if (place[i] == -2 ) :
                continue
            tmp = 0
            for j in range(place[i]+1,len(seq_data)) :
                if item == seq_data[i][j] : 
                    count += 1
                    place[i] = j
                    tmp = 1
            if tmp == 0 : 
                # If we don't have this item in the transaction
                place[i] = -2
            
        
        # recursion



"""
def prefixSpan(seq_data,pat, itemset, place) :
    count = 0
    ret = []

    #for pl in place : 
    for item in itemset : 
        i = place 
        for j in range(len(seq_data)) : 
            while (i <= len(seq_data[j])) :
                if item == seq_data[i][j] :
                    count += 1
                    pat.append(item)
                    prefixSpan(seq_data,pat, itemset, i)
                i += 1
            tmp = (pat, count)
            ret.append(tmp)
        
    return ret
"""


def seq_database(datas) : 
    seq_data = []
    tmp = []
    for line in datas :
        if (len(line) <= 1) or (line[0] == ' ') :
            if len(tmp) != 0 :
                seq_data.append(tmp)
                tmp = []
        else : 
            tmp.append(line[0])
    return seq_data


"""
Run Part
"""

positive, negative = open_file("Datasets/Test/positive.txt", "Datasets/Test/negative.txt")
itemset = get_itemset(negative)
print(itemset)

for line in positive :
    print(line)

seq_neg = seq_database(negative)
print(seq_neg)
seq_pos = seq_database(positive)
print(seq_pos)

q1 = prefixSpan(seq_neg,[], itemset, 0)
print(q1)



vertical_rep_negative = get_verticalReprensentation(negative, itemset)
#print("vertical_rep_negative : ", vertical_rep_negative)

vertical_rep_positive = get_verticalReprensentation(positive, itemset)
#print("vertical_rep_positive : ",vertical_rep_positive)
        