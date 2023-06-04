import numpy as np
import sys
import time

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


def seq_to_str(seq):
    """
    """
    str = ''
    for char in seq:
        str += char
    return str

def prefixSpan(seq_data, itemset, pat, ret, dico):
    """

    """
    #print("len(item()) = ", len(itemset)) # 3 - 20 - 26
    #print("len(seq_data()) = ", len(seq_data)) # 4 - 381 - 1595

    for item in itemset:
        count = 0
        new_seq = []
        for i in range(len(seq_data)): 
            if item in seq_data[i]:
                for j in range(len(seq_data[i])):
                    if item == seq_data[i][j]:
                        count += 1
                        new_seq.append(seq_data[i][j+1:])
                        break

        if count > 0:
            pat2 = pat.copy()
            pat2.append(item)
            tmp2 = (pat2, count)
            ret2 = ret.copy()
            ret2.append(tmp2)
            seq = []
            for tup in ret2:
                seq.append(tup[0][0])
            dico[seq_to_str(seq)] = count
            prefixSpan(new_seq, itemset, pat, ret2, dico)
    return dico

def prefixSpan2(seq_data, itemset, pat, place, ret, dico):
    """

    """
    #print("len(item()) = ", len(itemset)) # 3 - 20 - 26
    #print("len(seq_data()) = ", len(seq_data)) # 4 - 381 - 1595
    #print("len(place()) = ", len(place)) # 4 - 381 - 1595
    
    for item in itemset:
        count = 0
        new_place = []
        new_seq = []
        for i in range(len(seq_data)): 

            for j in range(place[i] + 1, len(seq_data[i])):
                if item == seq_data[i][j]:
                    count += 1
                    new_place.append(j)
                    new_seq.append(seq_data[i])
                    break

        if count > 0:
            pat2 = pat.copy()
            pat2.append(item)
            tmp2 = (pat2, count)
            ret2 = ret.copy()
            ret2.append(tmp2)
            seq = []
            for tup in ret2:
                seq.append(tup[0][0])
            dico[seq_to_str(seq)] = count
            prefixSpan2(new_seq, itemset, pat, new_place, ret2, dico)
    print(dico)
    return dico



def q1(itemset_neg, itemset_pos, neg, pos, k):
    """
    """
    start5 = time.time()
    dic_neg = prefixSpan(neg, itemset_neg, [], [], {})
    end5 = time.time()
    time5 = end5 - start5
    print("TIME for prefixSpan neg = ", time5)
    
    start6 = time.time()
    dic_pos = prefixSpan(pos, itemset_pos, [], [], {})
    end6 = time.time()
    time6 = end6 - start6
    print("TIME for prefixSpan pos)= ", time6)
    

    neg_seq = list(dic_neg)
    neg_freq = list(dic_neg.values())

    pos_seq = list(dic_pos)
    pos_freq = list(dic_pos.values())

    tab_pos_neg = []
    max_sup = 0
    tmp_j =  np.ones(len(neg_seq), dtype=int) * 1

    for i in range(len(pos_seq)):
        tmp_iii = 1
        for j in range(len(neg_seq)):
            if pos_seq[i] == neg_seq[j]:
                tmp = []
                summ = pos_freq[i] + neg_freq[j]
                if summ > max_sup : 
                    max_sup = summ
                tmp.append(pos_seq[i])
                tmp.append(pos_freq[i])
                tmp.append(neg_freq[j])
                tmp.append(summ)
                tab_pos_neg.append(tmp)
                tmp_iii = 0
                tmp_j[j] = 0
                break #0.0025310516357421875

        if tmp_iii == 1 : 
            tmp = []
            summ = pos_freq[i] 
            tmp.append(pos_seq[i])
            tmp.append(pos_freq[i])
            tmp.append(0)
            tmp.append(summ)
            tab_pos_neg.append(tmp)


    for j in range(len(tmp_j)) : 
        tmp = []
        if tmp_j[j] == 1 : 
            summ = neg_freq[j] 
            tmp.append(neg_seq[j])
            tmp.append(0)
            tmp.append(neg_freq[j])
            tmp.append(summ)
            tab_pos_neg.append(tmp)
    
    #f = open('q1.txt','w')
    while k > 0:
        for i in range(len(tab_pos_neg)):
            if max_sup == tab_pos_neg[i][3]:
                s = str(to_print(tab_pos_neg[i][0])) + ' ' +str(tab_pos_neg[i][1])+ ' ' +str(tab_pos_neg[i][2])+' ' + str(tab_pos_neg[i][3])
                print(s)
                #f.write(s+"\n")
        k -= 1
        max_sup -= 1
    #f.close()

def to_print(seq):
    to_print = '['
    for char in seq:
        to_print += char
        to_print += ', '
    to_print = to_print[:-2]
    to_print += ']'
    return to_print


def seq_database(datas):
    seq_data = []
    tmp = []
    itemset = []
    for line in datas:
        if (len(line) <= 1) or (line[0] == ' '):
            if len(tmp) != 0:
                seq_data.append(tmp)
                tmp = []
        else:
            tmp.append(line[0])
            if line[0] not in itemset:
                itemset.append(line[0])
    itemset = sorted(itemset)
    return itemset, seq_data


# """
# Run Part
# """
#

#start1 = time.time()
positive, negative = open_file("Datasets/Test/positive.txt", "Datasets/Test/negative.txt") # Small
# negative, positive = open_file("Datasets/Protein/PKA_group15.txt", "Datasets/Protein/SRC1521.txt") # Medium
#negative, positive = open_file("Datasets/Reuters/acq.txt", "Datasets/Reuters/earn.txt") # Big
#end1 = time.time()
#time1 = end1 - start1
#print("TIME 1 (open_file) = ", time1)

#start2 = time.time()
itemset_neg, seq_neg = seq_database(negative)
#end2 = time.time()
#time2 = end2 - start2
#print("TIME 2 (seq_database neg) = ", time2)

#start3 = time.time()
itemset_pos, seq_pos = seq_database(positive)
#end3 = time.time()
#time3 = end3 - start3
#print("TIME 3 (seq_database pos) = ", time3)

k = 6

start4 = time.time()
q1(itemset_neg, itemset_pos, seq_neg, seq_pos, k)
end4 = time.time()
time4 = end4 - start4
print("Time for function q1 = ", time4)


"""
def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])
    # TODO: read the dataset files and call your miner to print the top k itemsets
    positive, negative = open_file(pos_filepath, neg_filepath)
    itemset_neg, seq_neg = seq_database(negative)
    itemset_pos, seq_pos = seq_database(positive)

    q1(itemset_neg, itemset_pos, seq_neg, seq_pos, k)


if __name__ == "__main__":
    main()
"""
