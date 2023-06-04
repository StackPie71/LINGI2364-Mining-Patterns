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

def seq_database(datas):
    """ This function permit to find all items present in the database considered and to have also all the transactions

    Arguments : 
        * datas : The datas containing in a file txt aleardy open

    Return : 
        * itemset : A list containing all the items present in datas
        * seq_data : A list containing all the transactions present in datas
    """
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

def seq_to_str(seq):
    """
    """
    str = ''
    for char in seq:
        str += char
    return str

def prefixSpan1(seq_data, itemset, pat, ret, dico, k, supports):
    # TIME for prefixSpan neg =  42.77062201499939
    """ The function (recursive) which apply prefixSpan algorithm

    Arguments : 
        * seq_data : All the transaction considered
        * itemset : The itemset considered
        * pat : The pattern considered
        * ret : Ther pattern considered with its support
        * dico : A dictionnaire containing all the patterns presents in at least 1 transaction, which the support associated

    Return : 
        * dico : A dictionnaire containing all the patterns presents in at least 1 transaction, which the support associated
    """

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
            if k > 0 :
                if len(supports) == 0 :
                    supports.append(count)
                    k = k - 1

                elif count not in supports :
                    supports.append(count)
                    k = k - 1

            min_support = min(supports)
            if count >= min_support : 
                pat2 = pat.copy()
                pat2.append(item)
                tmp2 = (pat2, count)
                ret2 = ret.copy()
                ret2.append(tmp2)
                seq = []
                for tup in ret2:
                    seq.append(tup[0][0])
                dico[seq_to_str(seq)] = count
                prefixSpan1(new_seq, itemset, pat, ret2, dico, k, supports)
    return dico


def prefixSpan2(seq_data, itemset, pat, ret, dico, place, k, min_support):
    # TIME for prefixSpan neg = + 10 Mins
    """ The function (recursive) which apply prefixSpan algorithm

    Arguments : 
        * seq_data : All the transaction considered
        * itemset : The itemset considered
        * pat : The pattern considered
        * ret : Ther pattern considered with its support
        * dico : A dictionnaire containing all the patterns presents in at least 1 transaction, which the support associated

    Return : 
        * dico : A dictionnaire containing all the patterns presents in at least 1 transaction, which the support associated
    """
    for item in itemset:
        count = 0
        new_place = place.copy()
        
        for i in range(len(seq_data)): # Loop over all sequences
            if place[i] < (len(seq_data[i]) - 1) : # If we don't finish the sequence

                for j in range(place[i]+ 1, len(seq_data[i])): # We loop over pattern in a sequence from last place

                    if item == seq_data[i][j]: # If we find an pattern after last place
                        count += 1 # We increment the support
                        new_place[i] = j
                        break # We pass to the next sequence
                
                    else : 
                        new_place[i] = len(seq_data[i]) + 10                    

        if count > 0:
            if k > 0 :
                if min_support > count :
                    min_support = count
                    k = k - 1
            if count >= min_support : 
                pat2 = pat.copy()
                pat2.append(item)
                tmp2 = (pat2, count)
                ret2 = ret.copy()
                ret2.append(tmp2)
                seq = []
                for tup in ret2:
                    seq.append(tup[0][0])
                dico[seq_to_str(seq)] = count
                prefixSpan2(seq_data, itemset, pat, ret2, dico, new_place, k, min_support)
    return dico


def prefixSpanBestAncien(seq_data, itemset, pat, ret, dico, k, min_support):
    # TIME FOR NEG = 20.444960832595825
    # TIME FOR POS = 599.004688053298558
    """ The function (recursive) which apply prefixSpan algorithm

    Arguments : 
        * seq_data : All the transaction considered
        * itemset : The itemset considered
        * pat : The pattern considered
        * ret : Ther pattern considered with its support
        * dico : A dictionnaire containing all the patterns presents in at least 1 transaction, which the support associated

    Return : 
        * dico : A dictionnaire containing all the patterns presents in at least 1 transaction, which the support associated
    """

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
            if k > 0 :
                if min_support > count :
                    min_support = count
                    k = k - 1
            if count >= min_support : 
                pat2 = pat.copy()
                pat2.append(item)
                tmp2 = (pat2, count)
                ret2 = ret.copy()
                ret2.append(tmp2)
                seq = []
                for tup in ret2:
                    seq.append(tup[0][0])
                dico[seq_to_str(seq)] = count
                prefixSpanBest(new_seq, itemset, pat, ret2, dico, k, min_support)
    return dico    


def prefixSpanBest(seq_data, itemset, pat, ret, dico, k, min_support):
    # TIME for prefixSpan neg =  38.95663499832153
    #print(min_support)
    """ The function (recursive) which apply prefixSpan algorithm

    Arguments : 
        * seq_data : All the transaction considered
        * itemset : The itemset considered
        * pat : The pattern considered
        * ret : Ther pattern considered with its support
        * dico : A dictionnaire containing all the patterns presents in at least 1 transaction, which the support associated

    Return : 
        * dico : A dictionnaire containing all the patterns presents in at least 1 transaction, which the support associated
    """

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
            new_min_support = 0
            # When we find a new pattern in, at least, one transaction 
            if k >= 0 : 
                if k == 0: 
                    # If we found k best patterns, we change our min_support
                    new_min_support = count
                # If we don't find k best patterns
                new_k = k - 1
            else : 
                new_k = k
            if count >= min_support : 
                # If we have a bigger support than min_support, we continue the recursion
                pat2 = pat.copy()
                pat2.append(item)
                tmp2 = (pat2, count)
                ret2 = ret.copy()
                ret2.append(tmp2)
                seq = []
                for tup in ret2:
                    seq.append(tup[0][0])
                dico[seq_to_str(seq)] = count
                if new_min_support != 0: 
                    prefixSpanBest(new_seq, itemset, pat, ret2, dico, new_k, new_min_support)
                else : 
                    prefixSpanBest(new_seq, itemset, pat, ret2, dico, new_k, min_support)
    return dico  


def q1(itemset_neg, itemset_pos, neg, pos, k):
    """ This function give the results for the point 1

    Arguments :
        * itemset_neg : List of itemsets present in the negative file
        * itemset_pos : List of itemsets present in the positive file
        * neg : List of all the transactions present in the negative file
        * pos : List of all the transactions present in the positive file
        * k : The given k

    Return : 
        / (But print the patterns for the k best sum of support) 
    """
    # We obtains the dictionnaries contains all patterns and support found with prefixSpan algorithm
    
    sum_support_max = len(neg) + len(pos)
    supports = []
    supports.append(sum_support_max)

    #print(min_support)
    print("GO prefix span")

    start5 = time.time()
    dic_neg = prefixSpanBest(neg, itemset_neg, [], [], {}, k, 1)
    #dic_neg = prefixSpan1(neg, itemset_neg, [], [], {}, k, [])
    #dic_neg = prefixSpan2(neg, itemset_neg, [], [], {},np.ones(len(neg), dtype=int)*-1, k, min_support) 
    end5 = time.time()
    time5 = end5 - start5
    print("TIME for prefixSpan neg = ", time5)
    #print("DIC NEG = ", dic_neg)

    #DIC NEG =  {'A': 3, 'AA': 2, 'AAB': 1, 'AB': 2, 'ABA': 1, 'ABAB': 1, 'ABB': 2, 'ABC': 1, 'ABCA': 1, 
    # 'ABCAB': 1, 'ABCB': 1, 'AC': 1, 'ACA': 1, 'ACAB': 1, 'ACB': 1, 'ACBA': 1, 'ACBAB': 1, 'ACBB': 1, 'ACBC': 1, 
    # 'ACBCA': 1, 'ACBCAB': 1, 'ACBCB': 1, 'ACC': 1, 'ACCA': 1, 'ACCAB': 1, 'ACCB': 1, 'B': 3, 'BA': 3, 'BAA': 1, 
    # 'BAB': 2, 'BABB': 1, 'BB': 3, 'BBA': 2, 'BBAA': 1, 'BBAB': 1, 'BBABB': 1, 'BBB': 1, 'BBBB': 1, 'BC': 1, 'BCA': 1,
    # 'BCAB': 1, 'BCB': 1, 'C': 3, 'CA': 3, 'CAA': 1, 'CAB': 2, 'CABB': 1, 'CB': 3, 'CBA': 3, 'CBAA': 1, 'CBAB': 2, 'CBABB': 1, 
    # 'CBB': 3, 'CBBA': 2, 'CBBAA': 1, 'CBBAB': 1, 'CBBABB': 1, 'CBBB': 1, 'CBBBB': 1, 'CBC': 1, 'CBCA': 1, 'CBCAB': 1, 
    # 'CBCB': 1, 'CC': 2, 'CCA': 2, 'CCAA': 1, 'CCAB': 1, 'CCB': 2, 'CCBA': 1, 'CCBAA': 1, 'CCBB': 1, 'CCBBA': 1, 'CCBBAA': 1}

    start6 = time.time()
    dic_pos = prefixSpanBest(pos, itemset_pos, [], [], {}, k, 1)
    #dic_pos = prefixSpan1(pos, itemset_pos, [], [], {}, k, [])
    #dic_pos = prefixSpan2(pos, itemset_pos, [], [], {},np.ones(len(pos), dtype=int)*-1, k, min_support)
    end6 = time.time()
    time6 = end6 - start6
    print("TIME for prefixSpan pos = ", time6)
    #print("DIC POS = ", dic_pos)

    #DIC POS =  {'A': 4, 'AA': 4, 'AAA': 3, 'AAB': 2, 'AABA': 1, 'AABC': 1, 'AABCA': 1, 'AABCC': 1, 'AABCCA': 1, 
    # 'AAC': 3, 'AACA': 3, 'AACC': 1, 'AACCA': 1, 'AB': 3, 'ABA': 3, 'ABAA': 1, 'ABAB': 1, 'ABAC': 1, 'ABACA': 1, 
    # 'ABB': 2, 'ABBA': 1, 'ABBAA': 1, 'ABBAC': 1, 'ABBACA': 1, 'ABBC': 1, 'ABBCA': 1, 'ABC': 2, 'ABCA': 2, 'ABCC': 1,
    #  'ABCCA': 1, 'AC': 4, 'ACA': 4, 'ACAA': 1, 'ACAB': 1, 'ACAC': 1, 'ACACA': 1, 'ACB': 1, 'ACBA': 1, 'ACBAB': 1, 'ACBB': 1,
    #  'ACC': 3, 'ACCA': 3, 'ACCAA': 1, 'ACCAB': 1, 'ACCAC': 1, 'ACCACA': 1, 'ACCB': 1, 'ACCBA': 1, 'ACCBAB': 1, 'ACCBB': 1,
    #  'ACCC': 1, 'ACCCA': 1, 'B': 3, 'BA': 3, 'BAA': 1, 'BAB': 1, 'BAC': 1, 'BACA': 1, 'BB': 2, 'BBA': 1, 'BBAA': 1, 
    # 'BBAC': 1, 'BBACA': 1, 'BBC': 1, 'BBCA': 1, 'BC': 2, 'BCA': 2, 'BCC': 1, 'BCCA': 1, 'C': 4, 'CA': 4, 'CAA': 1, 'CAB': 1,
    #  'CAC': 1, 'CACA': 1, 'CB': 1, 'CBA': 1, 'CBAB': 1, 'CBB': 1, 'CC': 3, 'CCA': 3, 'CCAA': 1, 'CCAB': 1, 'CCAC': 1, 
    # 'CCACA': 1, 'CCB': 1, 'CCBA': 1, 'CCBAB': 1, 'CCBB': 1, 'CCC': 1, 'CCCA': 1}

    
    
    #print(dic_neg)
    #print(dic_pos)
    neg_seq = list(dic_neg) # Patterns present in negative file
    neg_freq = list(dic_neg.values()) # Supports for patterns in negative file

    pos_seq = list(dic_pos) # Patterns present in positive file
    pos_freq = list(dic_pos.values()) # Supports for patterns in positive file

    tab_pos_neg = []
    max_sup = 0
    tmp_j =  np.ones(len(neg_seq), dtype=int) * 1

    for i in range(len(pos_seq)):
        tmp_iii = 1
        for j in range(len(neg_seq)):
            if pos_seq[i] == neg_seq[j]:
                # If we have a patterns in positive and negative file
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
            # If we don't find any corresponding negative patterns for the positive patterns i
            tmp = []
            summ = pos_freq[i] 
            tmp.append(pos_seq[i])
            tmp.append(pos_freq[i])
            tmp.append(0)
            tmp.append(summ)
            tab_pos_neg.append(tmp)


    for j in range(len(tmp_j)) : 
        # We had nagative patterns that are not present in positive file
        tmp = []
        if tmp_j[j] == 1 : 
            summ = neg_freq[j] 
            tmp.append(neg_seq[j])
            tmp.append(0)
            tmp.append(neg_freq[j])
            tmp.append(summ)
            tab_pos_neg.append(tmp)
    
    f = open('q1.txt','w')
    while k > 0:
        for i in range(len(tab_pos_neg)):
            if max_sup == tab_pos_neg[i][3]:
                s = str(to_print(tab_pos_neg[i][0])) + ' ' +str(tab_pos_neg[i][1])+ ' ' +str(tab_pos_neg[i][2])+' ' + str(tab_pos_neg[i][3])
                print(s)
                f.write(s+"\n")
        k -= 1
        max_sup -= 1
    f.close()

def to_print(seq):
    """ This function permit to convert a list of characters to a string
    Args:
        * seq : list of characters (ex : seq = ['a', 'b', 'c'])
        
    Returns:
        * to_print : a string (ex : to_print = [a, b, c])

    """
    to_print = '['
    for char in seq:
        to_print += char
        to_print += ', '
    to_print = to_print[:-2]
    to_print += ']'
    return to_print




# """
# Run Part
# """
#

#start1 = time.time()
#positive, negative = open_file("Datasets/Test/positive.txt", "Datasets/Test/negative.txt") # Small
negative, positive = open_file("Datasets/Protein/PKA_group15.txt", "Datasets/Protein/SRC1521.txt") # Medium
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

#start4 = time.time()
q1(itemset_neg, itemset_pos, seq_neg, seq_pos, k)
#end4 = time.time()
#time4 = end4 - start4
#print("Time for function q1 = ", time4)


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
