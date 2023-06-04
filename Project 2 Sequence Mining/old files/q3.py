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

def prefixSpan(seq_data, itemset, pat, ret, dico):
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

def q3(itemset_neg, itemset_pos, neg, pos, k):
    """ This function give the results for the point 2

    Arguments :
        * itemset_neg : List of itemsets present in the negative file
        * itemset_pos : List of itemsets present in the positive file
        * neg : List of all the transactions present in the negative file
        * pos : List of all the transactions present in the positive file
        * k : The given k

    Return : 
        / (But print the patterns for the k best Wracc) 
    """
    
    # We obtains the dictionnaries contains all patterns and support found with prefixSpan algorithm
    dic_neg = prefixSpan(neg, itemset_neg, [], [], {})
    dic_pos = prefixSpan(pos, itemset_pos, [], [], {})    

    neg_seq = list(dic_neg) # Patterns present in negative file
    neg_freq = list(dic_neg.values()) # Supports for patterns in negative file

    pos_seq = list(dic_pos) # Patterns present in positive file
    pos_freq = list(dic_pos.values()) # Supports for patterns in positive file

    tab_pos_neg = []
    # Calcul of the first term for Wracc score
    P = len(pos)
    N = len(neg)
    coef = (P/(P+N)) * (N/(P+N))
    tmp_j =  np.ones(len(neg_seq), dtype=int) * 1
    Wracc_vector = []

    for i in range(len(pos_seq)):
        tmp_iii = 1
        for j in range(len(neg_seq)):
            if pos_seq[i] == neg_seq[j]:
                # If we have a patterns in positive and negative file
                tmp = []
                Wracc = coef * ((pos_freq[i]/P)-(neg_freq[j]/N))
                Wracc = round(Wracc, 5)
                if Wracc not in Wracc_vector :
                    Wracc_vector.append(Wracc)
                tmp.append(pos_seq[i])
                tmp.append(pos_freq[i])
                tmp.append(neg_freq[j])
                tmp.append(Wracc)
                tab_pos_neg.append(tmp)
                tmp_iii = 0
                tmp_j[j] = 0
                break 

        if tmp_iii == 1 : 
            # If we don't find any corresponding negative patterns for the positive patterns i
            tmp = []
            Wracc = coef * ((pos_freq[i]/P)-(0/N))
            Wracc = round(Wracc, 5)
            if Wracc not in Wracc_vector :
                Wracc_vector.append(Wracc)
            tmp.append(pos_seq[i])
            tmp.append(pos_freq[i])
            tmp.append(0)
            tmp.append(Wracc)
            tab_pos_neg.append(tmp)


    for j in range(len(tmp_j)) : 
        # We had nagative patterns that are not present in positive file
        tmp = []
        if tmp_j[j] == 1 : 
            Wracc = coef * ((0/P)-(neg_freq[j]/N))
            Wracc = round(Wracc, 5)
            if Wracc not in Wracc_vector :
                Wracc_vector.append(Wracc)
            tmp.append(neg_seq[j])
            tmp.append(0)
            tmp.append(neg_freq[j])
            tmp.append(Wracc)
            tab_pos_neg.append(tmp)

    Wracc_vector = sorted(Wracc_vector)
    new_tab = []

    #f = open('q2.txt','w')
    tmp = len(Wracc_vector) - 1
    while k > 0:
        # Loop to have the k best
        max_sup = Wracc_vector[tmp]
        for i in range(len(tab_pos_neg)):
            if max_sup == tab_pos_neg[i][3]:
                new_tab.append(tab_pos_neg[i])
                #s = str(to_print(tab_pos_neg[i][0])) + ' ' +str(tab_pos_neg[i][1])+ ' ' +str(tab_pos_neg[i][2])+' ' + str(tab_pos_neg[i][3])
                #print(s)
                #f.write(s+"\n")
        k -= 1
        tmp -= 1
        if tmp < 0 :
            break
##########################
##########################
    ## En fait ici je voulais, une fois qu'on a tous les patterns a print, 
    ## on garde juste ceux qui sont closed 
    ## -> mais je me demande si c'est Ok de faire comme ça, j'ai l'impression que c'est faut :/ 
    ## -> Peut-etre l'intégrer au prefixSpan ?
    ## J'ai du mal à voir comment faire
    ## Lien ou ils expliquent bien le closest (je trouve)
    # https://stats.stackexchange.com/questions/77465/maximal-closed-frequent-answer-included
##########################
##########################
    print(len(new_tab))
    final = []
    max_fin = 0
    
    for i in range(len(new_tab)) : 
        tmp_fin = -1
        for j in range(i+1, len(new_tab)) :
            if (new_tab[i][0] in new_tab[j][0]) and ((new_tab[i][1] == new_tab[j][1]) or (new_tab[i][2] == new_tab[j][2])) :
                if len(new_tab[i][0]) >= len(new_tab[j][0]) :
                    if max_fin < len(new_tab[i][0]) : 
                        tmp_fin = i
                else : 
                    if max_fin < len(new_tab[i][0]) : 
                        tmp_fin = j - 1
        if (tmp_fin != -1) :
            final.append(new_tab[tmp_fin])
    print(len(final))
    for i in range(len(final)) : 
        s = str(to_print(final[i][0])) + ' ' +str(final[i][1])+ ' ' +str(final[i][2])+' ' + str(final[i][3])
        print(s)

    #f.close()

def to_print(seq):
    """This function permit to convert a list of characters to a string
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

positive, negative = open_file("Datasets/Test/positive.txt", "Datasets/Test/negative.txt") # Small
#negative, positive = open_file("Datasets/Protein/PKA_group15.txt", "Datasets/Protein/SRC1521.txt") # Medium
#negative, positive = open_file("Datasets/Reuters/acq.txt", "Datasets/Reuters/earn.txt") # Big

itemset_neg, seq_neg = seq_database(negative)
itemset_pos, seq_pos = seq_database(positive)
k = 6

#start = time.time()
q3(itemset_neg, itemset_pos, seq_neg, seq_pos, k)
#end = time.time()
#time = end - start
#print("Time for function q2 = ", time)

"""
def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])
    
    positive, negative = open_file(pos_filepath, neg_filepath)
    itemset_neg, seq_neg = seq_database(negative)
    itemset_pos, seq_pos = seq_database(positive)

    q3(itemset_neg, itemset_pos, seq_neg, seq_pos, k)


if __name__ == "__main__":
    main()
"""