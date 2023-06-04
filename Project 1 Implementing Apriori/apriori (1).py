import pandas as pd


def read_file(filepath):
    """
    Function permit to read the file.
    Argument : 
        - filepath = Path to go to the file
    Return :
        - data = 2D array containing the value in the file
    """
    with open(filepath) as datFile:
        data = [line.split() for line in datFile]
    return data


# def generating_candidates(F) :
#     """
#     Function permit to generate the candidate at level i
#     Arugment :
#         - F = itemset from which we will generate candidates
#     Return :
#         - candidates = list of candidates
#     """
#     candidates = []
#     # Generate candidates by combining strings that are identical, except for the last symbol
#     for i in range(len(F)) :
#         strings = []
#         for j in range(i+1, len(F)) :
#             strings.append(F[i,j])
#
#         candidates.append(strings)
#
#     return candidates

def generating_candidates(F):
    """
    Function permit to generate the candidate at level i
    Arugment :
        - F = itemset from which we will generate candidates
    Return :
        - candidates = list of candidates
    """
    candidates = []


def frequence(c, dataset):
    """
    Function permit to calculate the frequency of a candidate 'c'
    Argument :
        - c = candidate
        - dataset = whole dataset
    Return : 
        - freq = frequency of the candidate c
        freq(c) = support(I)/len(dataset)
    """
    support = 0
    for transactions in dataset:
        for item in transactions:
            separator = ""
            c_str = separator.join(c)
            item_str = separator.join(item)
            if c_str in item_str:
                support += 1

    freq = support / len(dataset)

    return freq


def apriori(filepath, minFrequency):
    """
    - filepath = path to a dataset file (string)
    - minFrequency = minimum frequency that an itemset must have in order to be considered as frequent (double) 
        = support divided by the total number of transactions in the dataset
    """
    # Read data
    data = read_file(filepath)

    # Initialisation
    level = 0
    F = []
    candidates = []
    # Initialisation candidates (all itemset considered)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if (data[i][j] not in candidates):
                candidates.append(data[i][j])

    # Loop of the algorithm
    while (level == 0 or F[level - 1] != []):
        # Generate candidates
        candidates = generating_candidates(F[level - 1])

        # Test if each Candidate is frequent or not
        for c in range(len(candidates)):
            freq = frequence(candidates)
            if (freq >= minFrequency):
                F[level].append(c)

        level += 1

    return data


#########
## Run ##
#########
"""
We will use at first toy example because it is the smallest one. 
filepath = Datasets/toy.dat
minimum frequency = 0.125
itemsets that our algorithm should find = Datasets/toy_itemsets0125.txt

toy dataset = [['1', '2', '3'], ['2', '3', '4'], ['3', '4', '5'], ['2', '3'], ['3', '4'], ['1', '2', '3', '4'], ['1', '2', '4'], ['5']]
Candidates init = ['1', '2', '3', '4', '5']
"""
data = apriori("Datasets/toy.dat", 0.125)
print(data)

# toy_answer = read_file("Datasets/toy_itemsets0125.txt")
# print(toy_answer)
