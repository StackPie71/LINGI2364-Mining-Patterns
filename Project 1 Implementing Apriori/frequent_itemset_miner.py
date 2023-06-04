"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).
Tip: you can use Arrays.toString(int[] a) to print an itemset.

The items in an itemset must be printed in lexicographical order. However, the itemsets themselves can be printed in
any order.

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = "<write here your group, first name(s) and last name(s)>"
"""
import time

class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self.transactions = list()
        self.items = set()

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            lines = [line for line in lines if line]  # Skipping blank lines
            for line in lines:
                transaction = list(map(int, line.split(" ")))
                self.transactions.append(transaction)
                for item in transaction:
                    self.items.add(item)
        except IOError as e:
            print("Unable to read dataset file!\n" + e)

    def trans_num(self):
        """Returns the number of transactions in the dataset"""
        return len(self.transactions)

    def items_num(self):
        """Returns the number of different items in the dataset"""
        return len(self.items)

    def get_transaction(self, i):
        """Returns the transaction at index i as an int array"""
        return self.transactions[i]


def F_i(F, i):
    """ Return all the element at level i in dataset F """
    f_i = []
    for item in F:
        if len(item) == i:
            f_i.append(item)
    return f_i


def to_string(list_to_convert):
    """ Convert a list of item in a string ordered alphabetically/numericaly """
    string = ""
    for item in list_to_convert:
        string += str(item)

    return string


def to_list(list_of_string):
    """ Convert a list of string in a list of list of characters """
    listed = []
    for item in list_of_string:
        listed.append(sorted(item))

    return listed


def frequency(c, all_transaction, nbre_transaction):
    """ Function that return the frequency of the candidate c is the dataset """
    support = 0
    for transaction in all_transaction : # Go to each transaction
        count = 0
        for char in c : # Check if the item is in the transaction
            for i in range(len(transaction)) :
                if char == transaction[i]:
                    count += 1

            if count == len(c):
                support += 1
                break # Found

    freq = support / nbre_transaction
    return freq


def generate_candidates(candidates):
    """ Generate candidates by combining strings that are identical, except for the last symbol """
    generated_candidates = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
                if candidates[i][:-1] == candidates[j][:-1]:  # We compare the string except for the last symbol
                    generated_candidates.append(candidates[i][:-1])
                    generated_candidates[-1].append(candidates[i][-1])
                    generated_candidates[-1].append(candidates[j][-1])
    return generated_candidates



def apriori(filepath, minFrequency):
    """ Runs the apriori algorithm (the 'basic' one) on the specified file with the given minimum frequency """
    # Initialisation
    dataset = Dataset(filepath)
    F = []
    all_transaction = dataset.transactions
    nbre_transaction = dataset.trans_num()
    
    
    # We create first F_1
    for item in dataset.items:
        support = 0
        for transaction in all_transaction:
            if item in transaction:
                support += 1
        freq = support / nbre_transaction
        if freq >= minFrequency:
            # Convert to the correct output format 
            a = []
            a.append(item)
            str_print = " (" + str(freq) + ")"
            print(a,  str_print)
            F.append([item])
  
    # Loop to find the solution
    i = 1
    while F_i(F, i):
        candidates = generate_candidates(F_i(F, i))  # Generate candidates for the next level
        for c in candidates:
            freq = frequency(c, all_transaction, nbre_transaction)
            if freq >= minFrequency:
                # Convert to the correct output format 
                tmp = []
                for i in range(len(c)):
                    tmp.append(int(c[i]))
                str_print = " (" + str(freq) + ")"
                print(tmp, str_print)
                F.append(c)  # Keep the candidate with enough frequency in the dataset
        i += 1




"""
Part of our Alternative miner

"""

def projected_dataset_make(items,nbre_transaction, all_transaction, minFrequency) :
    """ This function permit to create the initial projected dataset. """
    projected_dataset = []
    new_items = []
    for it in items:
        support = 0
        tmp = []
        for i in range(nbre_transaction):
            if it in all_transaction[i]:
                support += 1
                tmp.append(i)
        freq = support / nbre_transaction
        if freq >= minFrequency:
            projected_dataset.append(tmp)   
            new_items.append(it)
    
    return  new_items, projected_dataset


def create_iteration(items, projected_dataset) :
    """ This function permit to create the new itemset and new projected dataset. """
    new_itemset = []
    new_projected = []

    # New itemset
    tmp = []
    for it in items[0] :
        tmp.append(it)

    for j in items[1:] :
        tmp2 = tmp.copy()
        for z in j :
            if z not in tmp2 :
                tmp2.append(z)
        new_itemset.append(tmp2)

    # New projected dataset
    for i in range(len(projected_dataset[1:])) :
        tmp = []
        for j in projected_dataset[1:][i] :
            if(j in projected_dataset[0]) :
                tmp.append(j)
        if tmp == [] : # If we don't find any intersection, we will be < min Frenquency and not use anymore this item (for the current branche)
            del new_itemset[i]

        else :
            new_projected.append(tmp)
    
    return new_itemset, new_projected




def depthFirstSearch(items, projected_dataset, nbre_transaction, minFrequency) :
    for i in range(len(items)) : 

        freq = len(projected_dataset[i])/nbre_transaction

        if freq >= minFrequency :
            # To print
            str_print = " (" + str(freq) + ")"
            print(items[i],  str_print)

            new_item, new_projected = create_iteration(items[i:], projected_dataset[i:])

            # Iteration
            depthFirstSearch(new_item, new_projected, nbre_transaction, minFrequency)


def alternative_miner(filepath, minFrequency):
    """ Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency """
    dataset = Dataset(filepath)
    all_transaction = dataset.transactions
    nbre_transaction = len(all_transaction) 
    items = list(dataset.items)

    new_items, projected_dataset = projected_dataset_make(items,nbre_transaction, all_transaction, minFrequency)

    tmp = []
    array_item = []
    for i in new_items :
        tmp = []
        tmp.append(i)
        array_item.append(tmp)
    depthFirstSearch(array_item, projected_dataset, nbre_transaction, minFrequency)


#######
# RUN #
#######
#data = Dataset("Datasets/toy.dat") #Dataset("Datasets/chess.dat")  # Dataset("Datasets/accidents.dat") #Dataset("Datasets/toy.dat")
#print("Data : ", data)
#print("data.transactions : ", data.transactions)
#print("data.trans_num : ", data.trans_num())
#print("data.items : ", data.items)
#print("data len(items) : ", len(data.items))


#start = time.time()
#<Run the algorithm on input_n (input of size n)>
#end = time.time()
#time_n = end - start
#start = time.time()
#apriori("Datasets/toy.dat", 0.125)
alternative_miner("Datasets/toy.dat", 0.125)
#end = time.time()
#time_n = end - start
#print(time_n)

#start = time.time()
#apriori("Datasets/chess.dat", 0.9)
#alternative_miner("Datasets/chess.dat", 0.9)
#end = time.time()
#time_n = end - start
#print(time_n)


#start = time.time()
#apriori("Datasets/accidents.dat", 0.8)
#alternative_miner("Datasets/accidents.dat", 0.8)
#end = time.time()
#time_n = end - start
#print(time_n)

