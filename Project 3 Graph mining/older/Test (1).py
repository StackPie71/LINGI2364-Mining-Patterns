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


real, test = open_file("examples/task2_small_5_5_4.txt", "point1.txt")

tmp = 0
for line1 in real :
    for i in range(len(test)) : 
        if line1 == test[i] : 
            tmp += 1
            print(i)
            break

print("Len of real = ", len(real))
print("Number of same lines = ", tmp)