sol = open("Datasets/Sols/test_supsum_6.txt", "r")
sol = sol.readlines()  # .read()

results = open("q1.txt", "r")
results = results.readlines()  # .read()

#print(sol)
#print("\n")
#print(results)

tmp = 0
if len(results) == len(sol) : 
    print("LEN OK")

for line_s in sol :
    for line_r in results : 
        if line_r == line_s : 
            #print(line_r)
            #print(line_s)
            tmp += 1

print("TMP = ", tmp)