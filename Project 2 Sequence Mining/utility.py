
def test_sol(path_sol, path_ans):
    sol = open(path_sol, "r")
    sol = sol.readlines()
    ans = open(path_ans, "r")
    ans = ans.readlines()
    false = []
    unfound = []
    counter = 0
    # If same length, answer is correct
    if len(ans) == len(sol):
        print("Correct answer")
        return false, unfound
    else:
        print("Bad answer: expected length:", len(sol), "obtained length:", len(ans))

    # Else, we count the number of differences
    for line_sol in sol:
        presence = False
        for line_ans in ans:
            if line_ans == line_sol:
                presence = True
        if not presence:
            unfound.append(line_sol)

    for line_ans in ans:
        presence = False
        for line_sol in sol:
            if line_ans == line_sol:
                presence = True
        if not presence:
            false.append(line_ans)

    # print("Counter:", counter)

    return false, unfound
