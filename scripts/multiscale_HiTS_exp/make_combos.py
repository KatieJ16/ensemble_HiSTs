import numpy as np
import matplotlib.pyplot as plt
import itertools

# Python3 program to find out all
# combinations of positive
# numbers that add upto given number
 
# arr - array to store the combination
# index - next location in array
# num - given number
# reducedNum - reduced number
def findCombinationsUtil(arr, index, num, reducedNum):
 
    # Base condition
    if (reducedNum < 0):
        print("base: arr = ", arr)
        return
    
     # If combination is
    # found, print it
    if (reducedNum == 0):
 
        for i in range(index):
            print(arr[i], end = " ")
        print("")
        return
 
    # Find the previous number stored in arr[].
    # It helps in maintaining increasing order
    prev = 1 if(index == 0) else arr[index - 1]
 
    # note loop starts from previous
    # number i.e. at array location
    # index - 1
    for k in range(prev, num + 1):
         
        # next element of array is k
        arr[index] = k
 
        # call recursively with
        # reduced number
        findCombinationsUtil(arr, index + 1, num,
                                 reducedNum - k)
    

    return arr
# Function to find out all
# combinations of positive numbers
# that add upto given number.
# It uses findCombinationsUtil()
def findCombinations(n):
     
    # array to store the combinations
    # It can contain max n elements
    arr = [0] * n
 
    # find all combinations
    arr = findCombinationsUtil(arr, 0, n, n)


step_sizes =[16, 32, 64]

max_repeat = 5
target = 64*4
# for target in target_list:
print("!=================================================================")
print("target = ", target)


#get all combinations of the step_sizes that sum to target
#https://stackoverflow.com/questions/34517540/find-all-combinations-of-a-list-of-numbers-with-a-given-sum
result = [seq for i in range(int(target/min(step_sizes)), 0, -1)
          for seq in itertools.combinations_with_replacement(step_sizes, i)
          if sum(seq) == target]

print(result)

#filter out things that have 5 of the same number

idx_keep = list()
for i, value in enumerate(result):
#     print(i)
#     print(np.count_nonzero(np.array(i) == 1))
    if (np.count_nonzero(np.array(value) == step_sizes[0]) <= max_repeat and  np.count_nonzero(np.array(value) == step_sizes[1]) <= max_repeat):
        idx_keep.append(i)
        print(i)
        print("value = ", value)

print(idx_keep)
result_less = T = [result[i] for i in idx_keep]#result[int(idx_keep)]
print(result_less)


def unique_permutations(elements):
    if len(elements) == 1:
        yield (elements[0],)
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in unique_permutations(remaining_elements):
                yield (first_element,) + sub_permutation
                

all_combos = []
# Print the obtained permutations
for this_poss in result_less:
    print(this_poss)
    all_combos.append(this_poss)
    if len(np.unique(this_poss)) > 1:
        all_poss = list(unique_permutations(this_poss))

#         all_poss = list(itertools.permutations(this_poss))
        print("len perm = ", len(all_poss))
        for i in all_poss:
            print("i = ", i)
            if i not in all_combos:
    #             print(i)
                print("added")
                all_combos.append(i)
    else:
        print("all same")
print("len all_combos = ", len(all_combos))
# print(all_combos)

np.save("all_combos_16.npy", all_combos)