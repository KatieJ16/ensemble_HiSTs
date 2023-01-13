"""
Make all combos

"""

import numpy as np
import itertools

def flatten_list(_2d_list):
    #from https://stackabuse.com/python-how-to-flatten-list-of-lists/
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    print("len(flat_list) = ", len(flat_list))
    return flat_list

def make_all_combos(step_sizes, max_repeat=4, target_list=None, file_name_base=None):
        '''
            Makes a list of all possible combinations to test
        '''
        
        if target_list is None:
            target_list = np.arange(np.min(step_sizes)*2, np.max(step_sizes) * max_repeat, np.min(step_sizes))
            
        all_combos = list()
        for target in target_list:
            file_name = file_name_base+'_'+str(target)+'.npy'
            try:
                all_combos = np.load(file_name, allow_pickle=True)
                print("all_combose_loaded for target = ", target)
                print("all_combos = ", all_combos)
                continue
            except:
                all_combos = list()
            
            print("!=================================================================")
            print("target = ", target)

            #get all combinations of the step_sizes that sum to target
            #https://stackoverflow.com/questions/34517540/find-all-combinations-of-a-list-of-numbers-with-a-given-sum
            result = [seq for i in range(target, 0, -1)
                      for seq in itertools.combinations_with_replacement(step_sizes, i)
                      if sum(seq) == target]
            
            result = np.array(result)

            #filter out things that have 5 of the same number
            idx_keep = list()
            for i, value in enumerate(result):
                if (np.count_nonzero(np.array(value) == step_sizes[0]) < max_repeat) and (np.count_nonzero(np.array(value) == step_sizes[1]) < max_repeat):
                    idx_keep.append(i)
                

            result_less = result[idx_keep]
            print(result_less)

            # Find all permutations
            for this_poss in result_less:
                for i in list(itertools.permutations(this_poss)):
                    if i not in all_combos:
                        try:
                            all_combos.append(i)
                        except:
                            np.append(all_combos, i)
            print("len all_combos = ", len(all_combos))
            
#             try:
            np.save(file_name, all_combos, allow_pickle=True)
            print("saved")
#             except:
#                 pass
            
    
        return #all_combos
    
#make each one
step_sizes = [4, 8, 16]
file_name_base='all_combos_'+str(min(step_sizes))
max_repeat=4
target_list = np.arange(np.min(step_sizes)*2, np.max(step_sizes) * max_repeat, np.min(step_sizes))


make_all_combos(step_sizes, target_list=target_list, file_name_base=file_name_base)


#put all together
all_combos = list()
for target in target_list:
    file_name = file_name_base+'_'+str(target)+'.npy'
    print("np.load(file_name, allow_pickle=True)
    all_combos = all_combos + np.load(file_name, allow_pickle=True)
    print("len(all_combos) = ", len(all_combos))

print("saving all")
print("flatten_list(all_combos) = ", len(flatten_list(all_combos)))
np.save(file_name_base+'.npy', flatten_list(all_combos), allow_pickle=True)

print(len(np.load(file_name_base+'.npy', allow_pickle=True)))

