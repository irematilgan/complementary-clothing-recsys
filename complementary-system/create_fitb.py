import numpy as np
import json

with open("data/compatibility_valid_v3.txt","r") as f:
    lines = f.readlines()

lines_arr = []
arr = []

for line in lines:
    lines_arr.append(line)


arr_len = len(lines_arr)
for ind, line in enumerate(lines_arr):
    fitb_dict = {}
    items = line.split()[1:]
    fitb_dict["question"] = items[0]
    fitb_dict["blank_position"] = 2
    ans = []
    ans.append(items[1])

    counter = 0
    while counter != 4:
        rand_num = np.random.randint(0,len(lines_arr))
        if rand_num != ind:
            counter +=1
            last_item = lines_arr[rand_num].split()[-1]
            ans.append(last_item)
    
    fitb_dict["answers"] = ans
    arr.append(fitb_dict)


with open('fitb_valid.json', 'w') as outfile:
    json.dump(arr, outfile,indent = 4, sort_keys=True)

