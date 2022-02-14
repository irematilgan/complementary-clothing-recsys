import json

with open("valid_updated_v3.json","r") as f:
    test = json.load(f)

with open("compatibility_valid.txt","r") as f:
    lines = f.readlines()

set_ids = []
for item_set in test:
    set_ids.append(item_set["set_id"])


with open("compatibility_valid_v3.txt","w") as f:
    for line in lines:
        comp_res = line.split()[0]
        items = line.split()[1:]

        item1_id = items[0].split("_")[0]
        item2_id = items[1].split("_")[0]
        
        if item1_id in set_ids and item2_id in set_ids:
            f.write(f"{comp_res} {item1_id}_1 {item2_id}_2\n")






