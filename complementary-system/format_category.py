import json
import os

with open("valid_updated.json","r") as f:
    data = json.load(f)

with open("metadata_updated.json","r") as f:
    metadata = json.load(f)

IMG_PATH = "C:/Users/irem/Desktop/SuitApp/polyvore/polyvore_outfits/images"
file_names = os.listdir(IMG_PATH)
counter = 0
result = []
for item_set in data:
    item_dict = {}
    item_dict["items"] = []
    item_dict["set_id"] = item_set["set_id"]
    bottom = 0
    top = 0
    arr = item_set["items"]
    items_list = []
    counter_img = 0
    for item in arr:
        if item["item_id"] + ".jpg" in file_names:
            counter_img += 1
    
            if(metadata[item["item_id"]]["semantic_category"] == "bottoms"):
                item["index"] = 1
                if(bottom < 1):
                    bottom +=1
                    items_list.append(item)
            elif(metadata[item["item_id"]]["semantic_category"] == "tops"):
                item["index"] = 2
                if(top < 1):
                    top +=1
                    items_list.append(item)
        else :
            counter+=1
    #print("ITEMS = ",len(items_list))
    
    
    if(len(items_list) > 1):
        item_dict["items"] = items_list
        result.append(item_dict)

print(len(result))
print(counter)



with open('valid_updated_v3.json', 'w') as outfile:
    json.dump(result, outfile,indent = 4, sort_keys=True)



    
