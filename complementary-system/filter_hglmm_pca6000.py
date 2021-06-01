import json

train_path = "data/test_v2.json"
metadata_path = "data/metadata_v2.json"
with open(train_path) as f:
    train_json =  json.load(f)

with open(metadata_path) as f:
    metadata =  json.load(f)

img_ids = []
descs = []

print("Reading descriptions of items..")
for outfit in train_json:
    items = outfit["items"]
    for item in items:
        img_ids.append(item["item_id"])
        desc = metadata[item["item_id"]]["title"]
        if not desc:
            desc = metadata[item["item_id"]]["url_name"]
            desc = desc.replace('\n','').strip().lower()
        descs.append(desc)

print("Reading HGLMM PCA text file..")
counter = 0
desc_len = len(descs)
with open("data/train_hglmm_pca6000.txt","r") as fread, open("data/train_hglmm_pca6000_v2.txt","x") as fwrite:
    for line in fread:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        else:
            vec = line_stripped.split(",")
            title = ','.join(vec[:-6000])
            if title in descs:
                fwrite.write(line)
                counter += 1
                if counter == desc_len:
                    break

