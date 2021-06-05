import torch
import numpy as np
import json
from os import path
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data
import torch

def load_image(path):
    return Image.open(path).convert('RGB')

def load_compatibility_questions():
    return

def load_fitb_questions():
    return

class TripletImageLoader(torch.utils.data.Dataset):

    


    def __init__(self,metadata,text_dim,transform = None, loader = load_image):
        super().__init__()
        print("INFO : TripletImageLoader class has been created..")
        self.id2cat = {}
        self.desc2vec = {}
        #ids = []
        self.img_ids = []
        self.metadata = {}
        self.typespaces = {}
        self.id2index = {}
        self.cat2ids = {}
        self.full_id2img_id = {}
        self.id2ind = {}
        self.pos_pairs = []
        self.text_dim = 6000
        self.metadata = metadata
        #self.ids = list(self.metadata.keys())
        self.text_dim = text_dim
        self.transform = transform
        self.loader = loader

        """
        for key in self.ids:
            self.id2cat.update({key : metadata[key]["semantic_category"]})"""

        # Read typespace file
        with open("complementary-system/type_space.txt") as f:
            for index, line in enumerate(f):
                self.typespaces[tuple(line.split(","))] = index

            
            #self.typespaces = [tuple(map(str, i.split('\n'))) for i in f]

        # Read HGLMM vectorized, PCA downsized text data
        print("INFO : Reading train_hglmm_pca6000_v2.txt")
        with open("data/train_hglmm_pca6000_v2.txt") as f:
            i = 0
            for ind, line in enumerate(f):
                #print(ind)
                line = line.strip()
                if not line:
                    continue

                vec = line.split(",")
                title = ','.join(vec[:-self.text_dim])
                vec = np.array([float(x) for x in vec[-self.text_dim:]], np.float32)
                assert(len(vec) == text_dim)
                self.desc2vec[title] = vec
                #i+=1
                #if i == 50:
                #    break
        

        with open("data/test_v2.json") as f:
            train_dict = json.load(f)

        for outfit_set in train_dict:
            outfit_id = outfit_set["set_id"]
            for item in outfit_set["items"]:
                img_id = item["item_id"]
                category = metadata[img_id]["semantic_category"]
                self.id2cat[img_id] = category
                if category not in self.cat2ids:
                    self.cat2ids[category] = {}
                
                if outfit_id not in self.cat2ids[category]:
                    self.cat2ids[category][outfit_id] = []

                self.cat2ids[category][outfit_id].append(img_id)
                self.full_id2img_id[f'{outfit_id}_{item["index"]}'] = img_id
                self.img_ids.append(img_id)

        self.id2desc = {}
        for img_id in self.img_ids:
            desc = metadata[img_id]["title"]

            #Use url name if title name does not exist
            if not desc:
                desc = metadata[img_id]["url_name"]
            
            desc = desc.replace('\n','').strip().lower()
            #print(img_id)
            #print(desc)
            if desc and desc in self.desc2vec:
                self.id2desc[img_id] = desc

        for ind,img_id in enumerate(self.img_ids):
            self.id2ind[img_id] = ind

        ## Get positive pairs
        ## img1, text_vec1, img2, text_vec2
        self.pos_pairs = []
        for outfit_set in train_dict:
            items = outfit_set["items"]
            outfit_id = outfit_set["set_id"]
            set_length = len(items)
            assert(set_length == 2)
            self.pos_pairs.append([outfit_id, items[0]["item_id"], items[1]["item_id"]])

        #print(self.id2desc)
        #img_pos, desc_pos, type_pos = self.load_item(pos_pairs[0][-1])
        
        #print(self.sample_negative(self.id2cat[pos_pairs[0][-1]],pos_pairs[0][-1]))
        print(self.id2cat['1027065'])


           
        
    def get_typespace(self, anchor, pos):
        query = (anchor,pos)
        if query not in self.typespaces:
            query = (pos,anchor)

        return self.typespaces[query]
        
   
    def sample_negative(self, pos_type,pos_id):
        outfits = self.cat2ids[pos_type]
        outfit_ids = list(outfits.keys())
        outfit_ind = np.random.randint(0,len(outfits))
        while outfits[outfit_ids[outfit_ind]][0] == pos_id:
            outfit_ind = np.random.randint(0,len(outfits))
        neg_id = outfits[outfit_ids[outfit_ind]][0]

        return neg_id

    def load_item(self, img_id):
        img_path = r"C:\Users\irem\Desktop\SuitApp\polyvore\polyvore_outfits\images"
        img = self.loader(path.join(img_path, img_id + ".jpg"))
        if self.transform is not None:
            img = self.transform(img)

        try:
            img_desc = self.id2desc[img_id]
            vec = self.desc2vec[img_desc]
            has_text = True
        except KeyError:
            vec = np.zeros(self.text_dim, np.float32)
            has_text = False
            
  
        item_type = self.id2cat[img_id]
        return img, vec, has_text, item_type

    def __getitem__(self,index):

        # For training
        outfit_id, anchor_id, pos_id = self.pos_pairs[index]
        img_anchor, desc_anchor, has_text1, type_anchor = self.load_item(anchor_id)
        img_pos, desc_pos, has_text2, type_pos = self.load_item(pos_id)
        condition = self.get_typespace(type_anchor, type_pos)

        neg_id = self.sample_negative(type_pos, pos_id)
        img_neg, desc_neg, has_text3, type_neg = self.load_item(neg_id)

        #print(f"anchor : {anchor_id} pos : {pos_id}, neg : {neg_id} condition : {condition}")
        return img_anchor, desc_anchor, has_text1, img_pos, desc_pos, has_text1, img_neg, desc_neg, has_text3, condition

        # TODO : Test


    def shuffle(self):
        np.random.shuffle(self.pos_pairs)

    def __len__(self):
        # For training
        return len(self.pos_pairs)

        # TODO : Test