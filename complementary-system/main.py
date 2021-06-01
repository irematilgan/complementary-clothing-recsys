
import json
import os
import torch
from torch.utils.data import DataLoader
from read_outfits import TripletImageLoader
from torchvision import transforms
import Resnet_18

with open("complementary-system/conf.json", "r") as f:
    config_dict = json.load(f)

batch_size = config_dict["batch_size"]
epochs = config_dict["epochs"]
cuda = config_dict["cuda"]
momentum = config_dict["momentum"]
learning_rate = config_dict["learning_rate"]
logging_interval = config_dict["logging_interval"]
l2_normalize = config_dict["l2_normalize"]
embedding_dim = config_dict["embedding_dim"]
use_fc = config_dict["use_fc"]
margin_triplet_loss = config_dict["margin_tl"]
pl_emb_norm = config_dict["pl_emb_norm"]
pl_mask_norm = config_dict["pl_mask_norm"]
pl_tt_sim = config_dict["pl_tt_sim"]
pl_vv_sim = config_dict["pl_vv_sim"]
pl_vs = config_dict["pl_vs"]
data_dir = config_dict["data_dir"]
optimizer = config_dict["optimizer"]

def main():
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)

    meta_file = os.path.join(data_dir,"metadata_v2.json")
    normalize = transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
    metadata = json.load(open(meta_file,"r"))
    text_feature_dim = 6000
    kwargs = {'num_workers' : 1, 'pin_memory' : True} if cuda else {}
    criterion = torch.nn.MarginRankingLoss(margin = margin_triplet_loss)
    model = Resnet_18.resnet18(pretrained = True, embedding_size = embedding_dim)
    print(list(model.parameters())[0].shape)
    triplet_img_loader = TripletImageLoader(metadata,text_feature_dim, transform = transforms.Compose([
        transforms.Scale(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        normalize
    ]))

    



    


if __name__ == '__main__':
    main()