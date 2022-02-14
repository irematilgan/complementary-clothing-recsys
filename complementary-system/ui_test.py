
import torch
from torchvision import transforms
from utils.triple_network import TripleNetwork
from utils.style_network import StyleNet
from torch.optim import Adam
import utils.Resnet_18 as Resnet_18
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os

class ComplementarySystem:
    def __init__(self):
        model = Resnet_18.resnet18(pretrained=True)
        stylenet = StyleNet(True,model,64,1,False)
        criterion = torch.nn.MarginRankingLoss(margin = 0.2)
        self.triplenet = TripleNetwork(stylenet,criterion,0.2,6000,64)
        checkpoint = torch.load('runs/model/important/model_best_0001_adam.pth.tar')
        self.triplenet.load_state_dict(checkpoint['state_dict'])
        self.data_transforms = transforms.Compose([
            transforms.Scale(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
        ])
    def test_fitb(self,triplenet, anchor_img, imgs):
        triplenet.eval()
        scores = np.zeros(len(imgs), dtype = np.float32)
        anchor_img = anchor_img.cuda()
        anchor_img = Variable(anchor_img)
        anchor_embedding = triplenet.typenet(anchor_img).data[0][0].unsqueeze(0)

        for ind, image in enumerate(imgs):
            image = image.cuda()
            image = Variable(image)
            emb = triplenet.typenet(image).data[0][0].unsqueeze(0)
            #print(emb.size())
            scores[ind] = F.pairwise_distance(anchor_embedding,emb,2)
        #print(scores)
        return scores
    def load_image(self,file_name, transform):
        self.files = os.listdir(file_name)
        #print(self.files)
        imgs = []
        for file in self.files:
            imgs.append(transform(Image.open(os.path.join(file_name,file)).convert('RGB')).unsqueeze(0))
        print(imgs[0].size())
        return imgs

    def startModel(self, file_anchor, file_complement, is_multiple = False):
        img_anchor = self.data_transforms(Image.open(file_anchor).convert('RGB')).unsqueeze(0)
        imgs_comp = self.load_image(file_complement,self.data_transforms)
        results = self.test_fitb(self.triplenet,img_anchor,imgs_comp)

        if is_multiple:
            sorted_indices = sorted(range(len(results)), key=lambda k: results[k])
            return [self.files[ind] for ind in sorted_indices]

        return self.files




#optimizer.load_state_dict(checkpoint['optimizer'])

#img1 = Image.open("complementary-system/utils/ui_data/3445983.jpg").convert('RGB')
#img2 = Image.open("complementary-system/utils/ui_data/7167294.jpg").convert('RGB')

#cs = ComplementarySystem()
#cs.startModel("complementary-system/utils/ui_data/3445983.jpg","complementary-system/utils/ui_data/complements")


"""
img_array, files = load_image("complementary-system/utils/ui_data/complements",data_transforms)


img_transformed1 = data_transforms(img1).unsqueeze(0)


#imgs_transformed = data_transforms(torch.Tensor(np.array(comps)))
#test_compat(triplenet,[img_transformed1,img_transformed2])
file_ind = test_fitb(triplenet,img_transformed1,img_array)
print(files[file_ind])"""
        
