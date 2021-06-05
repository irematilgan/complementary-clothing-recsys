
import json
import os
import torch
from torch._C import device
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.read_outfits as read_outfits
import utils.Resnet_18 as Resnet_18
import utils.style_network as style_network
import utils.triple_network as triple_network



TripletImageLoader = read_outfits.TripletImageLoader
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
#use_fc = config_dict["use_fc"]
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cuda:
        torch.cuda.manual_seed(1)

    meta_file = os.path.join(data_dir,"metadata_v2.json")
    normalize = transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
    metadata = json.load(open(meta_file,"r"))
    text_feature_dim = 6000
    kwargs = {'num_workers' : 1, 'pin_memory' : True} if cuda else {}
    criterion = torch.nn.MarginRankingLoss(margin = margin_triplet_loss)
    model = Resnet_18.resnet18(pretrained = True, embedding_size = embedding_dim)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    print(list(model.parameters())[0].shape)
    image_dataset = TripletImageLoader(metadata,text_feature_dim, transform = transforms.Compose([
        transforms.Scale(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        normalize
    ]))
    print(len(image_dataset))
    train_loader = DataLoader(image_dataset, batch_size=5, shuffle = True, **kwargs)
    num_conditions = len(image_dataset.typespaces)
    resnet = Resnet_18.resnet18(pretrained=True)
    stylenet = style_network.StyleNet(l2_normalize, resnet, embedding_dim, num_conditions)
    triplenet = triple_network.TripleNetwork(stylenet, criterion, margin_triplet_loss, text_feature_dim, embedding_dim)
    parameters = filter(lambda p : p.requires_grad, triplenet.parameters())
    optimizer = torch.optim.Adam(parameters, lr = learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma = 0.1)

    for epoch in range(epochs):
        train(triplenet, train_loader,optimizer,epoch,logging_interval, lr_scheduler)
        

def train(triplenet, train_loader, optimizer, epoch, log_interval, lr_scheduler):
    accs = AvgObject()
    losses = AvgObject()
    emb_losses = AvgObject()
    mask_losses = AvgObject()


    triplenet.train()
    for ind, (img_anc, desc_anc, has_text_anc, img_pos, desc_pos, has_text_pos, img_neg, desc_neg, has_text_neg, condition) in enumerate(train_loader):
        anc = TrainData(img_anc, desc_anc, has_text_anc, condition = condition)
        pos = TrainData(img_pos, desc_pos, has_text_pos, condition = condition)
        neg = TrainData(img_neg, desc_neg, has_text_neg, condition = condition)
        acc, compatibility_loss, sim_loss_image, sim_loss_text, vse_loss, mask_loss, embed_loss = triplenet(anc, pos, neg)
        loss_reg = pl_emb_norm*embed_loss + pl_mask_norm*mask_loss
        loss_sim = pl_vv_sim*sim_loss_image + pl_tt_sim*sim_loss_text
        loss_vse_weighted = pl_vs * vse_loss

        loss = compatibility_loss + loss_reg
        if pl_vs > 0:
            loss += loss_vse_weighted
        if pl_tt_sim > 0 or pl_vv_sim > 0:
            loss += loss_sim
        
        num_items = len(anc)

        accs.update(acc.data[0], num_items)
        losses.update(compatibility_loss.item(), num_items)
        emb_losses.update(embed_loss.item())
        mask_losses.update(mask_loss.item())

        optimizer.zero_grad()
        if loss == loss:
            loss.backward()
            optimizer.step()
            lr_scheduler.step()



        if ind % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, ind * num_items, len(train_loader.dataset),
                losses.data, losses.avg, 
                100. * accs.data, 100. * accs.avg, emb_losses.data, emb_losses.avg))

        
class TrainData:
    def __init__(self,img,desc, has_text, condition = None):
        has_text = has_text.float()
        if cuda:
            img = img.cuda()
            desc = desc.cuda()
            has_text = has_text.cuda()

        img, desc, has_text = Variable(img), Variable(desc), Variable(has_text)
        
        self.images = img
        self.text = desc
        self.has_text = has_text
        self.condition = condition

    def __len__(self):
        return self.images.size(0)

class AvgObject:
    def __init__(self):
        self.data = 0
        self.sum = 0
        self.avg = 0
        self.length = 0
    def update(self, value, n = 1):
        self.data = value
        self.sum += value*n
        self.length += n
        self.avg = self.sum / self.length
    


if __name__ == '__main__':
    main()