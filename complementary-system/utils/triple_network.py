import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

def modifiedRankingLoss(pos_dist, neg_dist, has_text, margin) :
    margin_diff = torch.clamp((pos_dist-neg_dist)+margin,min = 0, max = 1e6)
    total = max(torch.sum(has_text),1)
    return torch.sum(margin_diff*has_text)/total

def calcAccuracy(pos_dist, neg_dist):
    is_cuda = pos_dist.is_cuda
    diff = (pos_dist - neg_dist).cpu().data
    num_samples = pos_dist.size()[0]
    acc = (diff > 0).sum() * 1.0 / num_samples
    acc = torch.from_numpy(np.array([acc], dtype = np.float32))
    if is_cuda:
        acc = acc.cuda()
    return Variable(acc)

class TextBranch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextBranch, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim,output_dim).to(device = 'cuda:0'),
            nn.BatchNorm1d(output_dim, eps = 0.001, momentum=0.01).to(device = 'cuda:0'),
            nn.ReLU(inplace=True).to(device = 'cuda:0')
        )
        #self.fc1 = nn.ModuleList(self.fc1)
        self.fc2 = nn.Linear(output_dim, output_dim).to(device = 'cuda:0')
        #self.fc2 = torch.nn.ModuleList(self.fc2)

    def forward(self, x):
        #x = x.cpu()
        x = self.fc1(x)
        x = self.fc2(x)
        norm = torch.norm(x, p = 2, dim = 1) + 1e-10
        norm = norm.unsqueeze(-1)
        x = x/norm.expand_as(x)
        return x




class TripleNetwork(nn.Module):
    def __init__(self, stylenet, criterion, margin_triplet_loss, text_feature_dim, embedding_dim):
        super().__init__()
        self.in_feature = text_feature_dim
        self.out_feature = embedding_dim
        self.typenet = stylenet
        self.margin = margin_triplet_loss
        self.criterion = criterion
        self.text_branch = TextBranch(text_feature_dim, embedding_dim)

    def img_forward(self, imgs_x, imgs_y, imgs_z, condition, batch_size):
        embedding_x, mask_norm_x, embed_norm_x, general_x = self.typenet(imgs_x, c = condition)
        embedding_y, mask_norm_y, embed_norm_y, general_y = self.typenet(imgs_y, c = condition)
        embedding_z, mask_norm_z, embed_norm_z, general_z = self.typenet(imgs_z, c = condition)

        mask_norm = (mask_norm_x + mask_norm_y + mask_norm_z) / 3
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        mask_loss = mask_norm/batch_size
        embed_loss = embed_norm/np.sqrt(batch_size)



        dist_pos = F.pairwise_distance(embedding_x, embedding_y, 2)
        dist_neg = F.pairwise_distance(embedding_x, embedding_z, 2)
        acc = calcAccuracy(dist_neg, dist_pos)
        target = torch.FloatTensor(dist_pos.size()).fill_(1)
        if dist_pos.is_cuda:
            target = target.cuda()
        target = Variable(target)

        comp_loss = self.criterion(dist_neg, dist_pos, target)

        dist_cat_p = F.pairwise_distance(general_y, general_z, 2)
        dist_cat_n1 = F.pairwise_distance(general_x, general_y, 2)
        dist_cat_n2 = F.pairwise_distance(general_x, general_z, 2)

        target = target.cuda()

        sim_loss1 = self.criterion(dist_cat_p, dist_cat_n1,target)
        sim_loss2 = self.criterion(dist_cat_p, dist_cat_n2,target)
        sim_loss_image = (sim_loss1+sim_loss2)/2

        return acc, comp_loss, sim_loss_image, mask_loss, embed_loss, general_x, general_y, general_z

    def text_forward(self, desc_x, desc_y, desc_z, has_text_x, has_text_y, has_text_z):
        embed_x = self.text_branch(desc_x)
        embed_y = self.text_branch(desc_y)
        embed_z = self.text_branch(desc_z)

        has_text = has_text_x * has_text_y * has_text_z
        pos_dist = F.pairwise_distance(embed_z, embed_y, 2)
        neg_dist1 = F.pairwise_distance(embed_x, embed_y, 2)
        neg_dist2 = F.pairwise_distance(embed_x, embed_z, 2)

        sim_loss_1 = modifiedRankingLoss(pos_dist, neg_dist1, has_text, self.margin)
        sim_loss_2 = modifiedRankingLoss(pos_dist, neg_dist2, has_text, self.margin)
        sim_text_loss = (sim_loss_1+sim_loss_2)/2


        return sim_text_loss, embed_x, embed_y, embed_z

    def vse_loss(self, desc_x, has_text, general_x, general_y, general_z):
        loss_x = F.pairwise_distance(general_x, desc_x, 2)
        loss_y = F.pairwise_distance(general_y, desc_x, 2)
        loss_z = F.pairwise_distance(general_z, desc_x, 2)

        loss_vse1 = modifiedRankingLoss(loss_x, loss_y, has_text, self.margin)
        loss_vse2 = modifiedRankingLoss(loss_x, loss_z, has_text, self.margin)

        return (loss_vse1+loss_vse2)/2

    def forward(self, x, y, z):
        acc, compatibility_loss, sim_loss_image, mask_loss, embed_loss, general_x, general_y, general_z = self.img_forward(x.images, y.images, z.images, x.condition, len(x))
        sim_text_loss, embed_x, embed_y, embed_z = self.text_forward(x.text, y.text, z.text, x.has_text, y.has_text, z.has_text)
        vse_loss_x = self.vse_loss(embed_x, x.has_text, general_x, general_y, general_z)
        vse_loss_y = self.vse_loss(embed_y, y.has_text, general_y, general_x, general_z)
        vse_loss_z = self.vse_loss(embed_z, z.has_text, general_z, general_y, general_x)
        vse_loss = (vse_loss_x + vse_loss_y + vse_loss_z) / 3

        return acc, compatibility_loss, sim_loss_image, sim_text_loss, vse_loss, mask_loss, embed_loss

    