import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

#### TODO : Get image embedding from resnet
#### Create mask
#### Calculate masked_embedding using image embedding and mask
#### return masked embedding and image embedding
#### Not forget L2, L1 loss

class StyleNet(nn.Module):
    def __init__(self, l2_norm, resnet_model, embed_dim, n_conditions, c = None):
        super(StyleNet,self).__init__()
        self.n_conditions = n_conditions
        self.embed_dim = embed_dim
        self.masks = nn.Embedding(n_conditions,embed_dim)
        self.masks.weight.data.normal_(0.9,0.7)
        self.resnet_model = resnet_model.cuda()
        self.l2_norm = l2_norm
        #mask_weight = np.zeros((n_conditions, embed_dim), dtype = np.float32)

    def forward(self,x,c):
        x_embed = self.resnet_model(x)
        if c is None:
            masks = Variable(self.masks.weight.data)
            masks = masks.unsqueeze(0).repeat(x_embed.size(0),1,1)
            x_embed = x_embed.unsqueeze(1)
            masked_embedding = x_embed.expand_as(masks) * masks

            if self.l2_norm:
                mask_norm = torch.linalg.norm(masked_embedding, ord = 2, dim = 2) + 1e-10
                mask_norm.unsqueeze(-1)
                masked_embedding = masked_embedding/mask_norm.expand_as(masked_embedding.size())
                
            return torch.cat((masked_embedding,x_embed),1)

        else :
            self.mask = self.masks(c)
            relu = nn.ReLU()
            self.mask = relu(self.mask)
            masked_embedding = x_embed.cpu() * self.mask
            mask_norm = self.mask.norm(1)

        embed_norm = x_embed.norm(2)
        if self.l2_norm:
            norm = torch.norm(masked_embedding, p = 2, dim = 1) + 1e-10
            #masked_embedding = masked_embedding/norm.expand_as(masked_embedding)
            norm = norm.unsqueeze(-1)
            masked_embedding = masked_embedding/norm.expand_as(masked_embedding)

        return masked_embedding, mask_norm, embed_norm, x_embed


            




