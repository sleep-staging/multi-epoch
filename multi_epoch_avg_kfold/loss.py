import torch
import torch.nn.functional as F

# Loss function
class loss_fn(torch.nn.modules.loss._Loss):
    def __init__(self, device, margin=0.5, sigma=2.0, T=2):
        """
        T: softmax temperature (default: 0.07)
        """
        super(loss_fn, self).__init__()
        self.T = T
        self.device = device
        self.margin = margin
        self.sigma = sigma
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, anc, pos, neg):

        # L2 normalize
        anc = F.normalize(anc, p=2, dim=1)  # B, 128
        pos = F.normalize(pos, p=2, dim=2)  # B, 7, 128
        neg = F.normalize(neg, p=2, dim=2)  # B, 7, 128

        # No contrastive loss -> bmm
        pos_sim = torch.mean(pos, dim = 1)  # B, 128
        neg_sim = torch.mean(neg, dim = 1)  # B, 128
        
        # Triplet loss
        l_pos = torch.exp(
            -torch.sum(torch.pow(anc - pos_sim, 2), dim=1) / (2 * self.sigma ** 2)
        )
        l_neg = torch.exp(
            -torch.sum(torch.pow(anc - neg_sim, 2), dim=1) / (2 * self.sigma ** 2)
        )

        zero_matrix = torch.zeros_like(l_pos)
        loss = torch.max(zero_matrix, l_neg - l_pos + self.margin).mean()

        return loss