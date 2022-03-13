import torch
import torch.nn.functional as F

# Loss function
class loss_fn(torch.nn.modules.loss._Loss):
    def __init__(self, device, margin=0.5, sigma=2.0, T=2.0):
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
        pos = F.normalize(pos, p=2, dim=1)  # B, 128
        neg = F.normalize(neg, p=2, dim=1)  # B, 128

        pos_numerator = torch.exp((anc*pos).sum(axis=-1)/self.T)
        neg_denominator = torch.exp((anc*neg).sum(axis=-1).sum()/self.T)

        loss = -torch.log(pos_numerator/(pos_numerator+neg_denominator))

        loss = loss.mean()

        return loss