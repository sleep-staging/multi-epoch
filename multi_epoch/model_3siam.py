from braindecode.models import SleepStagerChambon2018
from torch import nn

def sleep_model(n_channels, input_size_samples, n_dim=128):
    
    emb_size = 128
    sfreq = 100


    encoder = SleepStagerChambon2018(
        n_channels,
        sfreq,
        n_classes=emb_size,
        n_conv_chs=16,
        input_size_s=input_size_samples / sfreq,
        dropout=0,
        apply_batch_norm=True,
    )

    class Net(nn.Module):
        
        def __init__(self, encoder, dim):
            super().__init__()
            self.enc = encoder
            self.n_dim = dim

            self.p1 = nn.Sequential(
                nn.Linear(128, self.n_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.n_dim, self.n_dim, bias=True),
            )

            self.p2 = nn.Sequential(
                nn.Linear(128, self.n_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.n_dim, self.n_dim, bias=True),
            )

            self.p3 = nn.Sequential(
                nn.Linear(128, self.n_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.n_dim, self.n_dim, bias=True),
            )

        def forward(self, x, proj='anc'):
            
            x = self.enc(x)
            
            if proj == 'anc':
                x = self.p1(x)
                return x
            elif proj == 'pos':
                x = self.p2(x)
                return x
            elif proj == 'neg':
                x = self.p3(x)
                return x

    return Net(encoder, n_dim)