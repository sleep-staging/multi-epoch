from augmentations import *
from loss import loss_fn
from model import sleep_model
from train import *
from utils import *

from braindecode.util import set_random_seeds

import os
import numpy as np
import copy
import wandb
import torch
from torch.utils.data import DataLoader, Dataset


PATH = '/mnt/sleepx/'

# Params
SAVE_PATH = "multi-epoch.pth"
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
lr = 5e-4
n_epochs = 50
NUM_WORKERS = 9
N_DIM = 128
EPOCH_LEN = 7

####################################################################################################

random_state = 1234
sfreq = 100
high_cut_hz = 30

# Seeds
rng = np.random.RandomState(random_state)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True

set_random_seeds(seed=random_state, cuda=device == "cuda")


##################################################################################################


# Extract number of channels and time steps from dataset
n_channels, input_size_samples = (2, 3000)
model = sleep_model(n_channels, input_size_samples, n_dim = N_DIM)


q_encoder = model.to(device)

optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
criterion = loss_fn(device).to(device)

#####################################################################################################


class pretext_data(Dataset):

    def __init__(self, filepath):
        
        self.file_path = filepath
        self.idx = np.array(range(len(self.file_path)))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        
        path = self.file_path[index]
        data = np.load(path)
        pos = data['pos'] #(7, 2, 3000)
        pos_len = len(pos) # 7
        pos = pos[pos_len // 2] # (2, 3000)
        neg = data['neg'][pos_len // 2] # (2, 3000)
        anc = copy.deepcopy(pos)
        
        # augment
        pos = augment(pos)
        anc = augment(anc)
        neg = augment(neg)
       
        return anc, pos, neg
    
class train_data(Dataset):

    def __init__(self, filepath):
        
        self.file_path = filepath
        self.idx = np.array(range(len(self.file_path)))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        
        path = self.file_path[index]
        data = np.load(path)
        
        return data['x'], data['y']
    
    

PRETEXT_FILE = os.listdir(os.path.join(PATH, "pretext"))
PRETEXT_FILE.sort(key=natural_keys)
PRETEXT_FILE = [os.path.join(PATH, "pretext", f) for f in PRETEXT_FILE]

TRAIN_FILE = os.listdir(os.path.join(PATH, "train"))
TRAIN_FILE.sort(key=natural_keys)
TRAIN_FILE = [os.path.join(PATH, "train", f) for f in TRAIN_FILE]

TEST_FILE = os.listdir(os.path.join(PATH, "test"))
TEST_FILE.sort(key=natural_keys)
TEST_FILE = [os.path.join(PATH, "test", f) for f in TEST_FILE]

print(f'Number of pretext files: {len(PRETEXT_FILE)}')
print(f'Number of train files: {len(TRAIN_FILE)}')
print(f'Number of test files: {len(TEST_FILE)}')

pretext_loader = DataLoader(pretext_data(PRETEXT_FILE), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
train_loader = DataLoader(train_data(TRAIN_FILE), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(train_data(TEST_FILE), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

##############################################################################################################################


wb = wandb.init(
        project="WTM",
        notes="single-epoch, triplet loss, symmetric loss, 7 epoch length, 2000 samples, same learning rate, using logistic regression with saga solver",
        save_code=True,
        entity="sleep-staging",
        name="single-epoch, epoch=7, samples=2000, symmetric loss, saga",
    )
wb.save('/home2/vivek.talwar/multi-epoch/single_epoch/*.py')
wb.watch([q_encoder],log='all',log_freq=500)

Pretext(
    q_encoder,
    optimizer,
    n_epochs,
    criterion,
    pretext_loader,
    train_loader,
    test_loader,
    wb,
    device,
    SAVE_PATH
)

wb.finish()