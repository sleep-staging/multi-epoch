from braindecode.datasets import BaseConcatDataset, BaseDataset
import os

import numpy as np
import pandas as pd
import mne
from mne.datasets.sleep_physionet.age import fetch_data


from braindecode.datautil.preprocess import preprocess, Preprocessor
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.preprocess import zscore

import numpy as np
import copy
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression as LR


class SleepPhysionet(BaseConcatDataset):
    def __init__(
        self,
        subject_ids=None,
        recording_ids=None,
        preload=False,
        load_eeg_only=True,
        crop_wake_mins=30,
        crop=None,
    ):
        if subject_ids is None:
            subject_ids = range(83)
        if recording_ids is None:
            recording_ids = [1, 2]

        paths = fetch_data(
            subject_ids,
            recording=recording_ids,
            on_missing="warn",
            path="/scratch/SLEEP_data/",
        )

        all_base_ds = list()
        for p in paths:
            raw, desc = self._load_raw(
                p[0],
                p[1],
                preload=preload,
                load_eeg_only=load_eeg_only,
                crop_wake_mins=crop_wake_mins,
                crop=crop,
            )
            base_ds = BaseDataset(raw, desc)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)

    @staticmethod
    def _load_raw(
        raw_fname,
        ann_fname,
        preload,
        load_eeg_only=True,
        crop_wake_mins=False,
        crop=None,
    ):
        ch_mapping = {
            "EOG horizontal": "eog",
            "Resp oro-nasal": "misc",
            "EMG submental": "misc",
            "Temp rectal": "misc",
            "Event marker": "misc",
        }
        exclude = list(ch_mapping.keys()) if load_eeg_only else ()

        raw = mne.io.read_raw_edf(raw_fname, preload=preload, exclude=exclude)
        annots = mne.read_annotations(ann_fname)
        raw.set_annotations(annots, emit_warning=False)

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [x[-1] in ["1", "2", "3", "4", "R"] for x in annots.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annots[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))

        # Rename EEG channels
        ch_names = {i: i.replace("EEG ", "") for i in raw.ch_names if "EEG" in i}
        raw.rename_channels(ch_names)

        if not load_eeg_only:
            raw.set_channel_types(ch_mapping)

        if crop is not None:
            raw.crop(*crop)

        basename = os.path.basename(raw_fname)
        subj_nb = int(basename[3:5])
        sess_nb = int(basename[5])
        desc = pd.Series({"subject": subj_nb, "recording": sess_nb}, name="")

        return raw, desc


random_state = 1234
n_jobs = 1
sfreq = 100

SUBJECTS = np.arange(83)
RECORDINGS = [1, 2]

dataset = SleepPhysionet(
    subject_ids=SUBJECTS, recording_ids=RECORDINGS, crop_wake_mins=30
)


high_cut_hz = 30

preprocessors = [
    Preprocessor(lambda x: x * 1e6),
    Preprocessor("filter", l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs),
]

# Transform the data
preprocess(dataset, preprocessors)


window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq

mapping = {  # We merge stages 3 and 4 following AASM standards.
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    preload= False,
    mapping=mapping,
)


preprocess(windows_dataset, [Preprocessor(zscore)])


###################################################################################################################################


from torch.utils.data.sampler import Sampler
from sklearn.utils import check_random_state


class RecordingSampler(Sampler):
    def __init__(self, metadata, random_state=None, epoch_len=7):

        self.metadata = metadata
        self._init_info()
        self.rng = check_random_state(random_state)
        self.epoch_len = epoch_len

    def _init_info(self):
        keys = ["subject", "recording"]

        self.metadata = self.metadata.reset_index().rename(
            columns={"index": "window_index"}
        )
        self.info = (
            self.metadata.reset_index()
            .groupby(keys)[["index", "i_start_in_trial"]]
            .agg(["unique"])
        )
        self.info.columns = self.info.columns.get_level_values(0)

    def sample_recording(self):
        """Return a random recording index."""
        return self.rng.choice(self.n_recordings)

    def sample_window(self, rec_ind=None):
        """Return a specific window."""
        if rec_ind is None:
            rec_ind = self.sample_recording()
        win_ind = self.rng.choice(
            self.info.iloc[rec_ind]["index"][self.epoch_len // 2 : -self.epoch_len // 2]
        )
        return win_ind, rec_ind

    def __iter__(self):
        raise NotImplementedError

    @property
    def n_recordings(self):
        return self.info.shape[0]


class RelativePositioningSampler(RecordingSampler):
    def __init__(
        self,
        metadata,
        tau_pos,
        tau_neg,
        n_examples,
        same_rec_neg=True,
        random_state=None,
        epoch_len=7,
    ):
        super().__init__(metadata, random_state=random_state, epoch_len=epoch_len)

        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.epoch_len = epoch_len
        self.n_examples = n_examples
        self.same_rec_neg = same_rec_neg

    def _sample_pair(self):
        """Sample a pair of two windows."""
        # Sample first window
        win_ind1, rec_ind1 = self.sample_window()
        ts1 = self.metadata.iloc[win_ind1]["i_start_in_trial"]
        ts = self.info.iloc[rec_ind1]["i_start_in_trial"]

        epoch_min = self.info.iloc[rec_ind1]["i_start_in_trial"][self.epoch_len // 2]
        epoch_max = self.info.iloc[rec_ind1]["i_start_in_trial"][-self.epoch_len // 2]

        if self.same_rec_neg:
            mask = ((ts <= ts1 - self.tau_neg) & (ts >= epoch_min)) | (
                (ts >= ts1 + self.tau_neg) & (ts <= epoch_max)
            )

        if sum(mask) == 0:
            raise NotImplementedError
        win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]["index"][mask])

        return win_ind1, win_ind2

    def __iter__(self):

        for i in range(self.n_examples):

            yield self._sample_pair()

    def __len__(self):
        return self.n_examples


import numpy as np
from braindecode.datasets import BaseConcatDataset

rng = np.random.RandomState(1234)

NUM_WORKERS = 0 if n_jobs <= 1 else n_jobs
PERSIST = False if NUM_WORKERS <= 1 else True


subjects = np.unique(windows_dataset.description["subject"])
sub_pretext = rng.choice(subjects, 58, replace=False)
sub_train = sorted(
    rng.choice(sorted(list(set(subjects) - set(sub_pretext))), 10, replace=False)
)
sub_test = sorted(list(set(subjects) - set(sub_pretext) - set(sub_train)))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Pretext: {sub_pretext} \n")
print(f"Train: {sub_train} \n")
print(f"Test: {sub_test} \n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


# Augmentations
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, periodogram


def denoise_channel(ts, bandpass, signal_freq):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1

    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)

    return np.array(ts_out)


def noise_channel(ts, mode, degree):
    """
    Add noise to ts

    mode: high, low, both
    degree: degree of noise, compared with range of ts

    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)

    """
    len_ts = len(ts)
    num_range = np.ptp(ts) + 1e-4  # add a small number for flat signal

    ### high frequency noise
    if mode == "high":
        noise = degree * num_range * (2 * np.random.rand(len_ts) - 1)
        out_ts = ts + noise

    ### low frequency noise
    elif mode == "low":
        noise = degree * num_range * (2 * np.random.rand(len_ts // 100) - 1)
        x_old = np.linspace(0, 1, num=len_ts // 100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind="linear")
        noise = f(x_new)
        out_ts = ts + noise

    ### both high frequency noise and low frequency noise
    elif mode == "both":
        noise1 = degree * num_range * (2 * np.random.rand(len_ts) - 1)
        noise2 = degree * num_range * (2 * np.random.rand(len_ts // 100) - 1)
        x_old = np.linspace(0, 1, num=len_ts // 100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind="linear")
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts

    return out_ts


def add_noise(x, ratio):
    """
    Add noise to multiple ts
    Input:
        x: (n_channel, n_length)
    Output:
        x: (n_channel, n_length)
    """
    for i in range(x.shape[0]):

        mode = np.random.choice(["high", "low", "both", "no"])
        x[i, :] = noise_channel(x[i, :], mode=mode, degree=0.05)

    return x


def remove_noise(x, ratio):
    """
    Remove noise from multiple ts
    Input:
        x: (n_channel, n_length)
    Output:
        x: (n_channel, n_length)

    Three bandpass filtering done independently to each channel
    sig1 + sig2
    sig1
    sig2
    """
    bandpass1 = (1, 5)
    bandpass2 = (30, 49)
    signal_freq = 100

    for i in range(x.shape[0]):
        rand = np.random.rand()
        if rand > 0.75:
            x[i, :] = denoise_channel(
                x[i, :], bandpass1, signal_freq
            ) + denoise_channel(x[i, :], bandpass2, signal_freq)
        elif rand > 0.5:
            x[i, :] = denoise_channel(x[i, :], bandpass1, signal_freq)
        elif rand > 0.25:
            x[i, :] = denoise_channel(x[i, :], bandpass2, signal_freq)
        else:
            pass
    return x


def crop(x):
    n_length = x.shape[1]
    l = np.random.randint(1, n_length - 1)
    x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

    return x


def augment(x):
    t = np.random.rand()
    if t > 0.75:
        x = add_noise(x, ratio=0.5)
    elif t > 0.5:
        x = remove_noise(x, ratio=0.5)
    elif t > 0.25:
        x = crop(x)
    else:
        x = x[[1, 0], :]  # channel flipping
    return x


class RelativePositioningDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target."""

    def __init__(self, list_of_ds, epoch_len=7):
        super().__init__(list_of_ds)
        self.return_pair = True
        self.epoch_len = epoch_len

    def __getitem__(self, index):

        pos, neg = index
        pos_data = []
        neg_data = []

        assert pos != neg, "pos and neg should not be the same"

        for i in range(-(self.epoch_len // 2), self.epoch_len // 2 + 1):
            pos_data.append(super().__getitem__(pos + i)[0])
            neg_data.append(super().__getitem__(neg + i)[0])

        anc = super().__getitem__(pos)[0]
        y = super().__getitem__(pos)[1]

        pos_data = np.stack(pos_data, axis=0)
        neg_data = np.stack(neg_data, axis=0)

        for i in range(pos_data.shape[0]):
            pos_data[i] = augment(pos_data[i])
            neg_data[i] = augment(neg_data[i])

        anc = augment(anc)  # (2, 3000)

        return anc, pos_data, neg_data, y


class TuneDataset(BaseConcatDataset):
    """BaseConcatDataset for train and test"""

    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)

    def __getitem__(self, index):

        X = super().__getitem__(index)[0]
        y = super().__getitem__(index)[1]

        return X, y


split_ids = {"pretext": sub_pretext, "train": sub_train, "test": sub_test}
splitted = dict()

splitted["pretext"] = RelativePositioningDataset(
    [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_pretext]
)

splitted["train"] = TuneDataset(
    [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_train]
)

splitted["test"] = TuneDataset(
    [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_test]
)


# Sampler
tau_pos, tau_neg = int(sfreq * 60), int(sfreq * 15 * 60)

n_examples_pretext = 200 * len(splitted["pretext"].datasets)

pretext_sampler = RelativePositioningSampler(
    splitted["train"].get_metadata(),
    tau_pos=tau_pos,
    tau_neg=tau_neg,
    n_examples=n_examples_pretext,
    same_rec_neg=True,
    random_state=random_state,
)


import torch
from torch import nn
from braindecode.util import set_random_seeds
from braindecode.models import SleepStagerChambon2018

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True

set_random_seeds(seed=random_state, cuda=device == "cuda")

# Extract number of channels and time steps from dataset
n_channels, input_size_samples = windows_dataset[0][0].shape

emb_size = 128
n_dim = 128

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

    def forward(self, x, proj_first=True):

        if proj_first:
            x = self.enc(x)
            x = self.p1(x)
            return x
        else:
            x = self.enc(x)
            x = self.p2(x)
            return x


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
        pos = F.normalize(pos, p=2, dim=2)  # B, 7, 128
        neg = F.normalize(neg, p=2, dim=2)  # B, 7, 128

        # No contrastive loss -> bmm
        pos_sim = torch.bmm(anc.unsqueeze(1), pos.transpose(1, 2))  # B, 1, 7
        pos_sim = self.softmax(pos_sim / self.T)  # B, 1, 7
        pos_agg = torch.bmm(pos_sim, pos).squeeze(1)  # B, 128

        neg_sim = torch.bmm(anc.unsqueeze(1), neg.transpose(1, 2))  # B, 1, 7
        neg_sim = self.softmax(neg_sim / self.T)  # B, 1, 7
        # neg_sim = torch.ones_like(neg_sim).to(device) - neg_sim # B, 1, 7 --> Penalizing the similarity scores for neagtive samples
        neg_agg = torch.bmm(neg_sim, neg).squeeze(1)  # B, 128

        # Triplet loss
        l_pos = torch.exp(
            -torch.sum(torch.pow(anc - pos_agg, 2), dim=1) / (2 * self.sigma ** 2)
        )
        l_neg = torch.exp(
            -torch.sum(torch.pow(anc - neg_agg, 2), dim=1) / (2 * self.sigma ** 2)
        )

        zero_matrix = torch.zeros(l_pos.shape).to(self.device)
        loss = torch.max(zero_matrix, l_neg - l_pos + self.margin).mean()

        return loss


# Train, test
def evaluate(q_encoder, train_loader, test_loader):

    # eval
    q_encoder.eval()

    # process val
    emb_val, gt_val = [], []

    with torch.no_grad():
        for (X_val, y_val) in train_loader:
            X_val = X_val.float()
            y_val = y_val.long()
            X_val = X_val.to(device)
            emb_val.extend(q_encoder(X_val).cpu().tolist())
            gt_val.extend(y_val.numpy().flatten())
    emb_val, gt_val = np.array(emb_val), np.array(gt_val)

    emb_test, gt_test = [], []

    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.float()
            y_test = y_test.long()
            X_test = X_test.to(device)
            emb_test.extend(q_encoder(X_test).cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())

    emb_test, gt_test = np.array(emb_test), np.array(gt_test)

    acc, cm, f1, kappa, gt, pd = task(emb_val, emb_test, gt_val, gt_test)

    q_encoder.train()
    return acc, cm, f1, kappa, gt, pd


def task(X_train, X_test, y_train, y_test):

    cls = LR(solver="lbfgs", multi_class="multinomial", max_iter=2000, n_jobs=-1)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    kappa = cohen_kappa_score(y_test, pred)

    return acc, cm, f1, kappa, y_test, pred


# Pretrain
def Pretext(
    q_encoder,
    optimizer,
    Epoch,
    criterion,
    pretext_loader,
    train_loader,
    test_loader,
    wandb,
):

    q_encoder.train()  # for dropout

    step = 0
    best_acc = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )

    all_loss, acc_score = [], []
    pretext_loss = []

    for epoch in range(Epoch):

        z1, z2 = [], []
        print()

        for index, (anc, pos, neg, _) in enumerate(
            tqdm(pretext_loader, desc="pretrain")
        ):

            anc = anc.float()
            pos = pos.float()
            neg = neg.float()
            anc, pos, neg = (
                anc.to(device),
                pos.to(device),
                neg.to(device),
            )  # (B, 2, 3000)  (B, 7, 2, 3000) (B, 7, 2, 3000)

            anc_features = q_encoder(anc, proj_first=True)  # (B, 128)

            pos_features = []
            neg_features = []

            for i in range(pos.shape[1]):
                pos_features.append(q_encoder(pos[:, i], proj_first=False))  # (B, 128)
                neg_features.append(q_encoder(neg[:, i], proj_first=False))  # (B, 128)

            pos_features = torch.stack(pos_features, dim=1)  # (B, 7, 128)
            neg_features = torch.stack(neg_features, dim=1)  # (B, 7, 128)

            # backprop
            loss = criterion(anc_features, pos_features, neg_features)

            # z1.append(F.normalize(emb_aug1, p=2, dim=1))
            # z2.append(F.normalize(emb_aug2, p=2, dim=1))

            # loss back
            all_loss.append(loss.item())
            pretext_loss.append(loss.cpu().detach().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # only update encoder_q

            N = 1000
            if (step + 1) % N == 0:
                scheduler.step(sum(all_loss[-50:]))
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({"ssl_lr": lr, "epoch": epoch})
            step += 1

        # z1 = torch.std(torch.cat(z1, dim=0), 0, unbiased= False).mean()
        # z2 = torch.std(torch.cat(z2, dim=0), 0, unbiased= False).mean()
        # wandb.log({'first_std': z1, 'epoch': epoch})
        # wandb.log({'second_std': z2, 'epoch': epoch})

        test_acc, _, test_f1, test_kappa, gt, pd = evaluate(
            q_encoder, train_loader, test_loader
        )

        acc_score.append(test_acc)

        wandb.log({"ssl_loss": np.mean(pretext_loss), "epoch": epoch})

        wandb.log({"test_acc": test_acc, "epoch": epoch})
        wandb.log({"test_f1": test_f1, "epoch": epoch})
        wandb.log({"test_kappa": test_kappa, "epoch": epoch})

        if epoch >= 30 and (epoch + 1) % 10 == 0:
            print("Logging confusion matrix ...")
            wandb.log(
                {
                    f"conf_mat_{epoch+1}": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=gt,
                        preds=pd,
                        class_names=["Wake", "N1", "N2", "N3", "REM"],
                    )
                }
            )

        # print the lastest result
        print("epoch: {}".format(epoch))

        if epoch > 5:
            print(
                "recent five epoch, mean: {}, std: {}".format(
                    np.mean(acc_score[-5:]), np.std(acc_score[-5:])
                )
            )
            wandb.log({"accuracy std": np.std(acc_score[-5:]), "epoch": epoch})

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(q_encoder.state_dict(), SAVE_PATH)
                print("save best model on test set")


SAVE_PATH = "multi-epoch.pth"

WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
lr = 5e-4
n_epochs = 50

q_encoder = Net(encoder, dim=n_dim).to(device)

optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
criterion = loss_fn(device).to(device)

#####################################################################################################


# Dataloader
pretext_loader = DataLoader(
    splitted["pretext"],
    batch_size=BATCH_SIZE,
    sampler=pretext_sampler, 
    num_workers=18,
    persistent_workers = True
)
train_loader = DataLoader(
    splitted["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=18, persistent_workers = True
)
test_loader = DataLoader(
    splitted["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=18, persistent_workers = True
)

wandb.init(
    project="multi-epoch",
    notes="triplet loss",
    save_code=True,
    entity="sleep-staging",
    name="test",
)

# optimize
Pretext(
    q_encoder,
    optimizer,
    n_epochs,
    criterion,
    pretext_loader,
    train_loader,
    test_loader,
    wandb,
)
