import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import mne
from mne.datasets.sleep_physionet.age import fetch_data

from braindecode.datautil.preprocess import preprocess, Preprocessor
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.preprocess import zscore
from braindecode.datasets import BaseConcatDataset, BaseDataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from sklearn.utils import check_random_state

import xml.etree.ElementTree as ET

PATH = '/scratch/shhs500same/'


# Params
BATCH_SIZE = 1
POS_MIN = 1
NEG_MIN = 15
EPOCH_LEN = 7
NUM_SAMPLES = 500
SUBJECTS = np.arange(0)
RECORDINGS = [1]


##################################################################################################

random_state = 1234
n_jobs = 1
sfreq = 125
high_cut_hz = 30

window_size_s = 30
sfreq = 125
window_size_samples = window_size_s * sfreq


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

        paths = os.listdir(os.path.join(PATH,'data'))
        dict_paths = {}

        for file in paths:
            file = os.path.join(PATH,'data',file)
            if 'edf' in file:
                sid = file.split(".")[0]
                if sid not in dict_paths:
                    dict_paths[sid]=[file]
                else:
                    dict_paths[sid] = [file,dict_paths[sid][0]]
            else:
                sid = file.split(".")[0][:-10]
                if sid in dict_paths:
                    dict_paths[sid].append(file)
                else:
                    dict_paths[sid] = [file]

        paths = [dict_paths[key] for key in list(dict_paths.keys())]
        all_base_ds = list()
        for p in paths:
            raw, desc = self._load_raw(
                p[0],
                p[1],
                preload=preload,
                load_eeg_only=load_eeg_only,
                crop_wake_mins=crop_wake_mins,
                crop=crop
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

        exclude = ['SaO2','H.R.','ECG','EMG','EOG(L)','EOG(R)','THOR RES','ABDO RES','POSITION','LIGHT','NEW AIR','OX stat']
        raw = mne.io.read_raw_edf(raw_fname, preload=preload,exclude=exclude)

        labels = []
        # Read annotation and its header
        t = ET.parse(ann_fname)
        r = t.getroot()
        faulty_File = 0
        for i in range(len(r[4])):
            lbl = int(r[4][i].text)
            if lbl == 4:  # make stages N3, N4 same as N3
                labels.append(3)
            elif lbl == 5:  # Assign label 4 for REM stage
                labels.append(4)
            else:
                labels.append(lbl)
            if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
                faulty_File = 1

        myannot = mne.Annotations(onset=[i*30 for i in range(len(labels))],duration=[30 for i in range(len(labels))],description=labels,orig_time=raw.info['meas_date'])
        raw.set_annotations(myannot, emit_warning=False)
        if crop_wake_mins > 0:
            # Find first and last sleep stages
            sleep_event_inds = np.where(np.array(labels)!=0)[0]

            # Crop raw
            tmin = myannot[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
            tmax = myannot[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))

        # Rename EEG channels
        #ch_names = {i: i.replace("EEG ", "") for i in raw.ch_names if "EEG" in i}
        #raw.rename_channels(ch_names)

        #if not load_eeg_only:
        #    raw.set_channel_types(ch_mapping)

        if crop is not None:
            raw.crop(*crop)

        basename = os.path.basename(raw_fname)
        basename = basename.split(".")[0].split("-")[-1]
        subj_nb = int(basename)
        sess_nb = 1
        desc = pd.Series({"subject": subj_nb, "recording": [sess_nb]}, name="")

        return raw, desc


dataset = SleepPhysionet(
    subject_ids=SUBJECTS, recording_ids=RECORDINGS, crop_wake_mins=30
)


preprocessors = [
    Preprocessor(lambda x: x * 1e6),
    Preprocessor("filter", l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs),
]

# Transform the data
preprocess(dataset, preprocessors)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    preload= True,
    accepted_bads_ratio= 0.0009699321047526673
    )
breakpoint()
preprocess(windows_dataset, [Preprocessor(zscore)])

###################################################################################################################################
""" Subject sampling """

#rng = np.random.RandomState(1234)
#
#NUM_WORKERS = 0 if n_jobs <= 1 else n_jobs
#PERSIST = False if NUM_WORKERS <= 1 else True
#
#
#subjects = np.unique(windows_dataset.description["subject"])
#sub_pretext = rng.choice(subjects, 3, replace=False)
#sub_train = sorted(
#    rng.choice(sorted(list(set(subjects) - set(sub_pretext))), 10, replace=False)
#)
#sub_test = sorted(list(set(subjects) - set(sub_pretext) - set(sub_train)))
#
#
#print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#print(f"Pretext: {sub_pretext} \n")
#print(f"Train: {sub_train} \n")
#print(f"Test: {sub_test} \n")
#print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#
#
########################################################################################################################################
#
#
#class RelativePositioningDataset(BaseConcatDataset):
#    """BaseConcatDataset with __getitem__ that expects 2 indices and a target."""
#
#    def __init__(self, list_of_ds, epoch_len=7):
#        super().__init__(list_of_ds)
#        self.return_pair = True
#        self.epoch_len = epoch_len
#
#    def __getitem__(self, index):
#
#        pos, neg = index
#        pos_data = []
#        neg_data = []
#
#        assert pos != neg, "pos and neg should not be the same"
#
#        for i in range(-(self.epoch_len // 2), self.epoch_len // 2 + 1):
#            pos_data.append(super().__getitem__(pos + i)[0])
#            neg_data.append(super().__getitem__(neg + i)[0])
#
#        pos_data = np.stack(pos_data, axis=0) # (7, 2, 3000)
#        neg_data = np.stack(neg_data, axis=0) # (7, 2, 3000)
#
#        return pos_data, neg_data
#
#
#class TuneDataset(BaseConcatDataset):
#    """BaseConcatDataset for train and test"""
#
#    def __init__(self, list_of_ds):
#        super().__init__(list_of_ds)
#
#    def __getitem__(self, index):
#
#        X = super().__getitem__(index)[0]
#        y = super().__getitem__(index)[1]
#
#        return X, y
#
#
#class RecordingSampler(Sampler):
#    def __init__(self, metadata, random_state=None, epoch_len=7):
#
#        self.metadata = metadata
#        self._init_info()
#        self.rng = check_random_state(random_state)
#        self.epoch_len = epoch_len
#
#    def _init_info(self):
#        keys = ["subject", "recording"]
#
#        self.metadata = self.metadata.reset_index().rename(
#            columns={"index": "window_index"}
#        )
#        self.info = (
#            self.metadata.reset_index()
#            .groupby(keys)[["index", "i_start_in_trial"]]
#            .agg(["unique"])
#        )
#        self.info.columns = self.info.columns.get_level_values(0)
#
#    def sample_recording(self):
#        """Return a random recording index."""
#        return self.rng.choice(self.n_recordings)
#
#    def sample_window(self, rec_ind=None):
#        """Return a specific window."""
#        if rec_ind is None:
#            rec_ind = self.sample_recording()
#        win_ind = self.rng.choice(
#            self.info.iloc[rec_ind]["index"][self.epoch_len // 2 : -self.epoch_len // 2]
#        )
#        return win_ind, rec_ind
#
#    def __iter__(self):
#        raise NotImplementedError
#
#    @property
#    def n_recordings(self):
#        return self.info.shape[0]
#
#
#class RelativePositioningSampler(RecordingSampler):
#    def __init__(
#        self,
#        metadata,
#        tau_pos,
#        tau_neg,
#        n_examples,
#        same_rec_neg=True,
#        random_state=None,
#        epoch_len=7,
#    ):
#        super().__init__(metadata, random_state=random_state, epoch_len=epoch_len)
#
#        self.tau_pos = tau_pos
#        self.tau_neg = tau_neg
#        self.epoch_len = epoch_len
#        self.n_examples = n_examples
#        self.same_rec_neg = same_rec_neg
#
#    def _sample_pair(self):
#        
#        """Sample a pair of two windows."""
#        # Sample first window
#        win_ind1, rec_ind1 = self.sample_window()
#        
#        ts1 = self.metadata.iloc[win_ind1]["i_start_in_trial"]
#        ts = self.info.iloc[rec_ind1]["i_start_in_trial"]
#
#        epoch_min = self.info.iloc[rec_ind1]["i_start_in_trial"][self.epoch_len // 2]
#        epoch_max = self.info.iloc[rec_ind1]["i_start_in_trial"][-self.epoch_len // 2]
#        
#        rng = np.random.Random
#        if self.same_rec_neg:
#            mask = ((ts <= ts1 - self.tau_neg) & (ts >= epoch_min)) | (
#                (ts >= ts1 + self.tau_neg) & (ts <= epoch_max)
#            )
#
#        if sum(mask) == 0:
#            raise NotImplementedError
#        win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]["index"][mask])
#        
#        return win_ind1, win_ind2
#
#    def __iter__(self):
#
#        for i in range(self.n_examples):
#
#            yield self._sample_pair()
#
#    def __len__(self):
#        return self.n_examples
#    
#    
#######################################################################################################################
#
#
#PRETEXT_PATH = os.path.join(PATH, "pretext")
#TRAIN_PATH = os.path.join(PATH, "train")
#TEST_PATH = os.path.join(PATH, "test")
#
#if not os.path.exists(PRETEXT_PATH): os.mkdir(PRETEXT_PATH)
#if not os.path.exists(TRAIN_PATH): os.mkdir(TRAIN_PATH)
#if not os.path.exists(TEST_PATH): os.mkdir(TEST_PATH)
#
#
#splitted = dict()
#
#splitted["pretext"] = RelativePositioningDataset(
#    [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_pretext],
#    epoch_len = EPOCH_LEN
#)
#
#splitted["train"] = TuneDataset(
#    [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_train]
#)
#
#splitted["test"] = TuneDataset(
#    [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_test]
#)
#
#
#
## Sampler
#tau_pos, tau_neg = int(sfreq * POS_MIN * 60), int(sfreq * NEG_MIN * 60)
#
#n_examples_pretext = NUM_SAMPLES * len(splitted["pretext"].datasets)
#
#print(f'Number of pretext subjects: {len(splitted["pretext"].datasets)}')
#print(f'Number of pretext epochs: {n_examples_pretext}')
#
#pretext_sampler = RelativePositioningSampler(
#    splitted["pretext"].get_metadata(),
#    tau_pos=tau_pos,
#    tau_neg=tau_neg,
#    n_examples=n_examples_pretext,
#    same_rec_neg=True,
#    random_state=random_state  # Same samples for every iteration of dataloader
#)
#
#
## Dataloader
#pretext_loader = DataLoader(
#    splitted["pretext"],
#    batch_size=BATCH_SIZE,
#    sampler=pretext_sampler
#)
#
#train_loader = DataLoader(
#    splitted["train"], batch_size=BATCH_SIZE, shuffle= False
#)
#
#test_loader = DataLoader(
#    splitted["test"], batch_size=BATCH_SIZE, shuffle=False
#    )

#for i, arr in tqdm(enumerate(pretext_loader), desc = 'pretext'):
#    temp_path = os.path.join(PRETEXT_PATH, str(i) + '.npz')
#    np.savez(temp_path, pos = arr[0].numpy().squeeze(0), neg = arr[1].numpy().squeeze(0))
#  
#for i, arr in tqdm(enumerate(train_loader), desc = 'train'):
#    temp_path = os.path.join(TRAIN_PATH, str(i) + '.npz')
#    np.savez(temp_path, x = arr[0].numpy().squeeze(0), y = arr[1].numpy().squeeze(0))
#    
#for i, arr in tqdm(enumerate(test_loader), desc = 'test'):
#    temp_path = os.path.join(TEST_PATH, str(i) + '.npz')
#    np.savez(temp_path, x = arr[0].numpy().squeeze(0), y = arr[1].numpy().squeeze(0))
#    
