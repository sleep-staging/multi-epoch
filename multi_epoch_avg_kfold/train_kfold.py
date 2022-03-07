from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)
import torch
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
from utils import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader, Dataset
from braindecode.util import set_random_seeds

random_state = 1234
# Seeds
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False    
set_random_seeds(seed=random_state, cuda=device == "cuda")


# Train, test
def evaluate(q_encoder, train_loader, test_loader, device):

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

    acc, cm, f1, kappa, bal_acc, gt, pd = task(emb_val, emb_test, gt_val, gt_test)

    q_encoder.train()
    return acc, cm, f1, kappa, bal_acc, gt, pd


def task(X_train, X_test, y_train, y_test):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    cls = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', multi_class='multinomial', random_state=1234, n_jobs=-1, max_iter = 2000)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    kappa = cohen_kappa_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)

    return acc, cm, f1, kappa, bal_acc, y_test, pred


def kfold_evaluate(q_encoder, test_subjects, device, BATCH_SIZE):

    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

    total_acc, total_f1, total_kappa, total_bal_acc = [], [], [], []
    i = 1

    for train_idx, test_idx in kfold.split(test_subjects):

        test_subjects_train = [test_subjects[i] for i in train_idx]
        test_subjects_test = [test_subjects[i] for i in test_idx]
        test_subjects_train = [rec for sub in test_subjects_train for rec in sub]
        test_subjects_test = [rec for sub in test_subjects_test for rec in sub]

        train_loader = DataLoader(TuneDataset(test_subjects_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
        test_loader = DataLoader(TuneDataset(test_subjects_test), batch_size=BATCH_SIZE, shuffle= False, num_workers=4, persistent_workers=True)
        test_acc, _, test_f1, test_kappa, bal_acc, gt, pd = evaluate(q_encoder, train_loader, test_loader, device)

        total_acc.append(test_acc)
        total_f1.append(test_f1)
        total_kappa.append(test_kappa)
        total_bal_acc.append(bal_acc)
        
        print("+"*50)
        print(f"Fold{i} acc: {test_acc}")
        print(f"Fold{i} f1: {test_f1}")
        print(f"Fold{i} kappa: {test_kappa}")
        print(f"Fold{i} bal_acc: {bal_acc}")
        print("+"*50)
        i+=1 

    return np.mean(total_acc), np.mean(total_f1), np.mean(total_kappa), np.mean(total_bal_acc)

class TuneDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, subjects):
        self.subjects = subjects
        self._add_subjects()

    def __getitem__(self, index):

        X = self.X[index]
        y =  self.y[index]
        return X, y

    def __len__(self):
        return self.X.shape[0]
        
    def _add_subjects(self):
        self.X = []
        self.y = []
        for subject in self.subjects:
            self.X.append(subject['windows'])
            self.y.append(subject['y'])
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)



##################################################################################################################################################



# Pretrain
def Pretext(
    q_encoder,
    optimizer,
    Epoch,
    criterion,
    pretext_loader,
    test_subjects,
    wandb,
    device, 
    SAVE_PATH,
    BATCH_SIZE
):

    q_encoder.train()  # for dropout

    step = 0
    best_f1 = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )
    # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps = 500, cycle_mult = 1.2, max_lr = 0.01, min_lr = 5e-6, warmup_steps = 100, gamma = 0.5)

    all_loss, acc_score = [], []
    pretext_loss = []

    for epoch in range(Epoch):

        print('=========================================================\n')
        print("Epoch: {}".format(epoch))
        print('=========================================================\n')
        
        for index, (aug1, aug2, neg) in enumerate(
            tqdm(pretext_loader, desc="pretrain")
        ):

            aug1 = aug1.float()
            aug2 = aug2.float()
            neg = neg.float()
            aug1, aug2, neg = (
                aug1.to(device),
                aug2.to(device),
                neg.to(device),
            )  # (B, 7, 2, 3000)  (B, 7, 2, 3000) (B, 7, 2, 3000)
            
            num_len = aug1.shape[1]

            pos1_features = []
            pos2_features = []
            neg_features = []
        
            anc1_features = q_encoder(aug1[:, num_len // 2], proj_first=True) #(B, 128)
            anc2_features = q_encoder(aug2[:, num_len // 2], proj_first=True) #(B, 128)
            
            for i in range(num_len):
                pos1_features.append(q_encoder(aug2[:, i], proj_first=False))  # (B, 128)
                pos2_features.append(q_encoder(aug1[:, i], proj_first=False))  # (B, 128)
                neg_features.append(q_encoder(neg[:, i], proj_first=False))  # (B, 128)

            pos1_features = torch.stack(pos1_features, dim=1)  # (B, 7, 128)
            pos2_features = torch.stack(pos2_features, dim=1)  # (B, 7, 128)
            neg_features = torch.stack(neg_features, dim=1)  # (B, 7, 128)
           
            # backprop
            loss1 = criterion(anc1_features, pos1_features, neg_features)
            loss2 = criterion(anc2_features, pos2_features, neg_features)
            loss = loss1 + loss2

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
                wandb.log({"ssl_lr": lr, "Epoch": epoch})
            step += 1
            # scheduler.step()
            # lr = optimizer.param_groups[0]["lr"]
            # wandb.log({"ssl_lr": lr, "Epoch": epoch})

        if epoch >= 60 and (epoch) % 5 == 0:

            test_acc, test_f1, test_kappa, bal_acc = kfold_evaluate(
                q_encoder, test_subjects, device, BATCH_SIZE
            )

            wandb.log({"ssl_loss": np.mean(pretext_loss), "Epoch": epoch})

            wandb.log({"Valid Acc": test_acc, "Epoch": epoch})
            wandb.log({"Valid F1": test_f1, "Epoch": epoch})
            wandb.log({"Valid Kappa": test_kappa, "Epoch": epoch})
            wandb.log({"Valid Balanced Acc": bal_acc, "Epoch": epoch})

            if test_f1 > best_f1:   
                best_f1 = test_f1
                torch.save(q_encoder.state_dict(), SAVE_PATH)
                wandb.save(SAVE_PATH)
                print("save best model on test set with best F1 score")
