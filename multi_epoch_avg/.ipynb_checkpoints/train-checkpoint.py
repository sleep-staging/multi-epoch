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
from tqdm import tqdm
import numpy as np



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
    device, 
    SAVE_PATH
):

    q_encoder.train()  # for dropout

    step = 0
    best_f1 = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )

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

        if epoch >= 70 and epoch % 5 == 0: 
            
            test_acc, _, test_f1, test_kappa, bal_acc, gt, pd = evaluate(
                q_encoder, train_loader, test_loader, device
            )

            wandb.log({"ssl_loss": np.mean(pretext_loss), "Epoch": epoch})

            wandb.log({"Valid Acc": test_acc, "Epoch": epoch})
            wandb.log({"Valid F1": test_f1, "Epoch": epoch})
            wandb.log({"Valid Kappa": test_kappa, "Epoch": epoch})
            wandb.log({"Valid Balanced Acc": bal_acc, "Epoch": epoch})

        # if epoch >= 30 and (epoch + 1) % 10 == 0:
        #     print("Logging confusion matrix ...")
        #     wandb.log(
        #         {
        #             f"conf_mat_{epoch}": wandb.plot.confusion_matrix(
        #                 probs=None, 
        #                 y_true=gt,
        #                 preds=pd,
        #                 class_names=["Wake", "N1", "N2", "N3", "REM"],
        #             )
        #         }
        #     )
        

        if epoch > 5:
            if test_f1 > best_f1:
                best_f1 = test_f1
                torch.save(q_encoder.state_dict(), SAVE_PATH)
                wandb.save(SAVE_PATH)
                print("save best model on test set with best F1 score")
