from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import torch
from sklearn.linear_model import LogisticRegression as LR
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

    acc, cm, f1, kappa, gt, pd = task(emb_val, emb_test, gt_val, gt_test)

    q_encoder.train()
    return acc, cm, f1, kappa, gt, pd


def task(X_train, X_test, y_train, y_test):

    cls = LR(solver="lbfgs", multi_class="multinomial", max_iter=300, n_jobs=-1)
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
    device, 
    SAVE_PATH
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
        
        print("Epoch: {}".format(epoch))

        for index, (anc, pos, neg) in enumerate(
            tqdm(pretext_loader, desc="pretrain")
        ):

            anc = anc[:,3].float()
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


        test_acc, _, test_f1, test_kappa, gt, pd = evaluate(
            q_encoder, train_loader, test_loader, device
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
