import torch
from tqdm import tqdm
import numpy as np
import copy
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1, cohen_kappa
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn


# For train and pretrain
def evaluate_loop(q_encoder, train_loader, test_loader, wandb, EPOCH):
            
    class eval_class(pl.LightningModule):
        
        def __init__(self, q_encoder, n_dim = 128):
            super(eval_class, self).__init__()
            
            self.n_dim = n_dim
            self.encoder = copy.deepcopy(q_encoder)
            self.fc1 = nn.Linear(self.n_dim, 5)
            
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        def forward(self, x):
            x = self.encoder(x)
            x = self.fc1(x)
            return x
    
     
    class sleep_ft(pl.LightningModule):
        
        def __init__(self, eval_model, train_dl,valid_dl,wandb, EPOCH):
            super(sleep_ft,self).__init__()
            
            self.model = eval_model
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.learning_rate = 0.0003
            self.weight_decay = 3e-5
            self.batch_size = 256
            self.wandb = wandb
            self.criterion = nn.CrossEntropyLoss()
            self.train_dl = train_dl
            self.valid_dl = valid_dl
            
            self.best_score = 0
            self.EPOCH = EPOCH

        def configure_optimizers(self):
            
            optimizer = torch.optim.Adam(self.parameters(), self.learning_rate, betas=(self.beta1,self.beta2), weight_decay=self.weight_decay)
            self.scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=25, factor=0.5, threshold=5e-3)     
            return optimizer

        def train_dataloader(self):
            return self.train_dl
        
        def val_dataloader(self):
            return self.valid_dl
        
        def training_step(self,batch,batch_idx):
            
            data, y = batch
            outs = self.model(data)
            loss = self.criterion(outs, y.long())
            batch_acc = accuracy(outs,y)
            
            return {'loss':loss,'acc':batch_acc,'preds':outs.detach(),'target':y.detach()}
        
        def validation_step(self,batch,batch_idx):
            
            data, y = batch
            outs = self.model(data)
            loss = self.criterion(outs, y.long())
            batch_acc = accuracy(outs,y)

            return {'loss':loss,'acc':batch_acc,'preds':outs.detach(),'target':y.detach()} 

        def training_epoch_end(self,outputs):
            
            epoch_preds = torch.vstack([x['preds'] for x in outputs])
            epoch_targets = torch.hstack([x['target'] for x in outputs])
            epoch_loss = torch.hstack([x['loss'] for x in outputs]).mean()
            epoch_acc = torch.hstack([x['acc'] for x in outputs]).mean()
            
            # self.train_dict = {"Train F1":f1(epoch_preds,epoch_targets,average='macro',num_classes=5),                                    
            #                 "Train Kappa":cohen_kappa(epoch_preds,epoch_targets,num_classes=5),
            #                 "Train Acc":epoch_acc,
            #                 "Train Loss":epoch_loss,
            #                 "Epoch":self.EPOCH}


        def validation_epoch_end(self,outputs):
            
            epoch_preds = torch.vstack([x['preds'] for x in outputs])
            epoch_targets = torch.hstack([x['target'] for x in outputs])
            epoch_loss = torch.hstack([x['loss'] for x in outputs]).mean()
            epoch_acc = torch.hstack([x['acc'] for x in outputs]).mean()
            # class_preds = epoch_preds.cpu().detach().argmax(dim=1)
            
            f1_score = f1(epoch_preds,epoch_targets,average='macro',num_classes=5)
            
            if f1_score > self.best_score:
                self.best_score = f1_score
                
                print(f'Validation F1: {f1_score}')
                print(f'best_score: {self.best_score}')
         
                self.val_dict = {"Valid F1": f1_score,
                            "Valid Kappa":cohen_kappa(epoch_preds,epoch_targets,num_classes=5),
                            "Valid Acc":epoch_acc,
                            "Valid Loss":epoch_loss,
                            "Epoch":self.EPOCH}
                      
            self.log('val_loss',epoch_loss)  
            self.scheduler.step(epoch_loss)
            
        def on_fit_end(self):
            self.wandb.log(self.val_dict)
            # self.wandb.log(self.train_dict)
            

    pl.seed_everything(1234, workers = True)

    eval_model = eval_class(q_encoder)
    NUM_EPOCHS = 100
       
    eval_finetune = sleep_ft(eval_model, train_loader, test_loader, wandb, EPOCH)       
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience= 15, mode="min", verbose=True)
    le_trainer = pl.Trainer(callbacks=[early_stop_callback], enable_checkpointing=False, max_epochs= NUM_EPOCHS, gpus=1, num_sanity_val_steps=0)
    le_trainer.fit(eval_finetune)


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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )

    all_loss, acc_score = [], []
    pretext_loss = []

    for epoch in range(Epoch):
        
        print("epoch: {}".format(epoch))

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

        wandb.log({"ssl_loss": np.mean(pretext_loss), "Epoch": epoch})
        
        # Run train and test
        if (epoch+1) % 5 == 0:
            evaluate_loop(q_encoder, train_loader, test_loader, wandb, epoch)
        
