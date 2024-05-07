import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Keep track of loss values and determine whether to stop model."""
    def __init__(self, 
                 patience:int=10, 
                 delta:int=0, 
                 logger=None) -> None:
        self.patience = patience
        self.delta = delta
        self.logger = logger

        self.best_score = float("inf")
        self.overfit_count = 0

    def stop(self, loss):
        """Update stored values based on new loss and return whether to stop model."""
        threshold = self.best_score + self.delta

        # check if new loss is mode than threshold
        if loss > threshold:
            # increase overfit count and print message
            self.overfit_count += 1
            print_msg = f"Increment early stopper to {self.overfit_count} because val loss ({loss}) is greater than threshold ({threshold})"
            if self.logger:
                self.logger.info(print_msg)
            else:
                print(print_msg)
        else:
            # reset overfit count
            self.overfit_count = 0
        
        # update best_score if new loss is lower
        self.best_score = min(self.best_score, loss)
        
        # check if overfit_count is more than patience set, return value accordingly
        if self.overfit_count >= self.patience:
            return True
        else:
            return False

def train_epoch(epoch, max_epoch, model, criterion, optimizer, dataloader, device, split = 'train', debug=False):
    """Train step within epoch."""
    model.train()
    losses = []
    all_label = []
    all_pred = []
    tbar = tqdm(dataloader)
    width = 6  # Total width including the string length
    formatted_split = split.rjust(width)

    for idx, (inputs, labels) in enumerate(tbar): # 6 ~ 10 s
        # get the inputs and labels
        inputs, labels = inputs.to(device), labels.to(device)
        
        # print('inputs.shape', inputs.shape) # torch.Size([16, 30, 3, 720, 720])
        # print('labels.shape', labels.shape) # torch.Size([16])

        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        # outputs = torch.squeeze(outputs, dim=0)
        # print('outputs.shape', outputs.shape) #  torch.Size([16, 10])
        if isinstance(outputs, list):
            outputs = outputs[0]
        # print('outputs.shape', outputs.shape) # torch.Size([16, 10])

        # compute the loss
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        all_label.extend(labels)
        all_pred.extend(prediction)
        score = accuracy_score(labels.cpu().data.numpy(), prediction.cpu().data.numpy())
        
        # backward & optimize
        loss.backward()
        optimizer.step()

        loginfo = f'{formatted_split} Epoch: {epoch:03d}/{max_epoch:03d}, Iter: {idx:05d}/{idx:05d}, Loss: {loss.item():.4f}, Acc: {100*score:.2f}'
        tbar.set_description(loginfo)

        if debug:
            break

    # Compute the average loss & accuracy
    train_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    train_acc = accuracy_score(all_label.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    return train_loss, train_acc

def val_epoch(epoch, max_epoch, model, criterion, dataloader, device, split = 'val', debug=False):
    """Validation step within epoch."""
    model.eval()
    losses = []
    all_label = []
    all_pred = []
    tbar = tqdm(dataloader)
    width = 6  # Total width including the string length
    formatted_split = split.rjust(width)

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tbar): # 6 ~ 10 s
            # get the inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # compute the loss
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels)
            all_pred.extend(prediction)

            score = accuracy_score(labels.cpu().data.numpy(), prediction.cpu().data.numpy())

            loginfo = f'{formatted_split} Epoch: {epoch:03d}/{max_epoch:03d}, Iter: {idx:05d}/{idx:05d}, Loss: {loss.item():.4f}, Acc: {100*score:.2f}'
            tbar.set_description(loginfo)

            if debug:
                break


                
    # Compute the average loss & accuracy
    val_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    val_acc = accuracy_score(all_label.cpu().data.numpy(), all_pred.cpu().data.numpy())
    
    return val_loss, val_acc

def save_checkpoint(dict_saved, save_dir, is_best = False):
    if is_best:
        torch.save(dict_saved, f"{save_dir}/best_checkpoint.pt")
    else:
        torch.save(dict_saved, f"{save_dir}/checkpoint.pt")    






def test_runtime(model: nn.Module, 
          data_loader: DataLoader, 
          save_weight_dir:str=None, 
          device:str="cuda",
          debug:bool=False,
          max_epoch:int=999):
    """Train function for model."""
    
    criterion = nn.CrossEntropyLoss()
    # move model to device specified
    model.to(device)
    model.eval()

    dict_saved = torch.load(f"{save_weight_dir}/best_checkpoint.pt")
    best_epoch = dict_saved['epoch']
    model.load_state_dict(dict_saved['model_state_dict'])
    

    # test the model
    test_loss, test_acc = val_epoch(best_epoch, max_epoch, model, criterion, data_loader, device, split='test', debug=debug)
    return test_loss, test_acc
