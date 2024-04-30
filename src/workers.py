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

def _train_epoch(epoch, max_epoch, model, criterion, optimizer, dataloader, device, split = 'train', debug=False):
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

def _val_epoch(epoch, max_epoch, model, criterion, dataloader, device, split = 'val', debug=False):
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


def trainval(model: nn.Module, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          max_epoch:int, 
          logger,
          writer,
          save_weight_dir:str=None, 
          device:str="cuda", 
          patience:int=10, 
          optimizer_lr:int=0.001, 
          weight_decay:int=0, 
          use_scheduler:bool=False,
          resume:str=None,
          fine_tune:bool=False,
          debug:bool=False,
          test_loader: DataLoader=None):
    """Train function for model."""
    
    # if save_dir is specified and does not exist, make save_dir directory
    if save_weight_dir and not os.path.exists(save_weight_dir):
        os.mkdir(save_weight_dir)
    
    # move model to device specified
    model.to(device)

    # initialise loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//3)

    start_epoch = 0
    best_val_acc = float("-inf")
    best_epoch = None

    if resume is not None:
        # load model from checkpoint
        checkpoint = torch.load(resume)
        if not fine_tune:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']
    
    logger.info("Training Started".center(60, '#'))

    early_stopper = EarlyStopping(patience=patience, logger=logger)

    # start training
    for epoch in range(start_epoch, max_epoch+1):
        #logger.info(f"Epoch {epoch}")
        
        # train the model
        train_loss, train_acc = _train_epoch(epoch, max_epoch, model, criterion, optimizer, train_loader, device, debug=debug)

        # write train loss and acc to logger
        writer.add_scalars('Loss', {'train': train_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc}, epoch)
        logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch, train_loss, train_acc*100))

        # validate the model
        val_loss, val_acc = _val_epoch(epoch, max_epoch, model, criterion, val_loader, device, split='val', debug=debug)

        # write val loss and acc to logger
        writer.add_scalars('Loss', {'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'val': val_acc}, epoch)
        logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch, val_loss, val_acc*100))

        
        # step scheduler
        if use_scheduler:
            scheduler.step(val_loss)
        
        if not test_loader is None and epoch % 20 == 0:
            test_loss, test_acc = _val_epoch(epoch, max_epoch, model, criterion, test_loader, device, split='test', debug=debug)
            logger.info("Average Test Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch, test_loss, test_acc*100))

        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_epoch': best_epoch

        }
        # save model or checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(checkpoint_dict, save_weight_dir, is_best=True)
            logger.info("Best Epoch: {} | Validation Loss: {:.6f} | Acc: {:.2f}%".format(best_epoch, val_loss, val_acc*100))
        else:
            save_checkpoint(checkpoint_dict, save_weight_dir)
        
        # update and check early stopper
        #if early_stopper.stop(val_loss):
        #    logger.info("Model has overfit, early stopping...")
        #    break
        
        logger.info("")
        if debug:
            break
        
    logger.info("Training Finished".center(60, '#'))
    logger.info("")
    logger.info("")

    return model



def test(model: nn.Module, 
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
    test_loss, test_acc = _val_epoch(best_epoch, max_epoch, model, criterion, data_loader, device, split='test', debug=debug)
    return test_loss, test_acc
