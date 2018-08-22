# Import Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchvision.models.resnet import resnet18

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt

from torch.nn import functional as F

from PIL import Image



import copy

class BasicModel():
    def __init__(self, model, data_train, data_test, error, weight_decay=5e-4):
        self.data_train = data_train
        self.data_test = data_test
        self.model = copy.deepcopy(model)
        self.weight_decay = weight_decay
        self.epoch = 0
        self.error = error
        
    def train(self, number_of_cycles, lr=0.1, momentum=0.9, lr_find=False, cycle_len=1, sched_lr=None, sched_mom=None): 
        self.states = {'loss':[],
                       'train_loss': [],
                       'train_cycle_loss': [],
                        'valid_loss': [],
                        'train_acc': [],
                        'train_cycle_acc': [],
                        'valid_cycle_acc': [],
                        'lr': [],
                        'n_iter': [],
                        'n_cycle': []}

        iter_count = 0
        accuracy = 0
        best_model = 0
        best_loss = 10
        accuracy_test = np.nan
        stop = False
        
        # model
        if lr_find:
            model = copy.deepcopy(self.model)  
            self.epoch_save = self.epoch
        else:
            model = self.model
        
        if sched_lr is None:
            def sched_lr(n_iter):
                return lr
        
        if sched_mom is None:
            def sched_mom(n_iter):
                return momentum 
        
        # training
        for cycle in range(0, number_of_cycles):
            iter_cycle = 0
            if stop==True:
                break
                
            for n_cycle in range(0, cycle_len): 
                train_correct = 0
                train_total = 0
                train_loss = 0
                
                if stop==True:
                    break
            
                for iter_in_epoch, (inputs, targets) in enumerate(self.data_train):
                    inputs, targets  = inputs.cuda(), targets.cuda()

                    #optimizer = torch.optim.SGD(model.parameters(), lr=scheduler(cycle), momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
                    #lr = sched_lr(iter_count*np.power(0.9995, iter_count))/0.28478163764431206
                    #lr = sched_lr(iter_count*np.power(0.9995, iter_count))/0.1262922982758279
                    lr = sched_lr(iter_count)
                    mom = sched_mom(iter_count)
                    
                    if lr > 0:
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=self.weight_decay, nesterov=True)
                    else:
                        stop=True
                        break
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1) 
                    loss = self.error(outputs, targets)

                    loss.backward()
                    optimizer.step()
                        
                    train_correct += torch.sum(preds == targets.data)
                    train_loss += float(loss)
                    train_total += len(inputs)
        
                    self._save_iter_state(iter_count, loss, train_loss, train_correct, train_total, lr)

                    iter_count += 1
                    iter_cycle += 1

                    
                    if loss < best_loss:
                        best_loss = float(loss)
                        
                        if accuracy_test > 91:
                            accuracy_test = self._accuracy(model)
                            if accuracy_test > 93:
                                self.save_model('save_ep'+str(self.epoch)+"_acc"+str(int(accuracy_test))+".pth", to_google_drive=True)
                                best_model = accuracy_test
                            
                    if (lr_find)&(float(loss) > 4*best_loss):
                        stop=True
                        break
                            
                accuracy_test = self._accuracy(model)
                
                print('Epochs: {} Cycle: {} Iter: {}  Tr Loss: {}  Tr Acc: {} Test Acc: {} Last lr: {}'.format(self.epoch, cycle, iter_count, float(train_loss)/train_total, int(100*float(train_correct)/train_total), accuracy_test, lr))
                
                if accuracy_test > 90:
                    if accuracy_test >= int(best_model)+1:
                        self.save_model('save_ep'+str(self.epoch)+"_acc"+str(int(accuracy_test))+".pth", to_google_drive=True)
                        best_model = accuracy_test
                
                self.epoch += 1
                
                if not lr_find:
                    self.model = model
                else:
                    self.epoch = self.epoch_save
            

            
    
    def _save_iter_state(self, iter_count, loss, train_loss,  train_correct,  train_total, lr):            
        self.states['n_iter'].append(iter_count)
        self.states['loss'].append(float(loss))
        self.states['train_loss'].append(float(train_loss)/train_total)
        self.states['train_acc'].append(float(train_correct)/train_total)
        self.states['lr'].append(lr)
        
        
    def _accuracy(self, model, data='test'):
        # Calculate Accuracy         
        correct = 0
        total = 0
              
        # Iterate through test dataset
        if data=='test': data_acc = self.data_test
        else:    data_acc = self.data_train
        
        for inputs, targets in data_acc:
            inputs = inputs.cuda()
            targets = targets.cuda()

            # Forward propagation
            outputs = model(inputs)

            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]

            # Total number of labels
            total += len(targets)

            #correct += (predicted == targets).sum()
            correct += predicted.eq(targets).sum().item()

        accuracy = float(100 * correct) / total

        return accuracy
            
  
    def lr_find(self, lr, momentum=0.9, cycle_len=1, plot=True, smooth=None, sched_mom=None):
        if (type(lr) != list): 
            raise ValueError('lr must be a list with the min and max lr')
        
        sched = sch.CyclicLR(base_lr=lr[0], max_lr=lr[1], iter_by_epoch=len(self.data_train), cycle_len=2)
        
        self.train(1, cycle_len=cycle_len, momentum=momentum, lr_find=True, sched_lr=sched, sched_mom=sched_mom)
        
        if plot==True:
            self.lr_plot(smooth=smooth)

    def lr_plot(self, smooth=None):
        if smooth is not None:
            loss_list = [np.mean(self.states['train_loss'][k:k+smooth]) for k in range(0, len(self.states['train_loss'])-smooth)]
            lr_list = self.states['lr'][:-smooth]
        else:
            loss_list = self.states['train_loss']
            lr_list = self.states['lr']
              
        plt.plot(lr_list, loss_list, color = "red")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.title("Lr find")
        plt.xscale('log')
        plt.grid(True,which="both")
        plt.show()
        
        plt.plot(self.states['lr'], self.states['train_acc'], color = "blue")
        plt.xlabel("Learning rate")
        plt.ylabel("Train accuracy")
        plt.title("Lr find")
        plt.xscale('log')
        plt.grid(True,which="both")
        plt.show()


    def plot(self):
        plt.plot(self.iteration_list, self.loss_list, color = "red")
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Training")
        plt.grid(True,which="both")
        plt.show()
   
    def save_model(self, filename, to_google_drive=False):
        import sys
        sys.path.append('Utils')
        import colab_utils as utils

        torch.save(self.model.state_dict(), filename)
        if to_google_drive:     utils.upload(filename)
      
    def load_model(self, filename, from_google_drive=False):
        import sys
        sys.path.append('Utils')
        import colab_utils as utils

        if from_google_drive:     utils.download(filename)
        self.model.load_state_dict(torch.load(filename))