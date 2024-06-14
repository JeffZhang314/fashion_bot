import torch.nn as nn
from random import shuffle
from json import load
from PIL import Image
import torchvision.models as models
import torch
from torch.nested import nested_tensor
from torch.optim import AdamW
import pandas as pd
from matplotlib import pyplot as plt
import PIL
import numpy as np
import random
import torch.optim as optim
import math


class runModel():
    """
    Class that scafolds the training and evaluation methods and attributes
    for each test case (Test case: Adam, Test case: Lookahead(Adam)).
    """
    def __init__(self, model, optimizer, args) -> None:
        
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.model = model
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.optimizer = optimizer
        
    def train(self, train, val, criterion=nn.MSELoss(), epochs=512):#TBDDDD
        
        
        #scuffed and not done 
        for epoch in range(epochs):  # loop over whole dataset
            total_train_loss = 0.0
            
        
            n_warmup_steps = 20

            scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 512 ** -0.5 * n_warmup_steps ** -1.5 * epoch)
            scheduler2 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 512 ** -0.5 * (epoch + n_warmup_steps) ** -0.5)

            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers = [scheduler1, scheduler2], milestones = [n_warmup_steps])

            if torch.cuda.is_available():
                train = train.cuda()
                val = val.cuda()
            
            
            for i in range(n_warmup_steps * 21):
                out = model(train)
                loss = loss_fn(out, valid_labels)
                #not sure the labels situation from original code
                print(i)
                print(loss)
                print(scheduler.get_last_lr())
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                scheduler.step()
            
    def test(self, test):
        pass
    
    def get_train_loss(self):
        return self.train_loss
    
    def get_val_loss(self):
        return self.val_loss
    
    def get_train_acc(self):
        #train_acc = curr_acc = correct / total
        return self.train_acc    