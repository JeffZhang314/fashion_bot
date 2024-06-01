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
        
        self.model = model
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.optimizer = optimizer
        
    def train(self, train, val, criterion=nn.MSELoss(), epochs=512):#TBDDDD
        
        
        for epoch in range(epochs):  # loop over whole dataset
            total_train_loss = 0.0
            
            

        n_warmup_steps = 20

        

        scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 512 ** -0.5 * n_warmup_steps ** -1.5 * epoch)
        scheduler2 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 512 ** -0.5 * (epoch + n_warmup_steps) ** -0.5)

        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers = [scheduler1, scheduler2], milestones = [n_warmup_steps])

        for i in range(n_warmup_steps * 21):
            y_pred = model(valid_batch, valid_outfit_boundaries)
            loss = loss_fn(y_pred, valid_labels)
            print(i)
            print(loss)
            print(scheduler.get_last_lr())
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            scheduler.step()