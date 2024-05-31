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



def main():
  
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
