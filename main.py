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
  
  path = "clothing-coordination-dataset/" #variables and stuff
  category_ids = [
          2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 343, 67, 68, 69, 71, 74, 75, 76, 77, 78, 85, 93, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 106, 107, 108, 110, 113, 115, 116, 118, 120, 122, 123, 124, 126, 127, 129, 130, 132, 135, 136, 139, 140, 141, 143, 144, 4241, 4242, 147, 4244, 150, 4247, 4248, 153, 154, 155, 157, 4254, 159, 160, 4257, 162, 163, 164, 166, 167, 168, 169, 170, 4267, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 4292, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 231, 236, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 270, 271, 272, 273, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 318, 319, 320, 321, 4426, 4428, 333, 334, 335, 4432, 4433, 338, 339, 340, 4437, 342, 4439, 4440, 4441, 4442, 4443, 4445, 4446, 4447, 4448, 4449, 4450, 4451, 4452, 4454, 4455, 4456, 4457, 4458, 4459, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4468, 4470, 4472, 4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489, 4490, 4492, 4493, 4495, 4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 4512, 4513, 4514, 4516, 4517, 4518, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4438, 4240, 146, 148, 4246, 151, 152, 949, 161, 4258, 171, 4269, 4276, 196, 3336, 1605, 1606, 1607, 1967, 332, 4429, 4430, 4431, 336, 337, 4434, 4435, 4436, 341
      ]
          
  weights = models.ResNet152_Weights.IMAGENET1K_V2
  resnet = models.resnet152(weights = weights)

  preprocess = weights.transforms()
  likes = torch.empty(0)
  views = torch.empty(0)
  outfit_boundaries = torch.empty(0)

  cum_len = 0
  torch.manual_seed(0)
  batch_size = 64

  #data preparation
  myData = Data(path, category_ids, resnet, preprocess, cum_len, batch_size, likes, views, outfit_boundaries)
  annotated_batch, outfit_boundaries, likes, views = myData.prep_data()
  

  optimizer = optim.Adam(model.parameters(), lr = n_warmup_steps ** -0.5, betas = (0.9, 0.98), eps = 1e-09)

  print(annotated_batch)

  torch.save(annotated_batch, prefix + 'tensor.pt')
  print("done")

  loaded_batch, loaded_outfit_boundaries, loaded_likes, loaded_views = torch.load(path + 'tensor.pt')
  #this runs through the resnet layer, 

  #main model ResnetToTransformer Layer -> Transformer Layer -> TransformerToOutput Layer
  model = Model()
  run_model = runModel(model, optimizer, args)
  

