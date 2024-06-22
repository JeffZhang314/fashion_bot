import torch.nn as nn
import random
from json import load
from PIL import Image
import torch
import torchvision.models as models
import math


class Data():
    
    def __init__(self, path, category_ids, resnet, preprocess, cum_len, batch_size, likes, views, outfit_boundaries):
        self.path = path
        self.category_ids = category_ids
        self.resnet = resnet
        self.preprocess = preprocess
        self.cum_len = cum_len
        self.batch_size = batch_size
        self.likes = likes
        self.views = views
        self.outfit_boundaries = outfit_boundaries

    def prep_data(self): # get resnet output and category id one-hot encoding of whole dataset

        # Freeze all the pre-trained layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        batch = torch.tensor([])

        f = open(self.path + "test_no_dup.json",)
        data = load(f)
        f.close()
        random.Random(0).shuffle(data)

        print(len(data))

        for i in range(math.ceil(len(data)/self.batch_size)):
            print(i)
            inc_batch = self.prep_batch(data[self.batch_size * i:min(self.batch_size * (i + 1), len(data))])
            batch = torch.cat((batch, inc_batch))
            
        return batch, self.outfit_boundaries, self.likes, self.views

    def prep_batch(self, data): # get resnet output and category id one-hot encoding of just data

        #nonlocal category_ids, preprocess, resnet, cum_len, batch_size, likes, views, outfit_boundaries
        #should no longer need this

        batch = torch.empty(0, 3, 224, 224)
        categories = torch.empty(0, 380)
        
        for i in data:

            folder = self.path + "images\\" + i["set_id"] + "\\"
            self.likes = torch.cat((self.likes, torch.tensor([i["likes"]])))
            self.views = torch.cat((self.views, torch.tensor([i["views"]])))

            self.cum_len += len(i["items"])
            self.outfit_boundaries = torch.cat((self.outfit_boundaries, torch.tensor([self.cum_len])))

            j = 0
            while j < len(i["items"]):
                img = Image.open(folder + str(i["items"][j]["index"]) + ".jpg")
                if (img.mode != "RGB"):
                    img = img.convert('RGB')
                batch = torch.cat((batch, self.preprocess(img).unsqueeze(0)))
                img.close()
                one_hot = nn.functional.one_hot(torch.as_tensor(self.category_ids.index(i["items"][j]["categoryid"])), 380).unsqueeze(0)
                categories = torch.cat((categories, one_hot))
                j += 1

        print("before resnet")
        batch = self.resnet(batch).squeeze()
        print("after resnet")
        batch = torch.cat((batch, categories), 1)

        return batch

    def train_val_split(self):
        #return a split of the data into training and validation sets
        pass

    
    
    


