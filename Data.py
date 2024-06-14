import torch.nn as nn
from random import shuffle
from json import load
from PIL import Image
import torchvision.models as models
import torch


class data():
    
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
    
    def prep_data(): # get resnet output and category id one-hot encoding of whole dataset
        
        # Freeze all the pre-trained layers
        for param in resnet.parameters():
            param.requires_grad = False
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        batch = torch.tensor([])

        for i in range(2):
            inc_batch = prep_batch(data[batch_size * i:min(batch_size * (i + 1), len(data))])
            batch = torch.cat((batch, inc_batch))
            
        return batch, outfit_boundaries, likes, views

    def prep_batch(data): # get resnet output and category id one-hot encoding of just data

        #nonlocal category_ids, preprocess, resnet, cum_len, batch_size, likes, views, outfit_boundaries
        #should no longer need this

        batch = torch.empty(0, 3, 224, 224)
        categories = torch.empty(0, 380)
        n_processed = 0

        f = open("valid_no_dup.json",)
        data = load(f)
        random.Random(0).shuffle(data)
        f.close()
        
        for i in data:
            print(n_processed)
            n_processed += 1

            folder = path + i["set_id"] + "/"
            likes = torch.cat((likes, torch.tensor([i["likes"]])))
            views = torch.cat((views, torch.tensor([i["views"]])))

            cum_len += len(i["items"])
            outfit_boundaries = torch.cat((outfit_boundaries, torch.tensor([cum_len])))

            j = 0
            while j < len(i["items"]):
                img = Image.open(folder + str(i["items"][j]["index"]) + ".jpg")
                if (img.mode != "RGB"):
                    img = img.convert('RGB')
                batch = torch.cat((batch, preprocess(img).unsqueeze(0)))
                img.close()
                one_hot = nn.functional.one_hot(torch.as_tensor(category_ids.index(i["items"][j]["categoryid"])), 380).unsqueeze(0)
                categories = torch.cat((categories, one_hot))
                j += 1

        print("before resnet")
        batch = resnet(batch).squeeze()
        print("after resnet")
        batch = torch.cat((batch, categories), 1)

        return batch

    def train_val_split():
        #return a split of the data into training and validation sets
        pass

    
    
    


