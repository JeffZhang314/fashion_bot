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

        category_ids = [
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 343, 67, 68, 69, 71, 74, 75, 76, 77, 78, 85, 93, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 106, 107, 108, 110, 113, 115, 116, 118, 120, 122, 123, 124, 126, 127, 129, 130, 132, 135, 136, 139, 140, 141, 143, 144, 4241, 4242, 147, 4244, 150, 4247, 4248, 153, 154, 155, 157, 4254, 159, 160, 4257, 162, 163, 164, 166, 167, 168, 169, 170, 4267, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 4292, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 231, 236, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 270, 271, 272, 273, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 318, 319, 320, 321, 4426, 4428, 333, 334, 335, 4432, 4433, 338, 339, 340, 4437, 342, 4439, 4440, 4441, 4442, 4443, 4445, 4446, 4447, 4448, 4449, 4450, 4451, 4452, 4454, 4455, 4456, 4457, 4458, 4459, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4468, 4470, 4472, 4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489, 4490, 4492, 4493, 4495, 4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 4512, 4513, 4514, 4516, 4517, 4518, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4438, 4240, 146, 148, 4246, 151, 152, 949, 161, 4258, 171, 4269, 4276, 196, 3336, 1605, 1606, 1607, 1967, 332, 4429, 4430, 4431, 336, 337, 4434, 4435, 4436, 341
        ]

        weights = models.ResNet152_Weights.IMAGENET1K_V2
        
        # Load the pre-trained ResNet-152 model
        resnet = models.resnet152(weights = weights)
        preprocess = weights.transforms()
        likes = torch.empty(0)
        views = torch.empty(0)
        outfit_boundaries = torch.empty(0)
        
        
        # Freeze all the pre-trained layers
        for param in resnet.parameters():
            param.requires_grad = False
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        cum_len = 0

        torch.manual_seed(0)

        batch_size = 64

        f = open("valid_no_dup.json",)
        data = load(f)
        random.Random(0).shuffle(data)
        f.close()

        

    def prep_batch(data): # get resnet output and category id one-hot encoding of just data

        nonlocal category_ids, preprocess, resnet, cum_len, batch_size, likes, views, outfit_boundaries

        batch = torch.empty(0, 3, 224, 224)
        categories = torch.empty(0, 380)

        n_processed = 0

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

    batch = torch.tensor([])

    for i in range(2):
        inc_batch = prep_batch(data[batch_size * i:min(batch_size * (i + 1), len(data))])
        batch = torch.cat((batch, inc_batch))

    
    
    return batch, outfit_boundaries, likes, views


annotated_batch = prep_data()

print(annotated_batch)

torch.save(annotated_batch, prefix + 'tensor.pt')
print("done")


loaded_batch, loaded_outfit_boundaries, loaded_likes, loaded_views = torch.load(prefix + 'tensor.pt')