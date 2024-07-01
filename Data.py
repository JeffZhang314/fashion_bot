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
        self.clothing_dict = {
            "2": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Casual"},
            "4": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Day Dresses"},
            "5": {"gender": "Womens", "formality": "Formal", "type": "Top", "specific_type": "Cocktail Dresses"},
            "6": {"gender": "Womens", "formality": "Formal", "type": "Top", "specific_type": "Gowns"},
            "7": {"gender": "Womens", "formality": "Casual", "type": "Bottom", "specific_type": "Skirts"},
            "8": {"gender": "Womens", "formality": "Casual", "type": "Bottom", "specific_type": "Mini Skirts"},
            "9": {"gender": "Womens", "formality": "Casual", "type": "Bottom", "specific_type": "Knee Length Skirts"},
            "10": {"gender": "Womens", "formality": "Casual", "type": "Bottom", "specific_type": "Long Skirts"},
            "11": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Tops"},
            "15": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Tunics"},
            "17": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Blouses"},
            "18": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Cardigans", "occasion": "Casual"},
            "19": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Sweaters"},
            "21": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "T-Shirts"},
            "23": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Outerwear"},
            "24": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Coats"},
            "25": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Jackets"},
            "26": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Vests"},
            "27": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Jeans"},
            "28": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Pants"},
            "29": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Shorts"},
            "30": {"gender": "Unisex", "formality": "Formal", "type": "Top", "specific_type": "Suits"},
            "31": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Swimwear"},
            "33": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Activewear"},
            "237": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Skinny Jeans"},
            "238": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Bootcut Jeans"},
            "239": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Wide Leg Jeans"},
            "240": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Boyfriend Jeans"},
            "241": {"gender": "Womens", "formality": "Casual", "type": "Bottom", "specific_type": "Leggings"},
            "243": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Jumpsuits"},
            "244": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Rompers"},
            "247": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Camisoles"},
            "248": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Chemises"},
            "249": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Pajamas"},
            "250": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Robes"},
            "251": {"gender": "Womens", "formality": "Casual", "type": "Bottom", "specific_type": "Tights"},
            "252": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Activewear Tops"},
            "253": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Activewear Pants"},
            "254": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Activewear Skirts"},
            "255": {"gender": "Unisex", "formality": "Casual", "type": "Bottom", "specific_type": "Activewear Shorts"},
            "256": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Activewear Jackets"},
            "257": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Sports Bras"},
            "271": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Clothing"},
            "272": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Shirts"},
            "273": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Sweaters"},
            "275": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "T-Shirts"},
            "276": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Outerwear"},
            "277": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Sportcoats & Blazers"},
            "278": {"gender": "Mens", "formality": "Casual", "type": "Bottom", "specific_type": "Jeans"},
            "279": {"gender": "Mens", "formality": "Casual", "type": "Bottom", "specific_type": "Pants"},
            "280": {"gender": "Mens", "formality": "Casual", "type": "Bottom", "specific_type": "Shorts"},
            "281": {"gender": "Mens", "formality": "Formal", "type": "Top", "specific_type": "Suits"},
            "282": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Swimwear"},
            "283": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Underwear"},
            "284": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Sleepwear"},
            "285": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Activewear"},
            "286": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Activewear Tops"},
            "287": {"gender": "Mens", "formality": "Casual", "type": "Bottom", "specific_type": "Activewear Pants"},
            "288": {"gender": "Mens", "formality": "Casual", "type": "Bottom", "specific_type": "Activewear Shorts"},
            "289": {"gender": "Mens", "formality": "Casual", "type": "Top", "specific_type": "Activewear Jackets"},
            "309": {"gender": "Unisex", "formality": "Casual", "type": "Top", "specific_type": "Activewear Tank Tops"},
            "310": {"gender": "Both", "formality": "Casual", "type": "Bottom", "specific_type": "Straight Leg Jeans"},
            "332": {"gender": "Both", "formality": "Casual", "type": "Bottom", "specific_type": "Capri & Cropped Pants"},
            "4516": {"gender": "Both", "formality": "Formal", "type": "Top", "specific_type": "Wedding Dresses"},
            "1605": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Bikinis"},
            "1606": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "One Piece Swimsuits"},
            "1607": {"gender": "Womens", "formality": "Casual", "type": "Top", "specific_type": "Cover-ups"}
        }
        self.category_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 343, 67, 68, 69, 71, 74, 75, 76, 77, 78, 85, 93, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 106, 107, 108, 110, 113, 115, 116, 118, 120, 122, 123, 124, 126, 127, 129, 130, 132, 135, 136, 139, 140, 141, 143, 144, 4241, 4242, 147, 4244, 150, 4247, 4248, 153, 154, 155, 157, 4254, 159, 160, 4257, 162, 163, 164, 166, 167, 168, 169, 170, 4267, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 4292, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 
                                    224, 225, 226, 227, 228, 229, 231, 236, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 270, 271, 272, 273, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 318, 319, 320, 321, 4426, 4428, 333, 334, 335, 4432, 4433, 338, 339, 340, 4437, 342, 4439, 4440, 4441, 4442, 4443, 4445, 4446, 4447, 4448, 4449, 4450, 4451, 4452, 4454, 4455, 4456, 4457, 4458, 4459, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4468, 4470, 4472, 4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 
                                    4488, 4489, 4490, 4492, 4493, 4495, 4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 
                                    4512, 4513, 4514, 4516, 4517, 4518, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4438, 4240, 146, 148, 4246, 151, 152, 949, 161, 4258, 171, 4269, 4276, 196, 3336, 1605, 1606, 1607, 1967, 332, 4429, 4430, 4431, 336, 337, 4434, 4435, 4436, 341]
        
        self.genders = ["Unisex", "Womens", "Mens"]
        self.formalities = ["Casual", "Formal"]
        self.types = ["Top", "Bottom"]
        self.specific_types = ["Casual","Day Dresses","Cocktail Dresses","Gowns","Skirts","Mini Skirts","Knee Length Skirts","Long Skirts","Tops","Tunics","Blouses","Cardigans","Sweaters","T-Shirts","Outerwear","Coats",
                    "Jackets","Vests","Jeans","Pants","Shorts","Suits","Swimwear","Activewear","Skinny Jeans","Bootcut Jeans","Wide Leg Jeans","Boyfriend Jeans","Leggings","Jumpsuits","Rompers","Camisoles","Chemises","Pajamas","Robes","Tights","Activewear Tops",
                    "Activewear Pants","Activewear Skirts","Activewear Shorts","Activewear Jackets","Sports Bras","Clothing","Shirts","Sweaters","T-Shirts","Outerwear","Sportcoats & Blazers","Jeans","Pants","Shorts","Suits",
                    "Swimwear","Underwear","Sleepwear","Activewear","Activewear Tops","Activewear Pants","Activewear Shorts","Activewear Jackets","Activewear Tank Tops","Straight Leg Jeans","Capri & Cropped Pants","Wedding Dresses","Bikinis",
                    "One Piece Swimsuits","Cover-ups"]
        print("len specific types")
        print(len(self.specific_types))


    def prep_data(self): # get resnet output and category id one-hot encoding of whole dataset

        # Freeze all the pre-trained layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        batch = torch.tensor([])

        f = open(self.path + "valid_no_dup.json",)
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
        categories = torch.empty(0, len(self.genders) + len(self.formalities) + len(self.types) + len(self.specific_types))

        garments_printed = 0
        
        for i in data:
            
            folder = self.path + "images\\" + i["set_id"] + "\\"
            self.likes = torch.cat((self.likes, torch.tensor([i["likes"]])))
            self.views = torch.cat((self.views, torch.tensor([i["views"]])))

            self.cum_len += len(i["items"])
            self.outfit_boundaries = torch.cat((self.outfit_boundaries, torch.tensor([self.cum_len])))

            j = 0
            while j < len(i["items"]):
                if (str(i["items"][j]["categoryid"]) in self.clothing_dict):
                    img = Image.open(folder + str(i["items"][j]["index"]) + ".jpg")
                    if (img.mode != "RGB"):
                        img = img.convert('RGB')
                    batch = torch.cat((batch, self.preprocess(img).unsqueeze(0)))
                    img.close()
                    
                    
                    #one_hot = nn.functional.one_hot(torch.as_tensor(self.category_ids.index(i["items"][j]["categoryid"])), 380).unsqueeze(0)
                    
                    item_id = i["items"][j]["categoryid"]
                    
                    gender = self.clothing_dict[str(item_id)]["gender"]
                    gender_one_hot = nn.functional.one_hot(torch.as_tensor(self.genders.index(gender)), len(self.genders))
                    
                    formality = self.clothing_dict[str(item_id)]["formality"]
                    
                    formality_one_hot = nn.functional.one_hot(torch.as_tensor(self.formalities.index(formality)), len(self.formalities))
                    
                    type = self.clothing_dict[str(item_id)]["type"]
                    type_one_hot = nn.functional.one_hot(torch.as_tensor(self.types.index(type)), len(self.types))
                    
                    specific_type = self.clothing_dict[str(item_id)]["specific_type"]
                    specific_type_one_hot = nn.functional.one_hot(torch.as_tensor(self.specific_types.index(specific_type)), len(self.specific_types))
                    
                    one_hot = torch.cat((gender_one_hot, formality_one_hot, type_one_hot, specific_type_one_hot)).unsqueeze(0)
                    
                    categories = torch.cat((categories, one_hot))
                j += 1

        print("before resnet")
        batch = self.resnet(batch).squeeze()
        print("after resnet") 
        batch = torch.cat((batch, categories), 1)

        for i in range(20):
            print("category id one hot encoding")
            print(batch[i][2048:])

        return batch