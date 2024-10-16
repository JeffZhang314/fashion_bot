
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models 
#print to make sure running
print("hi")
#setup
weights = models.ResNet152_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
resnet = models.resnet152(weights = weights)
for param in resnet.parameters():
    param.requires_grad = False
resnet = nn.Sequential(*(list(resnet.children())[:-1]))
batch = torch.empty(0, 3, 224, 224)
categories = torch.empty(0, 380)

#load images
img = Image.open("assets/testshirt.jpg")
batch = torch.cat((batch, preprocess(img).unsqueeze(0)))
img.close()
category = torch.eye(380)[379]
categories = torch.cat((categories, category.unsqueeze(0)))
img = Image.open("assets/pants.jpg")
batch = torch.cat((batch, preprocess(img).unsqueeze(0)))
img.close()
category = torch.eye(380)[19]
categories = torch.cat((categories, category.unsqueeze(0)))
img = Image.open("assets/testshoes.jpg")
batch = torch.cat((batch, preprocess(img).unsqueeze(0)))
img.close()
category = torch.eye(380)[31]
categories = torch.cat((categories, category.unsqueeze(0)))
img = Image.open("assets/shirt2.jpg")
batch = torch.cat((batch, preprocess(img).unsqueeze(0)))
img.close()
category = torch.eye(380)[14]
categories = torch.cat((categories, category.unsqueeze(0)))
img = Image.open("assets/pants2.jpg")
batch = torch.cat((batch, preprocess(img).unsqueeze(0)))
img.close()
category = torch.eye(380)[20]
categories = torch.cat((categories, category.unsqueeze(0)))
img = Image.open("assets/shoes2.jpg")
batch = torch.cat((batch, preprocess(img).unsqueeze(0)))
img.close()
category = torch.eye(380)[31]
categories = torch.cat((categories, category.unsqueeze(0)))

#resnet
resnet_vecs = resnet(batch).squeeze()
resnet_categories = torch.cat((resnet_vecs, categories), 1)
#print tests
"""print(resnet_vecs[0][841])
print(resnet_vecs[1][608])
print(resnet_vecs[2][770])
print(torch.mean(resnet_vecs))
print(torch.std(resnet_vecs))
print(torch.sort(resnet_vecs[0]))
print(torch.sort(resnet_vecs[1]))
print(torch.sort(resnet_vecs[2]))"""




#other file for model





class Fasho(nn.Module): # connects resnet and transformer
  def __init__(self):
    super(Fasho, self).__init__()

    self.output_embedding_dict = nn.Embedding(1, 512)
    #Resnet to transformer layers
    self.lin0 = nn.Linear(2428, 4096)

    self.relu1 = nn.ReLU(True)
    self.bn1 = nn.BatchNorm1d(4096)
    self.dropout1 = nn.Dropout(0.05)
    self.lin1 = nn.Linear(4096, 4096)

    self.relu2 = nn.ReLU(True)
    self.bn2 = nn.BatchNorm1d(4096)
    self.dropout2 = nn.Dropout(0.05)
    self.lin2 = nn.Linear(4096, 4096)

    self.relu3 = nn.ReLU(True)
    self.bn3 = nn.BatchNorm1d(4096)
    self.dropout3 = nn.Dropout(0.05)
    self.lin3 = nn.Linear(4096, 2048)

    self.relu4 = nn.ReLU(True)
    self.bn4 = nn.BatchNorm1d(2048)
    self.dropout4 = nn.Dropout(0.05)
    self.lin4 = nn.Linear(2048, 512)

    self.skip34 = nn.Linear(4096, 512)
    
    #Transformer encoder layer
    self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = 512, nhead = 8, batch_first = True), num_layers = 6)
    
    #Transformer Output layers
    self.lin5 = nn.Linear(512, 512)

    self.relu6 = nn.ReLU(True)
    self.bn6 = nn.BatchNorm1d(512)
    self.dropout6 = nn.Dropout(0.05)
    self.lin6 = nn.Linear(512, 512)

    self.relu7 = nn.ReLU(True)
    self.bn7 = nn.BatchNorm1d(512)
    self.dropout7 = nn.Dropout(0.05)
    self.lin7 = nn.Linear(512, 512)

    self.relu8 = nn.ReLU(True)
    self.bn8 = nn.BatchNorm1d(512)
    self.dropout8 = nn.Dropout(0.05)
    self.lin8 = nn.Linear(512, 32)

    self.relu9 = nn.ReLU(True)
    self.bn9 = nn.BatchNorm1d(32)
    self.dropout9 = nn.Dropout(0.05)
    self.lin9 = nn.Linear(32, 2)

    self.skip89 = nn.Linear(512, 2)

  def forward(self, x, outfit_boundaries):

    #resnet to transformer layer
    out = self.lin0(x)

    x = out

    out = self.relu1(out)
    out = self.bn1(out)
    out = self.dropout1(out)
    out = self.lin1(out)

    out = self.relu2(out)
    out = self.bn2(out)
    out = self.dropout2(out)
    out = self.lin2(out)

    out += x

    x = out

    out = self.relu3(out)
    out = self.bn3(out)
    out = self.dropout3(out)
    out = self.lin3(out)

    out = self.relu4(out)
    out = self.bn4(out)
    out = self.dropout4(out)
    out = self.lin4(out)

    x = self.skip34(x)

    out += x

    out = out.tensor_split(outfit_boundaries[:-1])

    transformer_input = torch.empty(0, 9, 512)
    mask = torch.empty(0, 9)
  
    zeros = torch.zeros(1, dtype = torch.long)
    output_embedding = self.output_embedding_dict(zeros)
    
    for i in out:
      transformer_input = torch.cat((transformer_input, nn.ZeroPad2d((0, 0, 0, 8 - i.shape[0]))(torch.cat((output_embedding, i))).unsqueeze(0)))
      mask = torch.cat((mask, torch.cat((torch.full((i.shape[0] + 1,), True), torch.full((8 - i.shape[0],), False))).unsqueeze(0)))
    
    if torch.cuda.is_available():
      transformer_input, mask = transformer_input.cuda(), mask.cuda()
    
    #transformer layer
    out = self.transformer(transformer_input, src_key_padding_mask = mask)

    #getting the output we want
    out = out.select(1, 0)
    
    #Transformer output
    out = self.lin5(out)

    x = out

    out = self.relu6(out)
    out = self.bn6(out)
    out = self.dropout6(out)
    out = self.lin6(out)

    out = self.relu7(out)
    out = self.bn7(out)
    out = self.dropout7(out)
    out = self.lin7(out)

    out += x

    x = out

    out = self.relu8(out)
    out = self.bn8(out)
    out = self.dropout8(out)
    out = self.lin8(out)

    out = self.relu9(out)
    out = self.bn9(out)
    out = self.dropout9(out)
    out = self.lin9(out)

    x = self.skip89(x)

    out += x
    
    return out

#model
model = torch.load('model.pt')
outfit_boundaries = torch.tensor([3, 6]) 
print("before")
likes_views = model(resnet_categories,outfit_boundaries)
print("after")
results = []
for i in likes_views:
  results.append(i[0].item() ** 2 / i[1].item())
print(results)
print('bye')



