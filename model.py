import torch.nn as nn
import torch

class FeedForwardOne(nn.Module):
  def __init__(self):
    super(FeedForwardOne, self).__init__()

    # resnet to transformer layers
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

  def forward(self, x):
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

    return out

class FeedForwardTwo(nn.Module):
  def __init__(self):
    super(FeedForwardTwo, self).__init__()

    # transformer to output layers
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

  def forward(self, x):
    out = self.lin5(x)
    
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

class Transformer(nn.Module):
  def __init__(self):
    super(Transformer, self).__init__()

    # transformer encoder layer
    self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = 512, nhead = 8, batch_first = True), num_layers = 6)
    self.output_embedding_dict = nn.Embedding(1, 512)
    
  def init_transformer_mask(self, x):
    # transformer requires exactly 8 garments plus 1 dummy, even if outfit has fewer than 8 garments
    # mask makes sure even if outfit has fewer than 8 garments, the extra garments are ignored
    transformer_input = torch.empty(0, 9, 512)
    mask = torch.empty(0, 9)

    # dummy garment
    # transformer outputs something for each garment, but we want it to output something for whole outfit
    # add a dummy garment (at the beginning of training, its initialized to something random; with training, it gets better)
    zeros = torch.zeros(1, dtype = torch.long)
    output_embedding = self.output_embedding_dict(zeros)

    # go through outfits
    for i in x:
      # concatenates dummy garment to rest of garments, pad with 0s, add that to other outfits 
      # concatenate output_embedding and i
      concatenated_input = torch.cat((output_embedding, i))
      # apply padding using ZeroPad2d to the concatenated input
      padded_input = nn.ZeroPad2d((0, 0, 0, 8 - i.shape[0]))(concatenated_input)
      # unsqueeze the result and concatenate it with transformer_input
      transformer_input = torch.cat((transformer_input, padded_input.unsqueeze(0)))

      # see explanation of masks above and/or pytorch documentation for transformer masks
      # create a tensor of True values with the size i.shape[0] + 1
      true_values = torch.full((i.shape[0] + 1,), True)
      # create a tensor of False values with the size 8 - i.shape[0]
      false_values = torch.full((8 - i.shape[0],), False)
      # concatenate the True and False tensors
      concat_mask = torch.cat((true_values, false_values))
      # unsqueeze the concatenated mask and concatenate it with the existing mask
      mask = torch.cat((mask, concat_mask.unsqueeze(0)))
    
    # cuda
    if torch.cuda.is_available():
      transformer_input, mask = transformer_input.cuda(), mask.cuda()

    return transformer_input, mask

  def forward(self, x, mask):
    out = self.transformer(x, src_key_padding_mask = mask)
    # choose only dummy node
    out = out.select(1, 0)
    
    return out

class Fasho(nn.Module): # everything after resnet
  def __init__(self):
    super(Fasho, self).__init__()
    self.feed_forward_one = FeedForwardOne()
    self.feed_forward_two = FeedForwardTwo()
    self.transformer = Transformer()

  def forward(self, x, outfit_boundaries):
    feed_forward_one_output = self.feed_forward_one.forward(x)
    # transformer needs garments to be grouped by outfit
    feed_forward_one_output = feed_forward_one_output.tensor_split(outfit_boundaries[:-1])

    transformer_input, mask = self.transformer.init_transformer_mask(feed_forward_one_output)
    transformer_output = self.transformer.forward(transformer_input, mask)

    feed_forward_two_output = self.feed_forward_two.forward(transformer_output)

    return feed_forward_two_output