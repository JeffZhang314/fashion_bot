class Model(nn.Module): # whole model no batching no cuda
  def __init__(self):
    super(Model, self).__init__()
    self.resnet_to_transformer = resnet_to_transformer
    self.output_embedding_dict = nn.Embedding(1, 512)
    self.transformer = transformer
    self.transformer_to_output = transformer_to_output

  def forward(self, batch, outfit_boundaries):
    batch = self.resnet_to_transformer(batch);
    batch = batch.tensor_split(outfit_boundaries)
    transformer_input = torch.empty(0, 9, 512)
    mask = torch.empty(0, 9)
    zeros = torch.zeros(1, dtype=torch.long)
    output_embedding = self.output_embedding_dict(zeros)
    for i in batch:
      transformer_input = torch.cat((transformer_input, nn.ZeroPad2d((0, 0, 0, 8 - i.shape[0]))(torch.cat((output_embedding, i))).unsqueeze(0)))
      mask = torch.cat((mask, torch.cat((torch.full((i.shape[0] + 1,), True), torch.full((8 - i.shape[0],), False))).unsqueeze(0)))
    batch = self.transformer(transformer_input, src_key_padding_mask = mask)
    batch = batch.select(1, 0)
    batch = self.transformer_to_output(batch)
    batch = batch.squeeze()
    return batch

resnet_to_transformer = ResnetToTransformer()

encoder_layer = nn.TransformerEncoderLayer(d_model = 512, nhead = 8, batch_first = True)
transformer = nn.TransformerEncoder(encoder_layer, num_layers = 6)

class ResnetToTransformer(nn.Module): # connects resnet and transformer
  def __init__(self):
    super(ResnetToTransformer, self).__init__()

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

    self.skip3 = nn.Linear(4096, 512)

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

    x = out

    out = self.relu3(out)
    out = self.bn3(out)
    out = self.dropout3(out)
    out = self.lin3(out)

    out = self.relu4(out)
    out = self.bn4(out)
    out = self.dropout4(out)
    out = self.lin4(out)

    x = self.skip3(x)

    out += x

    return out




class TransformerToOutput(nn.Module): # connects transformer to output
  def __init__(self):
    super(TransformerToOutput, self).__init__()

    self.lin0 = nn.Linear(512, 512)

    self.relu1 = nn.ReLU(True)
    self.bn1 = nn.BatchNorm1d(512)
    self.dropout1 = nn.Dropout(0.05)
    self.lin1 = nn.Linear(512, 512)

    self.relu2 = nn.ReLU(True)
    self.bn2 = nn.BatchNorm1d(512)
    self.dropout2 = nn.Dropout(0.05)
    self.lin2 = nn.Linear(512, 512)

    self.relu3 = nn.ReLU(True)
    self.bn3 = nn.BatchNorm1d(512)
    self.dropout3 = nn.Dropout(0.05)
    self.lin3 = nn.Linear(512, 32)

    self.relu4 = nn.ReLU(True)
    self.bn4 = nn.BatchNorm1d(32)
    self.dropout4 = nn.Dropout(0.05)
    self.lin4 = nn.Linear(32, 1)

    self.skip3 = nn.Linear(512, 1)

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

    x = out

    out = self.relu3(out)
    out = self.bn3(out)
    out = self.dropout3(out)
    out = self.lin3(out)

    out = self.relu4(out)
    out = self.bn4(out)
    out = self.dropout4(out)
    out = self.lin4(out)

    x = self.skip3(x)

    out += x

    return out