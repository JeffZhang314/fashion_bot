import torch.nn as nn
import torch
import torch.optim as optim


class RunModel():
    """
    Class that scafolds the training and evaluation methods and attributes
    for each test case (Test case: Adam, Test case: Lookahead(Adam)).
    """
    def __init__(self, model, adam_optim, batch_size, train_size) -> None:
        if torch.cuda.is_available():
           self.model = model.cuda()
        else:
            self.model = model
        self.train_loss = []
        self.val_loss = []
        #self.train_acc = []
        #self.val_acc = []
        self.adam_optim = adam_optim
        self.batch_size = batch_size
        self.train_size = train_size
        self.d_model = 512
        self.n_steps = 0
        self.n_warmup_steps = 200

    def train(self, train_data, val_data, criterion = nn.MSELoss(), epochs = 1100):

        train_batch, train_boundaries, train_likes, train_views = train_data
        val_batch, val_boundaries, val_likes, val_views = val_data
        
        if torch.cuda.is_available():
            train_batch, train_boundaries, train_likes, train_views = train_batch.cuda(), train_boundaries.cuda(), train_likes.cuda(), train_views.cuda()
            val_batch, val_boundaries, val_likes, val_views = val_batch.cuda(), val_boundaries.cuda(), val_likes.cuda(), val_views.cuda()   
            
        
        batches_per_epoch = round(self.train_size / self.batch_size)
        num_outfits = train_boundaries.shape[0]
        train_boundaries = torch.cat((torch.zeros(1, dtype = torch.long), train_boundaries))

        for epoch in range(epochs):  # loop over whole dataset

            for batch_num in range(batches_per_epoch):

                batch_start = round(num_outfits * batch_num / batches_per_epoch)
                batch_end = round(num_outfits * (batch_num + 1) / batches_per_epoch)

                #prep data
                batch_start_garments = train_boundaries[batch_start]
                batch_end_garments = train_boundaries[batch_end]
                garments = train_batch[batch_start_garments:batch_end_garments]
                outfit_boundaries = train_boundaries[batch_start + 1:batch_end + 1].subtract(train_boundaries[batch_start])
                labels = torch.cat((train_likes[batch_start:batch_end].unsqueeze(1), train_views[batch_start:batch_end].unsqueeze(1)), dim = 1)
                
                #check if cuda is available 
                if torch.cuda.is_available():
                    garments, outfit_boundaries, labels = garments.cuda(), outfit_boundaries.cuda(), labels.cuda()
                
                #training
                out = self.model(garments, outfit_boundaries)
                loss = criterion(out, labels)
                self.train_loss.append(loss.item())
                self.adam_optim.zero_grad()
                loss.backward()
                
                self.n_steps += 1
                lr = (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))
                for param_group in self.adam_optim.param_groups:
                    param_group['lr'] = lr
                self.adam_optim.step()

            
            epoch_loss = 0
            for i in range(len(self.train_loss) - batches_per_epoch, len(self.train_loss)):
                epoch_loss += self.train_loss[i]
            
            val_out = self.model(val_batch, val_boundaries)
            val_labels = torch.cat((val_likes.unsqueeze(1), val_views.unsqueeze(1)), dim = 1)
            val_loss = criterion(val_out, val_labels)
            self.val_loss.append(val_loss.item())
            print(str(epoch_loss) + " " + str(val_loss.item()))
            
    def test(self, test):
        pass
    
    def get_train_loss(self):
        return self.train_loss
    
    def get_val_loss(self):
        return self.val_loss
    
    def get_train_acc(self):
        #train_acc = curr_acc = correct / total
        return self.train_acc    

        """
        8-bit integer (unsigned) torch.uint8 16-bit integer (unsigned) torch.uint16 (limited support) 4 32-bit integer (unsigned) torch.uint32 (limited support) 4 64-bit integer (unsigned) torch.uint64 (limited support) 4 8-bit integer (signed) torch.int8 16-bit integer (signed) torch.int16 or torch.short 32-bit integer (signed) torch.int32 or torch.int 64-bit integer (signed) torch.int64 or torch.long
        """