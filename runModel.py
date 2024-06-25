import torch.nn as nn
import torch
import torch.optim as optim


class RunModel():
    """
    Class that scafolds the training and evaluation methods and attributes
    for each test case (Test case: Adam, Test case: Lookahead(Adam)).
    """
    def __init__(self, model, adam_optim, batch_size, train_size) -> None:
        #if torch.cuda.is_available():
        #    self.model = model.cuda()
        #else:
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
        self.n_warmup_steps = 4000

    def train(self, train_batch, train_boundaries, val_batch, val_boundaries, criterion=nn.MSELoss()):
        
        batches_per_epoch = round(self.train_size / self.batch_size)
        self.adam_optim.zero_grad()
        train_boundaries = torch.cat((np.zeros(1), train_boundaries))

        for epoch in range(epochs):  # loop over whole dataset

            for batch_num in range(batches_per_epoch):

                batch_start = round(train_boundaries.shape[0] * batch_num / batches_per_epoch)
                batch_end = round(train_boundaries.shape[0] * (batch_num + 1) / batches_per_epoch)
                out = self.model(train_batch[train_boundaries[batch_start]:train_boundaries[batch_end]], train_boundaries[batch_start:batch_end])

                self.n_steps += 1
                lr = (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr
                adam_optim.step()

            #if torch.cuda.is_available():
            #    train = train.cuda()
            #    val = val.cuda()
            
            
            for i in range(n_warmup_steps * 21):
                out = model(train)
                loss = loss_fn(out, valid_labels)
                #not sure the labels situation from original code
                print(i)
                print(loss)
                print(scheduler.get_last_lr())
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                scheduler.step()
            
    def test(self, test):
        pass
    
    def get_train_loss(self):
        return self.train_loss
    
    def get_val_loss(self):
        return self.val_loss
    
    def get_train_acc(self):
        #train_acc = curr_acc = correct / total
        return self.train_acc    