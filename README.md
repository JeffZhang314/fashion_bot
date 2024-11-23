# Fashion_bot
Fashion Bot
# Contributers
- Jeff Zhang
- Edward Yang
- Terry Qiu
- Tim Wang
# Description
Fashion Bot ML Model

Please read the code before uncommenting, thanks

In Data.py, data prep (preprocess, resnet, one-hot encoding, etc.) and tkinter stuff (for data cleaning, a separate task) are in the same for loop (for i in data). Tyler and Anthony are separating this before Monday.

Once Data.py is clean,
myData = Data(...)
annotated_batch = myData.prep_data()
will return batch, self.outfit_boundaries, self.likes, self.views

batch is resnet vectors concatenated with one-hot encoding of the category (gender, formality, type, specific_type)
outfit_boundaries is used for tensor split: say we have a 2-garment outfit followed by a 3-garment outfit. then, outfit_boundaries will be (2, 5).
likes and views are the labels

This gets torch.save for later use to train.pt, valid.pt, test.pt (depending on if you're running data.py on test set, train set, or valid set, so you have to run it 3 times)

Then, this gets torch.load into the corresponding tensor (train_data, valid_data, and test_data)

Finally,
run_model = RunModel(model, adam_optim, batch_size, train_size)
run_model.train(train_data, valid_data, criterion=nn.MSELoss(), epochs=1100)
will train

Keep an eye on where cuda should and shouldnt be used