import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

#Constants:
EPOCHS = 100


def read_data(train_location, test_location):
    df_train = pd.read_csv(train_location)
    df_test = pd.read_csv(test_location)
    in_train = torch.tensor(df_train.iloc[0:, 1:].values, dtype=torch.float32)
    out_train = torch.tensor(df_train.iloc[0:, 0].values, dtype=torch.float32)
    #print(x)
    in_test = torch.tensor(df_test.iloc[:, 1:].values, dtype=torch.float32)
    out_test = torch.tensor(df_test.iloc[:, 0].values, dtype=torch.float32)
    return in_train, out_train, in_test, out_test


#1. Create the model/Parametrization
#We choose to override the torch.nn.Module class to create our own model.
class neural_network_model(torch.nn.Module):
    def __init__(self) -> None: #Constructor, but also we don't need *args and **kwargs
        super().__init__()
        self.input = torch.nn.Linear(52, 64)
        self.activation1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(64, 32)
        self.activation2 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(32, 1)
        self.activation3 = torch.nn.LeakyReLU()

    def forward(self, x): #Note that x should be an input tensor
        x = self.input(x)
        x = self.activation1(x)
        x = self.linear1(x)
        x = self.activation2(x)
        x = self.linear2(x)
        x = self.activation3(x)
        return x
    
    def string(self): #
        total_params = sum(p.numel() for p in self.parameters())
        return f"This is a two layer model, with 52 input nodes, 32 hidden nodes, and 1 output node. It has " + str(total_params)  + " parameters"
    
loan_eligibility_model = neural_network_model()

#0. Setup our tensorboard
writer = SummaryWriter('runs/Basic NN')
writer.add_graph(loan_eligibility_model, torch.rand([1, 52]))

#2. Create the loss function
loss_function = torch.nn.MSELoss(reduction="mean") #The mean

#3. Create the optimizer to do our calculations
optimizer = torch.optim.Adam(loan_eligibility_model.parameters(), lr=1e-3) #lr is the learning rate, and 1e-3 is a standard

#4. Create the training loop
def train(x, y, model, loss_function, optimizer):
    for iteration in range(EPOCHS):
        #Forward pass
        step = 0
        for index, elem in enumerate(x):
            y_pred = model(elem)
            loss = loss_function(y_pred, y[index])
            #Backward pass - this updates all parameters within our model and recalculates them.
            loss.backward() #Compute the gradients
            optimizer.step() #Update the parameters
            optimizer.zero_grad() #Clear the gradients
            step += 1
        #Write to tensorboard
        writer.add_scalar('training loss',
                            loss,
                            iteration)
        print(f"Epoch {iteration} | Loss: {loss.item()}")

#5. Create the test loop
def test(x, y, model):
    for index, elem in enumerate(x):
        y_pred = model(elem)
        loss = loss_function(y_pred, y[index])
        #Write test loss to tensorboard
        writer.add_scalar('test loss',
                    loss / 1000,
                    index)
        print(f"Test Loss: {loss.item()}")

#6. Create the main function with visualization
def main():
    training_loc = "car_price_training.csv"
    test_loc = "car_price_test.csv"
    x_train, y_train, x_test, y_test = read_data(training_loc, test_loc)
    train(x_train, y_train, loan_eligibility_model, loss_function, optimizer)
    test(x_test, y_test, loan_eligibility_model)
    writer.close()

#7. Run the main function!
main()