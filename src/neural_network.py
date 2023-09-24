import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

#Constants:
EPOCHS = 3



def read_data(train_location, test_location):
    df_train = pd.read_csv(train_location)
    df_test = pd.read_csv(test_location)
    in_train = torch.tensor(df_train.iloc[:, 1:].values, dtype=torch.float32).abs()
    out_train = torch.tensor(df_train.iloc[:, 0].values, dtype=torch.float32).abs()
    in_test = torch.tensor(df_test.iloc[:, 1:].values, dtype=torch.float32).abs()
    out_test = torch.tensor(df_test.iloc[:, 0].values, dtype=torch.float32).abs()

    print(out_train)
    print(in_train)
    return in_train, out_train, in_test, out_test


#1. Create the model class
class neural_network_model(torch.nn.Module):
    def __init__(self) -> None: #Constructor, but also we don't need *args and **kwargs
        super().__init__()
        self.input = torch.nn.Linear(13, 32)
        #self.activation1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(32, 32)
        self.activation2 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(32, 32)
        self.activation3 = torch.nn.LeakyReLU()
        self.linear3 = torch.nn.Linear(32, 16)
        self.activation4 = torch.nn.LeakyReLU()
        self.linear4 = torch.nn.Linear(16, 13)
        self.activation5 = torch.nn.LeakyReLU()

    def forward(self, x): #Note that x should be an input tensor
        x = self.input(x)
        #x = self.activation1(x)
        x = self.linear1(x)
        x = self.activation2(x)
        x = self.linear2(x)
        x = self.activation3(x)
        x = self.linear3(x)
        x = self.activation4(x)
        x = self.linear4(x)
        x = self.activation5(x)
        #print(x)
        return x
    
loan_eligibility_model = neural_network_model()

writer = SummaryWriter('runs/Basic NN')
writer.add_graph(loan_eligibility_model, torch.rand([1, 13]))

loss_function = torch.nn.CrossEntropyLoss() #The mean

optimizer = torch.optim.Adam(loan_eligibility_model.parameters(), lr=1e-3)

def train(x, y, model, loss_function, optimizer):
    for iteration in range(EPOCHS):
        #Forward pass
        step = 0
        for index, elem in enumerate(x):
            y_pred = model(elem)
            y_compare = torch.zeros_like(y_pred)
            y_compare[int(y[index] - 1)] = 1
            loss = loss_function(y_pred, y_compare)
            #Backward pass - this updates all parameters within our model and recalculates them.
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad() #Clear the gradients after training in order to get it better
            step += 1
        #Write to tensorboard
        writer.add_scalar('Training Loss',loss, iteration)
        print(f"Epoch {iteration} | Loss: {loss.item()}") #Write the epochs    

#5. Create the test loop
def test(x, y, model):
    print(x.shape)
    print(y.shape)
    y_predictions = model (x)
    print(y_predictions.shape)
    print(y)
    print(y_predictions)

    #for index, elem in enumerate(x):
    #y_pred = model(elem)
    #print(y_pred.shape)
    #loss = loss_function(y_pred, y[index])
    #writer.add_scalar('Test Loss', loss, index)
    #writer.add_scalar('Accuracy', accuracy, )
    #writer.add_scalar('Precision', )

#6. Create the main function with visualization
def main():
    training_loc = "../dataset/updated_train.csv"
    test_loc = "../dataset/updated_test.csv"
    x_train, y_train, x_test, y_test = read_data(training_loc, test_loc)
    train(x_train, y_train, loan_eligibility_model, loss_function, optimizer)
    test(x_test, y_test, loan_eligibility_model)
    torch.save(loan_eligibility_model.state_dict(), "DNN Model")
#Run our definitions
main()
writer.close()