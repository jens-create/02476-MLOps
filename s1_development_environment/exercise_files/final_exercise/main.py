import argparse
import sys

import torch
from torch import nn
import click

from data import mnist
from model import MyAwesomeModel



@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr, epochs = 30, print_every = 500):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    trainloader, testloader = mnist()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.NLLLoss()

    steps = 0
    running_loss = 0
    for e in range(epochs):
        model.train() #for dropout

        for images, labels in trainloader:
            steps += 1
            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()
            
            output = model.forward(images)
            #Yt_train = Yt_train.type(torch.LongTensor)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()

    #save checkpoint
    torch.save(model.state_dict(), '/Users/jenspt/Desktop/git/02476-MLOps/s1_development_environment/exercise_files/final_exercise/checkpoint.pth')





@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load('/Users/jenspt/Desktop/git/02476-MLOps/s1_development_environment/exercise_files/final_exercise/'+model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    _, testloader = mnist()


    criterion = nn.NLLLoss()
    _, accuracy = validation(model, testloader, criterion)

    print("Test Accuracy: {:.3f}".format(accuracy/len(testloader)))


def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    