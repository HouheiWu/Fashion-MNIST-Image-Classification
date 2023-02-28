import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_data_loader(training = True):
    """
    TODO: implement this function.


    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.FashionMNIST('./data', train = training, download = True, transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64)
    return dataloader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64,10)
    )
    return(model)

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for ephoch in range(T):
        loss_total = 0
        accuracy = 0
        total = 0
        for i, data in enumerate(train_loader):
            input, label = data
            batch_size = len(label)
            total += batch_size

            opt.zero_grad()
            output = model(input)
            prop, predicted = torch.max(output, 1)
            accuracy += (predicted == label).sum().item()

            loss = criterion(output, label)
            loss.backward()
            opt.step()
            loss_total += loss.item()*batch_size

        loss_avg = (loss_total/total)
        print(f'Train Epoch: {ephoch}  Accuracy: {accuracy}/{total}({accuracy/total:.2%}) Loss: {loss_avg:.3f}')

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """

    accuracy = 0
    loss_total = 0
    total = 0

    model.eval()

    for i, data in enumerate(test_loader):
        input, label = data
        batch_size = len(label)
        total += batch_size

        output = model(input)
        loss = criterion(output, label)
        loss_total += loss.item() * batch_size

        prop,predicted = torch.max(output, 1)
        accuracy += (label == predicted).sum().item()

    if show_loss:
        print(f'Average loss: {loss_total/total:.4f}')
    print(f'Accuracy: {accuracy/total:.2%}')


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt'
    , 'Sneaker', 'Bag', 'Ankle Boot']

    logits = model(test_images)
    logit = logits[index]
    prob, indexlist = torch.sort(F.softmax(logit, dim= 0), descending=True)
    prob = prob.detach().numpy()[0:3]
    indexlist = indexlist.detach().numpy()[0:3]
    for i, idx in enumerate(indexlist):
        print(f'{class_names[idx]}: {prob[i]:.2%}')


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to examine the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train = get_data_loader()
    test = get_data_loader(False)
    model = build_model()
    train_model(model,train,criterion,5)
    evaluate_model(model,test,criterion,True)
    test_img, _ = next(iter(test))
    predict_label(model,test_img,1)
