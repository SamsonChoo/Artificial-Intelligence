import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import copy
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torchvision import transforms, models
import matplotlib.pyplot as plt
import tkinter
import numpy as np

class mydata(Dataset):
    def __init__(self, x, y, transform, directory):
        self.x = x    # image
        self.y = torch.tensor(y, dtype = torch.long)    # label
        self.transform = transform
        self.directory = directory
        
    def __len__(self,):
        return len(self.x)
        
    def __getitem__(self,idx):
        name = self.x[idx]
        label = self.y[idx]
        
        img = Image.open(self.directory + name).convert("RGB")
        img = self.transform(img)
        
        return img, y
        
def loading_data(file):
    labels = []
    data = []
    
    for line in open(file,"r"):
        line_sep = line.split()
        if(line_sep[14]=='1'):
            labels.append([0])
            data.append(line_sep[0])
        elif(line_sep[15]=='1'):
            labels.append([1])
            data.append(line_sep[0])
    
    return labels, data
    
def split_data(labels,data):
    labels_train,labels_val,labels_test = np.split(labels, [int(len(labels)*0.6), int(len(labels)*0.7)])
    data_train,data_val,data_test = np.split(data, [int(len(data)*0.6), int(len(data)*0.7)])
    return data_train, labels_train, data_val, labels_val, data_test, labels_test
    
def train_model(model, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    epoch_loss_data = {"train": [],"val":[]}
    epoch_acc_data = {"train": [],"val":[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_len[phase]
            epoch_acc = running_corrects.double() / dataset_len[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            epoch_loss_data[phase].append(epoch_loss)
            epoch_acc_data[phase].append(epoch_acc.item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    print("epoch loss data \n", epoch_loss_data)
    print("epoch acc data \n", epoch_acc_data)
    
    model.load_state_dict(best_model_wts)
    torch.save(model, "trained_model")

    plt.figure(1)
    t = range(len(epoch_loss_data["train"]))
    plt.plot(t, epoch_loss_data["train"], "r")
    plt.plot(t, epoch_loss_data["val"], "b")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.savefig("loss.png")

    plt.figure(2)
    t = range(len(epoch_acc_data["train"]))
    plt.plot(t, epoch_acc_data["train"], "r")
    plt.plot(t, epoch_acc_data["val"], "b")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.savefig("accuracy.png")

    return model    
    
def test_model(model, device):
    phase="test"
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
    
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / dataset_len[phase]
    test_acc = running_corrects.double() / dataset_len[phase]

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
    
if __name__ == "__main__":
    labels, data = loading_data("trainset_gt_annotations.txt")
    data_train, labels_train, data_val, labels_val, data_test, labels_test = split_data(labels, data)
        
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    datasets = {"train": mydata(data_train, labels_train, transform, "photos_8000/"),
                      "val": mydata(data_val, labels_val, transform, "photos_8000/"),
                      "test": mydata(data_test, labels_test, transform, "photos_8000/")}
    dataset_len = {x: len(datasets[x]) for x in ['train', 'val', 'test']}    
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4)
                  for x in ['train', 'val', 'test']}
    
    model_ft = models.resnet18(pretrained=True)  #loading model weights, setting B,C
#    model_ft = models.resnet18(pretrained=False)  #without weights, setting A
    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to("cuda:0")
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)  #training all layers, setting A,B
#    optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)  #freeze all layers and only train last layer, setting C
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, "cuda:0", num_epochs=30)
    torch.save(model_ft, "trained_model2")

    test_model(model_ft,"cuda:0")