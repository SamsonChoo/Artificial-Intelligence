import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms,  models
from getimagenetclasses import parsesynsetwords, parseclasslabel, get_classes
from PIL import Image

img_dir = "imagenet2500/imagespart/"
lbl_dir = "ILSVRC2012_bbox_val_v3/val/"

class Mydata(Dataset):

    def __init__(self, img_dir, lbl_dir, transform=None):

        self.img_name_list = os.listdir(img_dir)     # list of names of images
        self.xml_list = os.listdir(lbl_dir)[:len(self.img_name_list)]     # list of the first 2500 xml
        self.lbl_list = []     # list of the labels
        self.transform = transform

        filen = 'synset_words.txt'
        indicestosynsets, synsetstoindices, synsetstoclassdescr = parsesynsetwords(filen)

        for xml in self.xml_list:
            label = parseclasslabel(lbl_dir + xml, synsetstoindices)[0] # get the label/index
            self.lbl_list.append(label)

        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        label = self.lbl_list[idx]        
        image = Image.open(img_dir + self.img_name_list[idx]).convert("RGB")

        if self.transform:
            return self.transform(image),label     # does transformation if exists
    
        return image, label
    
def rescale_by_hand(k,bilinear=True):
    img = Mydata(img_dir,lbl_dir)[k][0]
    ratio = min(img.width/224, img.height/224)     # get img ratio according to shorter side 
    if (bilinear):
        img = img.resize((int(img.width/ratio), int(img.height/ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(img.width/ratio), int(img.height/ratio)))

    img = np.array(img)
    img = img/255     # normalization
    img = img.transpose(2,0,1)
    img = np.expand_dims(img, axis=0)
    # pytorch standard for imagenet normalization
    m = np.array([0.485, 0.456, 0.406])
    s = np.array([0.229, 0.224, 0.225])
    img = (img-m[np.newaxis, :, np.newaxis, np.newaxis])/(s[np.newaxis, :, np.newaxis, np.newaxis])
    #print(img.shape)
    height,width = img.shape[2:4]
    h_bot = int(np.ceil((height-224)/2))
    h_top = h_bot + 224
    w_left = int(np.ceil((width-224)/2))
    w_right = w_left + 224
    img = img[:,:,h_bot:h_top,w_left:w_right]
    #print(img.shape)
    return torch.tensor(img).float()

def diff(k=0):
    transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data1 = Mydata(img_dir,lbl_dir,transform)[k][0]
    data2 = rescale_by_hand(k)
    print(type(data1))
    print(type(data2))
    d = data2 - data1
    
    print("Tensor diff (with bilinear resize): ", d)
    print("Squared sum diff (with bilinear resize): ", torch.sum(d**2))
    
def diff_no_bilinear(k=0):
    transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data1 = Mydata(img_dir,lbl_dir,transform)[k][0]
    data2 = rescale_by_hand(k,False)
    print(type(data1))
    print(type(data2))
    d = data2 - data1
    
    print("Tensor diff(without bilinear resize): ", d)
    print("Squared sum diff(without bilinear resize): ", torch.sum(d**2))
    
def diff_normalization(model,n=250):     # where n is the number of images evaluated
    
    transform1 = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # no normalization
    transform2 = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])
    
    
    data1 = Mydata(img_dir,lbl_dir,transform1)
    data2 = Mydata(img_dir,lbl_dir,transform2)
    
    correct = 0
    for i in range(n):
        img,lbl = data1[i]
        pred = model(torch.unsqueeze(img,0))
        _, index = torch.max(pred,1)
        if lbl == index[0].item():
            correct+=1
    norm_acc = correct/n
    print("Accuracy of prediction on normalized data is " + str(norm_acc))
    
    correct = 0
    for i in range(n):
        img,lbl = data2[i]
        pred = model(torch.unsqueeze(img,0))
        _, index = torch.max(pred,1)
        if lbl == index[0].item():
            correct+=1
    unnorm_acc = correct/n
    print("Accuracy of prediction on un-normalized data is " + str(unnorm_acc))
    
    diff_acc = norm_acc - unnorm_acc
    print("The difference in accuracy/loss over " + str(n) + " images is " + str(diff_acc))
    
def five_crop(model,n=250):     # where n is the number of images
    transform = transforms.Compose([
            transforms.Resize(280),
            transforms.FiveCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])

    data = Mydata(img_dir,lbl_dir,transform)
    
    correct = 0
    for i in range(n):     # iterage over each image
        overall_score = 0
        for j in range(5):     # iterate over each crop            
            img,lbl = data[i][0][j], data[i][1]
            pred = model(torch.unsqueeze(img,0))
            score = torch.nn.functional.softmax(pred, dim=1)[0]
            overall_score += score
            
        overall_score = overall_score/5
        _, index = torch.max(overall_score,0)
        
        if lbl == index.item():
            correct+=1
            
    norm_acc = correct/n
    print("Accuracy of prediction is " + str(norm_acc))
    
    
def crop_330(model1, model2):
    transform = transforms.Compose([
            transforms.Resize(400),
            transforms.FiveCrop(330),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])

    data=Mydata(img_dir,lbl_dir,transform)

    correct = 0
    for i in range(n):     # iterage over each image
        overall_score = 0
        for j in range(5):     # iterate over each crop            
            img,lbl = data[i][0][j], data[i][1]
            pred = model(torch.unsqueeze(img,0))
            score = torch.nn.functional.softmax(pred, dim=1)[0]
            overall_score += score

        overall_score = overall_score/5
        _, index = torch.max(overall_score,0)

        if lbl == index.item():
            correct+=1

    norm_acc = correct/n

    print("Accuracy of prediction on resnet is " + str(norm_acc))
    
    correct = 0
    for i in range(n):     # iterage over each image
        overall_score = 0
        for j in range(5):     # iterate over each crop            
            img,lbl = data[i][0][j], data[i][1]
            pred = model2(torch.unsqueeze(img,0))
            score = torch.nn.functional.softmax(pred, dim=1)[0]
            overall_score += score

        overall_score = overall_score/5
        _, index = torch.max(overall_score,0)

        if lbl == index.item():
            correct+=1

    norm_acc = correct/n

    print("Accuracy of prediction on mobilenet is " + str(norm_acc))
    

def Qn1(model):
    print("Qn 1")
    print("\n")
    diff()
    diff_no_bilinear()
    diff_normalization(model)
    print("\n")    
    
def Qn2(model):
    print("Qn 2")
    print("\n")
    five_crop(model)
    print("\n")
    
    
def Qn3(model1, model2):
    print("Qn 3")
    print("\n")
    crop_330(model1, model2)
    print("\n")
    
if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    model.eval()
    Qn1(model)
    Qn2(model)
    model2 = models.mobilenet_v2(pretrained=True)
    model2.eval()
    Qn3(model,model2)
