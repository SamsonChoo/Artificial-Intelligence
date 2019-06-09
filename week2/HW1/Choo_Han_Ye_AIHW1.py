# Python 3.5.2
# pip3 install torch
# pip3 install numpy
# pip3 install sklearn
# pip3 install os

import torch
import numpy as np
from sklearn.svm import SVC
import os

def savefile(data_train_feat, data_val_feat, data_test_feat):
    np.save(('train.npy'), data_train_feat)
    np.save(('validate.npy'), data_val_feat)
    np.save(('test.npy'), data_test_feat)
          
def prepare_data():
    data=open("trainset_gt_annotations.txt", 'r')

    spring=[]
    summer=[]
    autumn=[]
    winter=[]

    for line in data:
        line_sep = line.split()
        if(line_sep[10]=='1'):
            spring.append((line_sep[0],line_sep[10:14]))
        elif(line_sep[11]=='1'):
            summer.append((line_sep[0],line_sep[10:14]))
        elif(line_sep[12]=='1'):
            autumn.append((line_sep[0],line_sep[10:14]))
        elif(line_sep[13]=='1'):
            winter.append((line_sep[0],line_sep[10:14]))

    spring=np.array(spring)
    summer=np.array(summer)
    autumn=np.array(autumn)
    winter=np.array(winter)
    
    spring_train,spring_validate,spring_test = np.split(spring, [int(len(spring)*0.6), int(len(spring)*0.7)])
    summer_train,summer_validate,summer_test = np.split(summer, [int(len(summer)*0.6), int(len(summer)*0.7)])
    autumn_train,autumn_validate,autumn_test = np.split(autumn, [int(len(autumn)*0.6), int(len(autumn)*0.7)])
    winter_train,winter_validate,winter_test = np.split(winter, [int(len(winter)*0.6), int(len(winter)*0.7)])

    spring_train_label,spring_validate_label,spring_test_label = [],[],[]
    summer_train_label,summer_validate_label,summer_test_label = [],[],[]
    autumn_train_label,autumn_validate_label,autumn_test_label = [],[],[]
    winter_train_label,winter_validate_label,winter_test_label = [],[],[]

    for i in spring_train:
        spring_train_label.append(i[1])
    for i in spring_validate:
        spring_validate_label.append(i[1])
    for i in spring_test:
        spring_test_label.append(i[1])

    for i in summer_train:
        summer_train_label.append(i[1])
    for i in summer_validate:
        summer_validate_label.append(i[1])
    for i in summer_test:
        summer_test_label.append(i[1])

    for i in autumn_train:
        autumn_train_label.append(i[1])
    for i in autumn_validate:
        autumn_validate_label.append(i[1])
    for i in autumn_test:
        autumn_test_label.append(i[1])

    for i in winter_train:
        winter_train_label.append(i[1])
    for i in winter_validate:
        winter_validate_label.append(i[1])
    for i in winter_test:
        winter_test_label.append(i[1])

    spring_train_label,spring_validate_label,spring_test_label = np.array(spring_train_label),np.array(spring_validate_label),np.array(spring_test_label)
    summer_train_label,summer_validate_label,summer_test_label = np.array(summer_train_label),np.array(summer_validate_label),np.array(summer_test_label)
    autumn_train_label,autumn_validate_label,autumn_test_label = np.array(autumn_train_label),np.array(autumn_validate_label),np.array(autumn_test_label)
    winter_train_label,winter_validate_label,winter_test_label = np.array(winter_train_label),np.array(winter_validate_label),np.array(winter_test_label)
    
    data_train = np.concatenate([spring_train, summer_train, autumn_train, winter_train])
    data_validate =  np.concatenate([spring_validate, summer_validate, autumn_validate, winter_validate])
    data_test =  np.concatenate([spring_test, summer_test, autumn_test, winter_test])

    label_train = np.concatenate([spring_train_label, summer_train_label, autumn_train_label, winter_train_label])
    label_validate =  np.concatenate([spring_validate_label, summer_validate_label, autumn_validate_label, winter_validate_label])
    label_test =  np.concatenate([spring_test_label, summer_test_label, autumn_test_label, winter_test_label])
    
    data_train_feat=[]

    feat_path = "imageclef2011_feats"

    for filename in os.listdir(feat_path):
        for v in data_train:
            if((filename.split(".")[0]+".jpg")==v[0]):
                data_train_feat.append(np.load(os.path.join(feat_path,filename)))

    data_train_feat = np.array(data_train_feat)
    
    data_val_feat=[]

    for filename in os.listdir(feat_path):
        for v in data_validate:
            if((filename.split(".")[0]+".jpg")==v[0]):
                data_val_feat.append(np.load(os.path.join(feat_path,filename)))

    data_val_feat = np.array(data_val_feat)
    
    data_test_feat=[]

    for filename in os.listdir(feat_path):
        for v in data_test:
            if((filename.split(".")[0]+".jpg")==v[0]):
                data_test_feat.append(np.load(os.path.join(feat_path,filename)))

    data_test_feat = np.array(data_test_feat)
    
    savefile(data_train_feat, data_val_feat, data_test_feat)
    
    spring_train_feat=[]
    for i in range(len(spring_train)):
        spring_train_feat.append(data_train_feat[i])
    spring_train_feat=np.array(spring_train_feat)
    
    summer_train_feat=[]
    for i in range(len(spring_train),len(summer_train)):
        summer_train_feat.append(data_train_feat[i])
    summer_train_feat=np.array(summer_train_feat)
    
    autumn_train_feat=[]
    for i in range(len(spring_train)+len(summer_train),len(autumn_train)):
        autumn_train_feat.append(data_train_feat[i])
    autumn_train_feat=np.array(autumn_train_feat)
    
    winter_train_feat=[]
    for i in range(len(spring_train)+len(summer_train)+len(autumn_train),len(winter_train)):
        winter_train_feat.append(data_train_feat[i])
    winter_train_feat=np.array(winter_train_feat)

    label_train_spring=[]
    for i in range(len(spring_train)):
        label_train_spring.append(['1'])
    for i in range(len(label_train)-len(spring_train)):
        label_train_spring.append(['0'])
    label_train_spring=np.array(label_train_spring)

    label_test_spring=[]
    for i in range(len(spring_test)):
        label_test_spring.append(['1'])
    for i in range(len(label_test)-len(spring_test)):
        label_test_spring.append(['0'])
    label_test_spring=np.array(label_test_spring)

    label_validate_spring=[]
    for i in range(len(spring_validate)):
        label_validate_spring.append(['1'])
    for i in range(len(label_validate)-len(spring_validate)):
        label_validate_spring.append(['0'])
    label_validate_spring=np.array(label_validate_spring)


    label_train_summer=[]
    for i in range(len(spring_train)):
        label_train_summer.append(['0'])
    for i in range(len(summer_train)):
        label_train_summer.append(['1'])
    for i in range(len(label_train)-len(spring_train)-len(summer_train)):
        label_train_summer.append(['0'])
    label_train_summer=np.array(label_train_summer)

    label_test_summer=[]
    for i in range(len(spring_test)):
        label_test_summer.append(['0'])
    for i in range(len(summer_test)):
        label_test_summer.append(['1'])
    for i in range(len(label_test)-len(spring_test)-len(summer_test)):
        label_test_summer.append(['0'])
    label_test_summer=np.array(label_test_summer)

    label_validate_summer=[]
    for i in range(len(spring_validate)):
        label_validate_summer.append(['0'])
    for i in range(len(summer_validate)):
        label_validate_summer.append(['1'])
    for i in range(len(label_validate)-len(spring_validate)-len(summer_validate)):
        label_validate_summer.append(['0'])
    label_validate_summer=np.array(label_validate_summer)


    label_train_autumn=[]
    for i in range(len(spring_train)):
        label_train_autumn.append(['0'])
    for i in range(len(summer_train)):
        label_train_autumn.append(['0'])
    for i in range(len(autumn_train)):
        label_train_autumn.append(['1'])
    for i in range(len(winter_train)):
        label_train_autumn.append(['0'])
    label_train_autumn=np.array(label_train_autumn)

    label_test_autumn=[]
    for i in range(len(spring_test)):
        label_test_autumn.append(['0'])
    for i in range(len(summer_test)):
        label_test_autumn.append(['0'])
    for i in range(len(autumn_test)):
        label_test_autumn.append(['1'])
    for i in range(len(winter_train)):
        label_test_autumn.append(['0'])
    label_test_autumn=np.array(label_test_autumn)

    label_validate_autumn=[]
    for i in range(len(spring_validate)):
        label_validate_autumn.append(['0'])
    for i in range(len(summer_validate)):
        label_validate_autumn.append(['0'])
    for i in range(len(autumn_validate)):
        label_validate_autumn.append(['1'])
    for i in range(len(winter_validate)):
        label_validate_autumn.append(['0'])
    label_validate_autumn=np.array(label_validate_autumn)


    label_train_winter=[]
    for i in range(len(label_train)-len(winter_train)):
        label_train_winter.append(['0'])
    for i in range(len(winter_train)):
        label_train_winter.append(['1'])
    label_train_winter=np.array(label_train_winter)

    label_test_winter=[]
    for i in range(len(label_test)-len(winter_test)):
        label_test_winter.append(['0'])
    for i in range(len(winter_test)):
        label_test_winter.append(['1'])
    label_test_winter=np.array(label_test_winter)

    label_validate_winter=[]
    for i in range(len(label_validate)-len(winter_validate)):
        label_validate_winter.append(['0'])
    for i in range(len(winter_validate)):
        label_validate_winter.append(['1'])
    label_validate_winter=np.array(label_validate_winter)
    
    data_feat={"train":data_train_feat, "validate":data_val_feat, "test":data_test_feat}
    return data_feat, label_train_spring, label_train_summer, label_train_autumn, label_train_winter, label_validate_spring, label_validate_summer, label_validate_autumn, label_validate_winter, label_test_spring, label_test_summer, label_test_autumn, label_test_winter
    
def train_model(data_feat, label_train_spring, label_train_summer, label_train_autumn, label_train_winter, label_validate_spring, label_validate_summer, label_validate_autumn, label_validate_winter):
    
    C = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10]

    class_accuracy = {}
    vanilla_accuracy = {}
    
    for constant in C:

        scoredict={}
        vanillatotal = 0
        vanillascore = 0

        clfspring = SVC(C=constant, kernel='linear')
        clfspring.fit(data_feat["train"],np.ravel(label_train_spring))
        predspring = clfspring.predict(data_feat["validate"])
        total=0
        score=0
        for pred, true in zip(predspring, label_validate_spring):
            total+=1
            vanillatotal += 1
            vanillascore += (pred==true)
            score+= (pred==true)
        scoredict["spring "+str(constant)]=score/total

        clfsummer = SVC(C=constant, kernel='linear')
        clfsummer.fit(data_feat["train"],np.ravel(label_train_summer))
        predsummer = clfsummer.predict(data_feat["validate"])
        total=0
        score=0
        for pred, true in zip(predsummer, label_validate_summer):
            total+=1
            vanillatotal += 1
            vanillascore += (pred==true)
            score+= (pred==true)
        scoredict["summer "+str(constant)]=score/total

        clfautumn = SVC(C=constant, kernel='linear')
        clfautumn.fit(data_feat["train"],np.ravel(label_train_autumn))
        predautumn = clfautumn.predict(data_feat["validate"])
        total=0
        score=0
        for pred, true in zip(predautumn, label_validate_autumn):
            total+=1
            vanillatotal += 1
            vanillascore += (pred==true)
            score+= (pred==true)
        scoredict["autumn "+str(constant)]=score/total

        clfwinter = SVC(C=constant, kernel='linear')
        clfwinter.fit(data_feat["train"],np.ravel(label_train_winter))
        predwinter = clfwinter.predict(data_feat["validate"])
        total=0
        score=0
        for pred, true in zip(predwinter, label_validate_winter):
            total+=1
            vanillatotal += 1
            vanillascore += (pred==true)            
            score+= (pred==true)
        scoredict["winter "+str(constant)]=score/total

        ave = 0
        for i in scoredict.values():
            ave+=i
        ave = ave/4
        class_accuracy[constant]=ave
        vanilla_accuracy[constant]=vanillascore/vanillatotal

    best_constant = max(class_accuracy, key=class_accuracy.get)
    
    print("class-wise accuracy: " + str(class_accuracy))
    print("vanilla accuracy: " + str(vanilla_accuracy))    
    print("Best C is "+ str(best_constant))
    return best_constant
    

def test_model(best_constant, data_feat, label_train_spring, label_train_summer, label_train_autumn, label_train_winter, label_validate_spring, label_validate_summer, label_validate_autumn, label_validate_winter, label_test_spring, label_test_summer, label_test_autumn, label_test_winter):
    feat_train_validate=np.concatenate([data_feat["train"], data_feat["validate"]])

    springlabels_train_validate=np.concatenate([label_train_spring, label_validate_spring])
    summerlabels_train_validate=np.concatenate([label_train_summer, label_validate_summer])
    autumnlabels_train_validate=np.concatenate([label_train_autumn, label_validate_autumn])
    winterlabels_train_validate=np.concatenate([label_train_winter, label_validate_winter])

    scoredict={}
    vanillatotal = 0
    vanillascore = 0
    
    clfspring = SVC(C=best_constant, kernel='linear')
    clfspring.fit(feat_train_validate,np.ravel(springlabels_train_validate))
    predspring = clfspring.predict(data_feat["test"])
    total=0
    score=0
    for pred, true in zip(predspring, label_test_spring):
        total+=1
        vanillatotal += 1
        vanillascore += (pred==true)        
        score+= (pred==true)
    scoredict["spring "]=score/total

    clfsummer = SVC(C=best_constant, kernel='linear')
    clfsummer.fit(feat_train_validate,np.ravel(summerlabels_train_validate))
    predsummer = clfsummer.predict(data_feat["test"])
    total=0
    score=0
    for pred, true in zip(predsummer, label_test_summer):
        total+=1
        vanillatotal += 1
        vanillascore += (pred==true)
        score+= (pred==true)
    scoredict["summer "]=score/total

    clfautumn = SVC(C=best_constant, kernel='linear')
    clfautumn.fit(feat_train_validate,np.ravel(autumnlabels_train_validate))
    predautumn = clfautumn.predict(data_feat["test"])
    total=0
    score=0
    for pred, true in zip(predautumn, label_test_autumn):
        total+=1
        vanillatotal += 1
        vanillascore += (pred==true)
        score+= (pred==true)
    scoredict["autumn "]=score/total

    clfwinter = SVC(C=best_constant, kernel='linear')
    clfwinter.fit(feat_train_validate,np.ravel(winterlabels_train_validate))
    predwinter = clfwinter.predict(data_feat["test"])
    total=0
    score=0
    for pred, true in zip(predwinter, label_test_winter):
        total+=1
        vanillatotal += 1
        vanillascore += (pred==true)
        score+= (pred==true)
    scoredict["winter "]=score/total

    ave = 0
    for i in scoredict.values():
        ave+=i
    ave = ave/4
    vanilla = vanillascore/vanillatotal
    
    print("Vanilla accuracy for test data is " + str(vanilla))    
    print("Class-wise average accuracy for test data is " + str(ave))

    
def run():
    data_feat, label_train_spring, label_train_summer, label_train_autumn, label_train_winter, label_validate_spring, label_validate_summer, label_validate_autumn, label_validate_winter, label_test_spring, label_test_summer, label_test_autumn, label_test_winter = prepare_data()
    
    best_constant = train_model(data_feat, label_train_spring, label_train_summer, label_train_autumn, label_train_winter, label_validate_spring, label_validate_summer, label_validate_autumn, label_validate_winter)
    
    test_model(best_constant, data_feat, label_train_spring, label_train_summer, label_train_autumn, label_train_winter, label_validate_spring, label_validate_summer, label_validate_autumn, label_validate_winter, label_test_spring, label_test_summer, label_test_autumn, label_test_winter)
    
if __name__ == '__main__':
    run()