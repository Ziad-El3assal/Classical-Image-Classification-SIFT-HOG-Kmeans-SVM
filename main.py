import os 
import numpy as np
from FeatureExtraction import Get_Discriptor
from Get_BOWs import getClassesHistogram,train_svm,ClassHistoGram
from Perdictions import NaivePredicions

imgs=[]
imgPath = 'seg_train\seg_train'
classes={}
for i in os.listdir(imgPath):
    classes[i]=[]
    for j in os.listdir(imgPath+'\\'+i):
        classes[i].append(imgPath+'\\'+i+'\\'+j)
# print(imgs)
SIFTDiscriptors = Get_Discriptor('SIFT')
CDiscriptors = {}
Discriptors = []
for i in classes:
    CDiscriptors[i]=[]
    for j in classes[i]:
        x=SIFTDiscriptors(j)
        Discriptors.append(x)
        CDiscriptors[i].append(x)
        
            
dis=[]
for i in Discriptors:
    try:
        for j in i:
            dis.append(j)
    except:
        pass
dis=np.array(dis)
print("---> ",dis.shape)
print("Clustering")
clf = train_svm(dis)
print("Trained")

CLH = getClassesHistogram(CDiscriptors,clf)
print(classes['buildings'][20])
img=NaivePredicions(classes['forest'][20],CLH,clf)

print(img)
