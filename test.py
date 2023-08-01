import torch
import torch.nn as nn

from PIL import Image
import torchvision.transforms as tfs 
import os
import scipy.io as io
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image  
from sklearn.metrics import confusion_matrix
from src.resunet import ResUnet

LABELS = ["Building", "Non-building"]


def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    # print("Confusion matrix :")
    # print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100/float(total)
    # print("%d pixels processed" % (total))
    # print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    # for l_id, score in enumerate(Acc):
    #     print("%s: %.4f" % (label_values[l_id], score))
    # print("---")

    # Compute MIoU coefficient
    III = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    # print(MIoU)
    MIoU = np.nanmean(III[:5])
    # print('mean MIoU: %.4f' % (MIoU))
    # print("---")
    IoU=cm[1][1]/(cm[1][1]+cm[0][1]+cm[1][0])
    Acc = (cm[0][0]+cm[1][1])/(cm[1][1]+cm[0][1]+cm[1][0]+cm[0][0])
    return Acc,IoU
def start_point(size, split_size, overlap = 0.0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points



def normalize(img):
    min = img.min()
    max = img.max()
    x = (img - min) / (max - min) 
    return x
def val_multi(test_path,label_path):
    TP , FN, FP, TN= 0, 0,0,0

    for i in (range(len(os.listdir(test_path)))):

        # res=np.array(Image.open(test_path+str(i+1)+".png")).astype(np.int64)
        # ref=np.array(Image.open(label_path+"test_"+str(i+1)+".png")).astype(np.int64)
        res=np.array(Image.open(test_path+"split"+"_"+str(i+1)+".png")).astype(np.float32)
        ref=np.array(Image.open(label_path+"split"+"_"+str(i+1)+".png")).astype(np.float32)
        ref=ref[:,:,0]
        res[res==255]=1
        ref[ref==255]=1

        TP = TP+np.sum(res*ref==1)
        FN = FN+np.sum(ref*(1-res)==1)
        FP =FP+ np.sum(res*(1-ref)==1)
        TN = TN+np.sum((1-res)*(1-ref)==1)
    # print(f'TP={TP} | TN={TN} | FP={FP} | FN={FN}')

    Accu=(TP+TN)/(TP+TN+FP+FN)
    Precision=(TP)/(TP+FP)
    Recall=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    Sensitivity = TP/(TP+FN)
    F1=2*((Precision*Recall)/(Precision+Recall))

    pe=((TP+FN)*(TP+FP)+(TN+FP)*(TN+FN))/((TP+TN+FP+FN)**2) 
    kappa=(Accu-pe)/(1-pe)
    IoU=TP/(TP+FP+FN)

    # print(f'Accu={Accu} Precision={Precision} Recall={Recall} F1={F1} kappa={kappa} IoU={IoU} Specificity={Specificity} Sensitivity ={Sensitivity}')
    return IoU

def main():
    acc_list=0
    iou_list=0
    sites = ["Turkey", "Georgia", "Mexico", "Zimbabwe", "Kyrgyzstan", "Vietnam"]
    
    for site in sites:

        test_path="./data/LMdata5/Val/{}/image/".format(site)
        output_path = "./result/temp/"
        if os.path.exists(output_path):
            for i in os.listdir(output_path):
                os.remove(output_path+"/"+i)
        else:
            os.makedirs(output_path)

        client_result_path = "./result/fedproto2/{}/".format(site)
        if os.path.exists(client_result_path):
            for i in os.listdir(client_result_path):
                os.remove(client_result_path+"/"+i)
        else:
            os.makedirs(client_result_path)

        label_path="./data/LMdata5/Val/{}/label/".format(site)
        all_preds = []
        all_gts = []
        for i in range(len(os.listdir(test_path))):

            # img1=Image.open(test_path1+"test_"+str(i+1)+".png")
            # img2=Image.open(test_path2+"test_"+str(i+1)+".png")

            img=Image.open(test_path+"split_"+str(i+1)+".png")
            label=Image.open(label_path+"split_"+str(i+1)+".png")
            
            # label_ts=label_ts[0,:,:]
            # label_ts=label_ts.cuda()
            # img=normalize(np.array(img))

            model=ResUnet() 
            device = torch.device('cuda:0')
            model.to(device)
   
            model.load_state_dict(torch.load("./snapshot/2022-11-27_20_08_04_FedProto_LMdata5_42.pth"))#fedproto42 135

            model.eval()
        
            label = tfs.ToTensor()(label)
            label = label[0,:,:]
            split2 = tfs.ToTensor()(img)
            split2 = torch.unsqueeze(split2, dim=0)
            split2=split2.to(device)


            pred=model(split2)


            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            pred = torch.where(pred > 0.5, one, pred)
            pred = torch.where(pred <= 0.5, zero, pred)
            pred = pred.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))

            # cv2.imwrite(output_path+str(i+1)+".png", pred*255)
            # cv2.imwrite(output_path+"split"+"_"+str(i+1)+".png", pred*255)
            
            cv2.imwrite(client_result_path+"split"+"_"+str(i+1)+".png", pred*255)
            all_preds.append(pred)
            all_gts.append(label.detach().cpu().numpy())

          
        #先存下图片来再读图片算精度
        # acc = val_multi(output_path,label_path)
        #不通过存储的图片算精度
        acc,iou = metrics(np.concatenate([p.ravel() for p in all_preds]),
                    np.concatenate([p.ravel() for p in all_gts]).ravel())
        print(site,acc,iou)
        acc_list+= acc
        iou_list+= iou
    print("acc",acc_list/len(sites))
    print("Iou",iou_list/len(sites))

if __name__=="__main__":
    main()


