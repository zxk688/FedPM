import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from dataloader_all_centerlized import Dataset
from dataloader_cent import Dataset as singDataset
import time
from tensorboardX import SummaryWriter
import dataloader
from resunet import ResUnet
from SegNet import SegNet
import numpy as np
# from utils import *
from sklearn.metrics import confusion_matrix
LABELS = ["Building", "Non-building"]
import os
import cv2

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
    Acc = (cm[0][0]+cm[1][1])/(cm[1][1]+cm[0][1]+cm[1][0]+cm[0][0])
    IoU=cm[1][1]/(cm[1][1]+cm[0][1]+cm[1][0])
    return Acc,IoU


batch_size = 2
epoch = 40
base_lr = 1e-2
save_iter =epoch
set_snapshot_dir = "./snapshot/"
set_num_workers = 4
#set_momentum = 0.9
#set_weight_decay = 0.001
eval=True


def loss_calc(pred,label):
    label = torch.squeeze(label,dim=1)
    pred = torch.squeeze(pred,dim=1)
    loss = nn.BCELoss()#二分类交叉熵
    return loss(pred,label)



def main():
    # writer=SummaryWriter(comment="detrcd on datasetLM enc_8")#用来记录存储loss并打印在tensorboard
    device = torch.device('cuda:0')#指定用哪个显卡
    # clients = ["TaiwanChina","Vietnam",'Mexico','Turkey','Zimbabwe','Georgia']
    clients = ["Turkey","Georgia","Mexico","Zimbabwe","Kyrgyzstan","Vietnam"]
    # clients = ['Georgia']
    

    model = ResUnet()

    model.to(device)
    trainloader=data.DataLoader(
            Dataset(path_root="./data/LMdata5/",mode="Central"),
                batch_size=batch_size,shuffle=False,num_workers=set_num_workers,pin_memory=True)
    optimizer = optim.SGD(model.parameters(), lr=base_lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',verbose=True,patience=5,cooldown=3,min_lr=1e-8,factor=0.5)

    for i in range(epoch):
        torch.cuda.empty_cache()
        loss_list=[]
        model.train()

        for a,batch in enumerate((trainloader)):
                # print(i,batch)

                    
            optimizer.zero_grad()
            img, label = batch
            img = img.to(device)#数据加载到显存
            label = label.to(device)
            pred = model(img)#前向传播
            loss = loss_calc(pred,label)
            loss_list.append(loss.item())#用一个list来存loss
            loss.backward()#loss后向传播
            optimizer.step()#优化器必须更新
        scheduler.step(sum(loss_list)/len(loss_list))#每个epoch记录一次平均的loss
        lr = optimizer.param_groups[0]['lr']
        print(time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))+f', epoch={i} | loss={sum(loss_list)/len(loss_list):.7f} | lr={lr:.7f}')#显示的也是平均loss
    
        if (i+1)%save_iter==0 and i!=0:
            acc_list=0
            iou_list=0
            for client_name in clients:
                
                evalloader=data.DataLoader(
                            singDataset(path_root="./data/LMdata5/",mode="Val",client_name=client_name),
                            batch_size=1,shuffle=False,num_workers=set_num_workers,pin_memory=True)
                # optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0005)   
                client_result_path = "./result/centralized_all/{}/".format(client_name)
                if os.path.exists(client_result_path):
                    for i in os.listdir(client_result_path):
                        os.remove(client_result_path+"/"+i)
                else:
                    os.makedirs(client_result_path)

                
                    # writer.add_scalar('scalar/train_loss',sum(loss_list)/len(loss_list),i)#用来存储loss并打印在tensorboard
                    
                    # if (i+1)%save_iter==0 and i!=0:
                        # torch.save(model.state_dict(),set_snapshot_dir+time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))+"_centralized_"+str(i+1)+".pth")#在一定的epoch存储模型
                        # print(f'model saved at epoch{i}')
                
                
                # model2=ResUnet() 
                # device = torch.device('cuda:0')
                # model2.to(device)
                    
                # model2.load_state_dict(torch.load(set_snapshot_dir+"centralized_"+client_name+"2.pth"))
                model.eval()

                all_preds = []
                all_gts = []
                for a,batch in enumerate((evalloader)):
                    img, label = batch
                    img = img.to(device)
                    label = label.to(device)
                    pred = model(img)
                            
                    zero = torch.zeros_like(pred)
                    one = torch.ones_like(pred)
                    pred = torch.where(pred > 0.5, one, pred)
                    pred = torch.where(pred <= 0.5, zero, pred)
                    pred = pred.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
                    cv2.imwrite(client_result_path+"split"+"_"+str(a+1)+".png", pred*255)
                    all_preds.append(pred)
                    all_gts.append(label.detach().cpu().numpy())
                accuracy,iou = metrics(np.concatenate([p.ravel() for p in all_preds]),
                                np.concatenate([p.ravel() for p in all_gts]).ravel())
                acc_list+= accuracy
                iou_list+= iou

                print(client_name,accuracy,iou)
                torch.cuda.empty_cache()

                # del model

            print("acc",acc_list/len(clients))
            print("Iou",iou_list/len(clients))  


    torch.save(model.state_dict(),set_snapshot_dir+"centralized_all.pth")#在一定的epoch存储模型
    
            

if __name__=="__main__":
    main()


    
    
