from PIL import Image
import os
import numpy as np

def val_sing(test_path,label_path):



    res=np.array(Image.open(test_path)).astype(np.int64)
    ref=np.array(Image.open(label_path)).astype(np.int64)

    res[res==255]=1
    ref[ref==255]=1

    TP = np.sum(res*ref==1)
    FN = np.sum(ref*(1-res)==1)
    FP = np.sum(res*(1-ref)==1)
    TN = np.sum((1-res)*(1-ref)==1)

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

    print(f'Accu={Accu} Precision={Precision} Recall={Recall} F1={F1} kappa={kappa} IoU={IoU} Specificity={Specificity} Sensitivity ={Sensitivity}')
    return [Accu,Precision,Recall,F1,kappa,IoU,Specificity,Sensitivity]

def val_multi(test_path,label_path):
    TP , FN, FP, TN= 0, 0,0,0
    sites = ["Turkey","Georgia","Mexico","Zimbabwe","Kyrgyzstan","Vietnam"]
    for client_name in sites:
        # client_name = os.listdir(test_path)[i]
        for j in os.listdir(os.path.join(test_path,client_name)):
            # res=np.array(Image.open(test_path+str(i+1)+".png")).astype(np.int64)
            # ref=np.array(Image.open(label_path+"test_"+str(i+1)+".png")).astype(np.int64)
            # print(test_path+client_name+"/"+j)
            res=np.array(Image.open(test_path+client_name+"/"+j)).astype(np.int64)
            ref=np.array(Image.open(label_path+client_name+"/label/"+j)).astype(np.int64)
            ref = ref[:,:,0]
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

    print(f'Accu={Accu} Precision={Precision} Recall={Recall} F1={F1} kappa={kappa} IoU={IoU} Specificity={Specificity} Sensitivity ={Sensitivity}')


if __name__=="__main__":
    test_path="./result/fedavg2/"
    label_path="./data/LMdata5/Val/"
    val_multi(test_path,label_path)
