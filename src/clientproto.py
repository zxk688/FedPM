#!/usr/bin/env python
# -*- coding: utf-8 -*-


from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import CrossEntropyLoss,BCELoss,NLLLoss,MSELoss,SmoothL1Loss
from torch.utils.data import DataLoader, Subset

from utils_proto import inference
from dataloader import Dataset

def cal_prototype(feature_maps,labels,agg_protos_label):
#对一个batch 分开算prototype
    
    # c1=torch.zeros(32).cuda()
    # c2=torch.zeros(32).cuda()
    # agg_protos_label = {}
    # num1, num2 = 0, 0 
    for i in range(feature_maps.shape[0]):
        
        cennter_feat_S  =feature_maps[i]#32,512,512
        label_g = torch.unique(labels[i])
        for j in label_g:

            mask_feat_S = (labels[i] == j).float()#1,512,512


            num =  (mask_feat_S.sum(-1).sum(-1) + 1e-6)

            sum_c = ((mask_feat_S * cennter_feat_S).sum(-1).sum(-1))


            if j.item() in agg_protos_label:
                agg_protos_label[j.item()].append(sum_c/num)
            else:
                agg_protos_label[j.item()] = [sum_c/num]

        # if len(torch.unique(labels[i]))==2:
        #     c1 = c1 / num1
        #     c2 = c2 / num2
        # elif (torch.unique(labels[i]))==0:

        # elif (torch.unique(labels[i]))==1:

    return agg_protos_label
    # c1 = c1 / num1
    # c2 = c2 / num2
    # return c1.unsqueeze(0), c2.unsqueeze(0)

def cal_prototype_batch(feature_maps,labels):
#对一个batch算prototype
    
    c1=torch.zeros(32).cuda()
    c2=torch.zeros(32).cuda()
    num1, num2 = 0, 0 
    for i in range(feature_maps.shape[0]):

        cennter_feat_S  =feature_maps[i]#32,512,512
        # size_f = (im1.shape[2], im1.shape[3])
        # fea_S = nn.Upsample(size_f, mode='nearest')(gt.unsqueeze(1).float()).expand(im1.size())
        
        mask_feat_S = (labels[i] == 1).float()#1,512,512

        num1 = num1 + ((1 - mask_feat_S).sum(-1).sum(-1) + 1e-6)
        num2 = num2 + (mask_feat_S.sum(-1).sum(-1) + 1e-6)
        c1 = c1 + (((1 - mask_feat_S) * cennter_feat_S ).sum(-1).sum(-1))
        c2 = c2 + ((mask_feat_S * cennter_feat_S).sum(-1).sum(-1))


    c1 = c1 / num1
    c2 = c2 / num2
    return c1.unsqueeze(0), c2.unsqueeze(0)

class Client(object):
    def __init__(self, args, path_root,client_name):
        self.args = args

        # Create dataloaders
        self.train_bs = self.args.train_bs 
        self.loaders = {}
        self.loaders['train'] = DataLoader(Dataset(path_root=path_root,mode="Train",client_name=client_name), batch_size=self.train_bs, shuffle=True) 
        self.loaders['valid'] = DataLoader(Dataset(path_root=path_root,mode="Val",client_name=client_name), batch_size=self.train_bs, shuffle=True) 
        # DataLoader(Subset(datasets['valid'], idxs['valid']), batch_size=args.test_bs, shuffle=False) if idxs['valid'] is not None and len(idxs['valid']) > 0 else None
        # self.loaders['test'] = DataLoader(Subset(datasets['test'], idxs['test']), batch_size=args.test_bs, shuffle=False) if len(idxs['test']) > 0 else None

        # Set criterion
        # if args.fedir:
        #     # Importance Reweighting (FedIR)
        #     labels = set(datasets['train'].targets)
        #     p = torch.tensor([(torch.tensor(datasets['train'].targets) == label).sum() for label in labels]) / len(datasets['train'].targets)
        #     q = torch.tensor([(torch.tensor(datasets['train'].targets)[idxs['train']] == label).sum() for label in labels]) / len(torch.tensor(datasets['train'].targets)[idxs['train']])
        #     weight = p/q
        # else:
        #     # No Importance Reweighting
        #     weight = None
        weight = None
        self.criterion = BCELoss()

    def train(self, model, optim, device,global_protos):
        # Drop client if train set is empty
        if self.loaders['train'] is None:
            if not self.args.quiet: print(f'            No data!')
            return None, 0, 0, None

        # Determine if client is a straggler and drop it if required
        straggler = np.random.binomial(1, self.args.hetero)
        if straggler and self.args.drop_stragglers:
            if not self.args.quiet: print(f'            Dropped straggler!')
            return None, 0, 0, None
        epochs = np.random.randint(1, self.args.epochs) if straggler else self.args.epochs

        # Create training loader
        if self.args.vc_size is not None:
            # Virtual Client (FedVC)
            if len(self.loaders['train'].dataset) >= self.args.vc_size:
                train_idxs_vc = torch.randperm(len(self.loaders['train'].dataset))[:self.args.vc_size]
            else:
                train_idxs_vc = torch.randint(len(self.loaders['train'].dataset), (self.args.vc_size,))
            train_loader = DataLoader(Subset(self.loaders['train'].dataset, train_idxs_vc), batch_size=self.train_bs, shuffle=True)
        else:
            # No Virtual Client
            train_loader = self.loaders['train']

        client_stats_every = self.args.client_stats_every if self.args.client_stats_every > 0 and self.args.client_stats_every < len(train_loader) else len(train_loader)


        #fedProto
        model.train()
        criterion = BCELoss().to(device)

        # Train new model
        model.to(device)
        self.criterion.to(device)
        model.train()
        model_server = deepcopy(model)
        iter = 0
        for epoch in range(epochs):
            agg_protos_label = {}
            loss_sum, loss_num_images, num_images = 0., 0, 0
            for batch, (examples, label_g) in enumerate(train_loader):
                
                examples, labels = examples.to(device), label_g.to(device)
                model.zero_grad()
                log_probs,deep_feas= model(examples)
                loss1 = criterion(log_probs, labels)

                # loss_mse = MSELoss()
                loss_mse = SmoothL1Loss()
                #deep_feas to prototypes
                c1,c2 =cal_prototype_batch(deep_feas,labels)
                protos =  torch.cat([c1,c2])

                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = deepcopy(protos.data)
                    i = 0
                    for label in torch.unique(labels):
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2 * 1

                loss_sum += loss.item() * labels.shape[0]
                loss_num_images += labels.shape[0]
                num_images += labels.shape[0]

                loss.backward()
                optim.step()

                agg_protos_label = cal_prototype(deep_feas,labels,agg_protos_label)

                # for i in range(labels.shape[0]):
                #     for j in range(len(torch.unique(label_g[i]))+1):
                #         if j in agg_protos_label:
                #             agg_protos_label[j].append(protos[j,:])
                #         else:
                #             agg_protos_label[j] = protos[j,:]
                    # if torch.unique(label_g[i]).item() in agg_protos_label:
                    #     agg_protos_label[label_g[i].item()].append(protos[i,:])
                    # else:
                    #     agg_protos_label[label_g[i].item()] = [protos[i,:]]

                # After client_stats_every batches...
                if (batch + 1) % client_stats_every == 0:
                    # ...Compute average loss
                    loss_running = loss_sum / loss_num_images

                    # ...Print stats
                    if not self.args.quiet:
                        print('            ' + f'Epoch: {epoch+1}/{epochs}, '\
                                               f'Batch: {batch+1}/{len(train_loader)} (Image: {num_images}/{len(train_loader.dataset)}), '\
                                               f'Loss: {loss.item():.6f}, ' \
                                               f'Running loss: {loss_running:.6f}')

                    loss_sum, loss_num_images = 0., 0

                iter += 1

        # Compute model update
        model_update = {}
        for key in model.state_dict():
            model_update[key] = torch.sub(model_server.state_dict()[key], model.state_dict()[key])

        return model_update, len(train_loader.dataset), iter, loss_running,agg_protos_label

    def inference(self, model, type, device):
        return inference(model, self.loaders[type], device)
