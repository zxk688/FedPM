#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) 2022  Gabriele Cazzato

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

import io, re
from copy import deepcopy
from contextlib import redirect_stdout

import torch
from torch.nn import CrossEntropyLoss,BCELoss
from torchinfo import summary

import optimizers, schedulers
from sklearn.metrics import confusion_matrix
import numpy as np
# import cv2
types_pretty = {'train': 'training', 'valid': 'validation', 'test': 'test'}

def metrics(predictions, gts, label_values=["Building", "Non-building"]):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))
    # print(cm)
    # Compute global accuracy
    # total = sum(sum(cm))
    # accuracy = sum([cm[x][x] for x in range(len(cm))])
    # accuracy =  accuracy/ float(total)

    accuracy = cm[1][1]/(cm[1][1]+cm[0][1]+cm[1][0]+1e-10)#Iou
    return accuracy

class Scheduler():
    def __str__(self):
        sched_str = '%s (\n' % self.name
        for key in vars(self).keys():
            if key != 'name':
                value = vars(self)[key]
                if key == 'optimizer': value = str(value).replace('\n', '\n        ').replace('    )', ')')
                sched_str +=  '    %s: %s\n' % (key, value)
        sched_str += ')'
        return sched_str

def weighted_average_updates(w, n_k,weights):
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key]*weights[0]*len(w), n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key]*weights[i]*len(w), alpha=n_k[i])
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    return w_avg

def average_updates(w, n_k):
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    return w_avg

def inference(model, loader, device):
    if loader is None:
        return None, None

    criterion = BCELoss().to(device)
    loss, total, correct = 0., 0, 0
    model.eval()
    
    with torch.no_grad():
        all_preds = []
        all_gts = []
        for batch, (examples, labels) in enumerate(loader):
            examples, labels = examples.to(device), labels.to(device)
            log_probs,_ = model(examples)
            log_probs = torch.squeeze(log_probs,dim=1)
            labels = torch.squeeze(labels,dim=1)
            loss += criterion(log_probs, labels).item() * labels.shape[0]
            
            zero = torch.zeros_like(log_probs)
            one = torch.ones_like(log_probs)
            log_probs = torch.where(log_probs > 0.5, one, log_probs)
            log_probs = torch.where(log_probs <= 0.5, zero, log_probs)

            pred = log_probs.detach().cpu().numpy()
            # cv2.imwrite("result/val_res/"+"split"+"_"+str(batch+1)+".png", (pred.squeeze(0))*255)
            labels = labels.detach().cpu().numpy().astype(np.float32)
            
            # correct += metrics(pred.ravel(),
            #         labels.ravel())
            all_preds.append(pred)
            all_gts.append(labels)

            # _, pred_labels = torch.max(log_probs, 1)
            # pred_labels = pred_labels.view(-1)
            # correct += torch.sum(torch.eq(log_probs, labels)).item()

            total += labels.shape[0]
            
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
        np.concatenate([p.ravel() for p in all_gts]).ravel())
    # accuracy = correct/total
    loss /= total

    return accuracy, loss

# def inference(model, loader, device):
#     if loader is None:
#         return None, None

#     criterion = CrossEntropyLoss().to(device)
#     loss, total, correct = 0., 0, 0
#     model.eval()
#     with torch.no_grad():
#         for batch, (examples, labels) in enumerate(loader):
#             examples, labels = examples.to(device), labels.to(device)
#             log_probs = model(examples)
#             loss += criterion(log_probs, labels).item() * len(labels)
#             _, pred_labels = torch.max(log_probs, 1)
#             pred_labels = pred_labels.view(-1)
#             correct += torch.sum(torch.eq(pred_labels, labels)).item()
#             total += len(labels)

#     accuracy = correct/total
#     loss /= total

#     return accuracy, loss

def get_acc_avg(acc_types, clients, model, device):
    acc_avg = {}
    for type in acc_types:
        acc_avg[type] = 0.
        num_examples = 0
        for client_id in range(len(clients)):
            acc_client, _ = clients[client_id].inference(model, type=type, device=device)
            if acc_client is not None:
                acc_avg[type] += acc_client * len(clients[client_id].loaders[type].dataset)
                num_examples += len(clients[client_id].loaders[type].dataset)
        acc_avg[type] = acc_avg[type] / num_examples if num_examples != 0 else None
        # acc_avg[type] = acc_avg[type] /
    return acc_avg

def printlog_stats(quiet, logger, loss_avg, acc_avg, acc_types, lr, round, iter, iters):
    if not quiet:
        print(f'        Iteration: {iter}', end='')
        if iters is not None: print(f'/{iters}', end='')
        print()
        print(f'        Learning rate: {lr}')
        print(f'        Average running loss: {loss_avg:.6f}')
        for type in acc_types:
            print(f'        Average {types_pretty[type]} accuracy: {acc_avg[type]:.3%}')

    if logger is not None:
        logger.add_scalar('Learning rate (Round)', lr, round)
        logger.add_scalar('Learning rate (Iteration)', lr, iter)
        logger.add_scalar('Average running loss (Round)', loss_avg, round)
        logger.add_scalar('Average running loss (Iteration)', loss_avg, iter)
        for type in acc_types:
            logger.add_scalars('Average accuracy (Round)', {types_pretty[type].capitalize(): acc_avg[type]}, round)
            logger.add_scalars('Average accuracy (Iteration)', {types_pretty[type].capitalize(): acc_avg[type]}, iter)
        logger.flush()

def exp_details(args, model):
    if args.device == 'cpu':
        device = 'CPU'
    else:
        device = str(torch.cuda.get_device_properties(args.device))
        device = (', ' + re.sub('_CudaDeviceProperties\(|\)', '', device)).replace(', ', '\n            ')

    input_size = (args.train_bs,) + tuple([3,512,512])
    summ = str(summary(model, input_size, depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device=args.device))
    summ = '        ' + summ.replace('\n', '\n        ')

    optimizer = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    scheduler = getattr(schedulers, args.sched)(optimizer, args.sched_args)

    if args.centralized:
        algo = 'Centralized'
    else:
        if args.fedsgd:
            algo = 'FedSGD'
        else:
            algo = 'FedAvg'
        if args.server_momentum:
            algo += 'M'
        if args.fedir:
            algo += ' + FedIR'
        if args.vc_size is not None:
            algo += ' + FedVC'
        if args.mu:
            algo += ' + FedProx'
        if args.drop_stragglers:
            algo += ' (Drop Stragglers)'

    f = io.StringIO()
    with redirect_stdout(f):


        print('    Model:')
        print(summ)
        print()

        print('    Other:')
        print(f'        Test batch size: {args.test_bs}')
        print(f'        Random seed: {args.seed}')
        print(f'        Device: {device}')

    return f.getvalue()
