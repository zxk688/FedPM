#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random, re
from copy import deepcopy
from os import environ
import time
from datetime import timedelta
from collections import defaultdict
import torch.nn as nn

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import optimizers, schedulers
from options import args_parser
from utils_proto import average_updates, exp_details, get_acc_avg, printlog_stats,weighted_average_updates
from clientproto import Client
import os
from resunet import ResidualConv,Upsample
import torch.optim as optim

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label

def cal_weights(local_protos_list,global_protos):
    weights_list = []
    
    for idx in local_protos_list:
        proto_new=torch.zeros([2,32])
        glob_new = torch.zeros([2,32])
        local_protos = local_protos_list[idx]
        i=0
        for label in local_protos.keys():
            proto_new[i,:] = local_protos[label]
            glob_new[i,:]= global_protos[label][0]
            i=i+1
        

        # dis = torch.exp(-(proto_new - glob_new).pow(2).sum().sqrt())
        # weights_list.append(dis)
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        weights_list.append(torch.sum(cos(proto_new,glob_new)))
    sum_p = 0
    for j in range(len(weights_list)):
        sum_p+=weights_list[j]
    for j in range(len(weights_list)):
        weights_list[j] = weights_list[j]/sum_p
    return weights_list

class ResUnet(nn.Module):
    def __init__(self, channel=3, filters=[32,64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.up_residual_conv5 = ResidualConv(filters[0],filters[0],1,1)

        self.output_layer = nn.Sequential(
            ResidualConv(filters[0],1,1,1),
            # nn.LogSoftmax(dim=1),
            nn.Sigmoid()
        )

    def forward(self,m):

        # Encode1
        #stage1
        x1 = self.input_layer(m) + self.input_skip(m)#950*750

        #stage2
        x2 = self.residual_conv_1(x1) #475*375
        #stage3
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        x5 = self.bridge(x4)
        
        # Decode
        x5 = self.upsample_1(x5) #512*20*20
        
        x6 = torch.cat([x5,x4],dim=1) # 768*20*20
        x7 = self.up_residual_conv1(x6) #256*20*20

        x7 = self.upsample_2(x7) #256*40*40

        x8 = torch.cat([x7, x3], dim=1) #384*40*40

        x9 = self.up_residual_conv2(x8) #128*40*40

        x9 = self.upsample_3(x9) #128*80*80
        x10 = torch.cat([x9, x2], dim=1) #192*80*80

        x11 = self.up_residual_conv3(x10) #64*80*80

        x11 = self.upsample_4(x11) #64*160*160
        x12 = torch.cat([x11, x1], dim=1) #96*160*160

        x13 = self.up_residual_conv4(x12) 
        x14 = self.up_residual_conv5(x13)
        output = self.output_layer(x14)

        return output,x14

if __name__ == '__main__':
    # Start timer
    start_time = time.time()

    # Parse arguments and create/load checkpoint
    args = args_parser()
    if not args.resume:
        checkpoint = {}
        checkpoint['args'] = args
    else:
        checkpoint = torch.load(f'save/{args.name}')
        rounds = args.rounds
        iters = args.iters
        device =args.device
        args = checkpoint['args']
        args.resume = True
        args.rounds = rounds
        args.iters = iters
        args.device = device

    algo = "FedProto"
    method_name=algo

    ## Initialize RNGs and ensure reproducibility
    if args.seed is not None:
        environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        if not args.resume:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
        else:
            torch.set_rng_state(checkpoint['torch_rng_state'])
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['python_rng_state'])

    # Load datasets and splits
    acc_types =  ['train', 'valid']

    # Load model
    model = ResUnet().to(args.device)
    # Load optimizer and scheduler
    optimizer = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    scheduler = getattr(schedulers, args.sched)(optim, args.sched_args)
    
    clients_list =  ["Turkey","Georgia","Mexico","Zimbabwe","Kyrgyzstan","Vietnam"]
    dataset_path = "./data/LMdata6/"
    dataset_name = "LMdata6"

    if not args.resume:
        clients = []
        for client_id in clients_list:
            
            clients.append(Client(args=args,path_root=dataset_path,client_name=client_id))
        checkpoint['clients'] = clients
    else:
        clients = checkpoint['clients']

    # Set client sampling probabilities
    if args.vc_size is not None:
        # Proportional to the number of examples (FedVC)
        p_clients = np.array([len(client.loaders['train'].dataset) for client in clients])
        p_clients = p_clients / p_clients.sum()
    else:
        # Uniform
        p_clients = None

    # Determine number of clients to sample per round
    m = max(int(args.frac_clients * args.num_clients), 1)

    # Print experiment summary
    # summary = exp_details(args, model)
    # print('\n' + summary)
    torch.use_deterministic_algorithms(False)
    # Log experiment summary, client distributions, example images
    if not args.no_log:
        logger = SummaryWriter(f'runs/{args.name}')
        if not args.resume:
            input_size = (1,) + tuple([3,512,512])
            fake_input = torch.zeros(input_size).to(args.device)
            logger.add_graph(model, fake_input)
    else:
        logger = None

    if not args.resume:
        # Compute initial average accuracies
        acc_avg = get_acc_avg(acc_types, clients, model, args.device)
        acc_avg_best = acc_avg[acc_types[1]]

        # Print and log initial stats
        if not args.quiet:
            print('Training:')
            # print('    Round: 0' + (f'/{args.rounds}' if args.iters is None else ''))
        loss_avg, lr = torch.nan, torch.nan
        # printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, 0, 0, args.iters)
    else:
        acc_avg_best = checkpoint['acc_avg_best']

    init_end_time = time.time()

    # Train server model
    if not args.resume:
        last_round = -1
        iter = 0
        v = None
    else:
        last_round = checkpoint['last_round']
        iter = checkpoint['iter']
        v = checkpoint['v']

    global_protos = []
    for round in range(last_round + 1, args.rounds):
        if not args.quiet:
            print(f'    Round: {round+1}' + (f'/{args.rounds}' if args.iters is None else ''))

        # Sample clients每次选10个
        client_ids = np.random.choice(range(args.num_clients), m, replace=False, p=p_clients)

        # Train client models
        updates, num_examples, max_iters, loss_tot = [], [], 0, 0.
        local_protos = {} 
        for i, client_id in enumerate(client_ids):
            if not args.quiet: print(f'        Client: {client_id} ({i+1}/{m})')

            client_model = deepcopy(model)
            optimizer.__setstate__({'state': defaultdict(dict)})
            optimizer.param_groups[0]['params'] = list(client_model.parameters())

            client_update, client_num_examples, client_num_iters, client_loss,protos = clients[client_id].train(model=client_model, optim=optimizer, device=args.device,global_protos=global_protos)
            agg_protos = agg_func(protos)
            local_protos[i] = agg_protos
            if client_num_iters > max_iters: max_iters = client_num_iters

            if client_update is not None:
                updates.append(deepcopy(client_update))
                loss_tot += client_loss * client_num_examples
                num_examples.append(client_num_examples)

        iter += max_iters
        lr = optimizer.param_groups[0]['lr']
        
        # update global weights
        global_protos = proto_aggregation(local_protos)

        #local_protos 算权重
        weights_list = cal_weights(local_protos,global_protos)
        

        if len(updates) > 0:
            # Update server model
            # update_avg = average_updates(updates, num_examples)
            update_avg = weighted_average_updates(updates,num_examples,weights_list)
            if v is None:
                v = deepcopy(update_avg)
            else:
                for key in v.keys():
                    v[key] = update_avg[key] + v[key] * args.server_momentum
            #new_weights = deepcopy(model.state_dict())
            #for key in new_weights.keys():
            #    new_weights[key] = new_weights[key] - v[key] * args.server_lr
            #model.load_state_dict(new_weights)
            for key in model.state_dict():
                # model.state_dict()[key] = torch.from_numpy(model.state_dict()[key].cpu().numpy().astype(np.float32))
                if model.state_dict()[key].type()==v[key].type():
                    # print('eq!')
                    model.state_dict()[key] -= v[key] * args.server_lr

            # Compute round average loss and accuracies
            if round % args.server_stats_every == 0:
                loss_avg = loss_tot / sum(num_examples)
                acc_avg = get_acc_avg(acc_types, clients, model, args.device)

                if acc_avg[acc_types[1]] > acc_avg_best:
                    acc_avg_best = acc_avg[acc_types[1]]

        # Save checkpoint
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['optim_state_dict'] = optimizer.state_dict()
        checkpoint['sched_state_dict'] = scheduler.state_dict()
        checkpoint['last_round'] = round
        checkpoint['iter'] = iter
        checkpoint['v'] = v
        checkpoint['acc_avg_best'] = acc_avg_best
        checkpoint['torch_rng_state'] = torch.get_rng_state()
        checkpoint['numpy_rng_state'] = np.random.get_state()
        checkpoint['python_rng_state'] = random.getstate()
        os.makedirs('.save/{args.name}',exist_ok=True)
        torch.save(checkpoint, f'.save/{args.name}')
        # if (round+1)%10==0 and round!=0:
            # torch.save(model.state_dict(),"./snapshot/"+time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))+"_"+method_name+"_"+dataset_name+"_"+str(round+1)+".pth")
        # Print and log round stats
        # if acc_avg['valid']>0.51:
            # torch.save(model.state_dict(),"./snapshot/"+time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))+"_"+method_name+"_"+dataset_name+"_"+str(round+1)+".pth")
        if round % args.server_stats_every == 0:
            printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, round+1, iter, args.iters)

        # Stop training if the desired number of iterations has been reached
        if args.iters is not None and iter >= args.iters: break

        # Step scheduler
        if type(scheduler) == schedulers.plateau_loss:
            scheduler.step(loss_avg)
        else:
            scheduler.step()
            # scheduler.step(loss_avg)

    train_end_time = time.time()

    # Compute final average test accuracy
    acc_avg = get_acc_avg(['valid'], clients, model, args.device)

    test_end_time = time.time()

    # Print and log test results
    print('\nResults:')
    print(f'    Average test accuracy: {acc_avg["valid"]:.3%}')
    print(f'    Train time: {timedelta(seconds=int(train_end_time-init_end_time))}')
    print(f'    Total time: {timedelta(seconds=int(time.time()-start_time))}')

    if logger is not None: logger.close()
