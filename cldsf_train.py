from __future__ import print_function
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
#from PreResNet import *
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import dataloader_clsdf as dataloader
from feature_model import FeaModel
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns 
import math

parser = argparse.ArgumentParser(description='PyTorch MSTAR Training')
parser.add_argument('--batch_size', default=16, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.6, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.2, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.3, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=1)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=4, type=int)
parser.add_argument('--num_warmup', default=4, type=int)
parser.add_argument('--data_path', default='./dataset/eoc2', type=str, help='path to dataset')
parser.add_argument('--dataset', default='MSTAR', type=str)
parser.add_argument('--flag', default=True, action='store_false', dest='flag', help='whether to use semi-supervised learning')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,scheduler,pre_data,pre_data1,duilie,flag=True):#
    net.train()
    net2.eval() #fix one network and train the other
    if flag==False:
        unlabeled_train_iter = iter(unlabeled_trainloader)    
        num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
        for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
            try:
                inputs_u, inputs_u2, inputs_u11, inputs_u22, _, _, inputs_u3, inputs_u4, inputs_u33, inputs_u44 = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u11, inputs_u22, _, _,inputs_u3, inputs_u4, inputs_u33, inputs_u44 = next(unlabeled_train_iter)               
            batch_size = inputs_x.size(0)
            
            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
            w_x = w_x.view(-1,1).type(torch.FloatTensor) 

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, inputs_u11, inputs_u22, inputs_u3, inputs_u4, inputs_u33, inputs_u44 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u11.cuda(), inputs_u22.cuda(), inputs_u3.cuda(), inputs_u4.cuda(), inputs_u33.cuda(), inputs_u44.cuda()
     
            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = net(inputs_u, inputs_u3)
                outputs_u12 = net(inputs_u2, inputs_u4)
                outputs_u21 = net2(inputs_u, inputs_u3)
                outputs_u22 = net2(inputs_u2, inputs_u4)            
                
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
                ptu = pu**(1/args.T) # temparature sharpening
                
                targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
                targets_u = targets_u.detach()       
                
                # label refinement of labeled samples
                outputs_x = net(inputs_x, inputs_x3)
                outputs_x2 = net(inputs_x2, inputs_x4)            
                
                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

                px = w_x*labels_x + (1-w_x)*px              
                ptx = px**(1/args.T) # temparature sharpening 
                        
                targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
                targets_x = targets_x.detach()       
            
            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)        
            l = max(l, 1-l)
                    
            all_inputs1 = torch.cat([inputs_x, inputs_x2], dim=0)
            all_inputs2 = torch.cat([inputs_x3, inputs_x4], dim=0)
            all_targets = torch.cat([targets_x, targets_x], dim=0)

            idx = torch.randperm(all_inputs1.size(0))

            input_a, input_b = all_inputs1, all_inputs1[idx]
            input_a1, input_b1 = all_inputs2, all_inputs2[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_input1 = l * input_a1 + (1 - l) * input_b1
            mixed_target = l * target_a + (1 - l) * target_b
                    
            logits = net(mixed_input, mixed_input1)
            logits_x = logits[:batch_size*2]
            logits_u = logits[batch_size*2:]
            
            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
            # regularization
            prior = torch.ones(args.num_class)/args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            loss = Lx + lamb * Lu  + penalty
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            sys.stdout.write('\r')

            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f'
                    %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
            sys.stdout.flush()
    elif flag==True: 
        ln = len(labeled_trainloader.dataset)
        un = len(unlabeled_trainloader.dataset)
        num1 = (ln + un)/args.num_class
        print(num1)
        unlabeled_train_iter = iter(unlabeled_trainloader)    
        num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
        for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
            try:
                inputs_u, inputs_u2, inputs_u11, inputs_u22, inputs_u111, inputs_u222, inputs_u3, inputs_u4, inputs_u33, inputs_u44 = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u11, inputs_u22, inputs_u111, inputs_u222, inputs_u3, inputs_u4, inputs_u33, inputs_u44 = next(unlabeled_train_iter)                
            batch_size = inputs_x.size(0)
            
            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
            w_x = w_x.view(-1,1).type(torch.FloatTensor) 
            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, inputs_u11, inputs_u22, inputs_u111, inputs_u222, inputs_u3, inputs_u4, inputs_u33, inputs_u44 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u11.cuda(), inputs_u22.cuda(), inputs_u111.cuda(), inputs_u222.cuda(), inputs_u3.cuda(), inputs_u4.cuda(), inputs_u33.cuda(), inputs_u44.cuda()

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = net(inputs_u, inputs_u3)
                outputs_u12 = net(inputs_u2, inputs_u4)
                outputs_u21 = net2(inputs_u, inputs_u3)
                outputs_u22 = net2(inputs_u2, inputs_u4)      
                
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                
                # label refinement of labeled samples
                outputs_x = net(inputs_x, inputs_x3)
                outputs_x2 = net(inputs_x2, inputs_x4)
           
                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                
                # JDA
                num = torch.ones(args.num_class).cuda().float()
                if epoch < 1 or epoch > 100:
                    ptu = pu**(1/args.T)
                    targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
                    targets_u = targets_u.detach()       
                    _, pseudo_label = torch.max(targets_u, dim=-1)
                    pseudo_label = pseudo_label.long()
  
                else:
                    #Distribution Alignment
                    pre_data1.append(labels_x.sum(0))
                    if len(pre_data1)>ln/args.batch_size:
                        while len(pre_data1)>ln/args.batch_size: 
                            pre_data1.pop(0)

                    prob_avg1 = torch.stack(pre_data1,dim=0).sum(0)/num1
                    prob_avg1 = num - prob_avg1
                    prob_avg1 = prob_avg1.cpu().numpy()
                    
                    if min(prob_avg1) < 0:
                        prob_avg1 = prob_avg1 - min(prob_avg1)
                    prob_avg1 = prob_avg1*1.4
                    prob_avg1[np.where(prob_avg1<=0.2)]=0.2
                    prob_avg1[np.where(prob_avg1>1.5)]=1.5
                    prob_avg1 = torch.tensor(prob_avg1).cuda()
                    pre_data.append(pu.mean(0))
                    if len(pre_data)>ln/args.batch_size:
                        while len(pre_data)>ln/args.batch_size: 
                            pre_data.pop(0)
                    prob_avg = torch.stack(pre_data,dim=0).mean(0)
                    pu = pu*prob_avg1/prob_avg
                    pt = pu**(1/args.T)
                    targets_u = pt / pt.sum(dim=1, keepdim=True)
                    pseudo_label = targets_u.detach()
                    targets_u = targets_u.detach()
                    _, pseudo_label = torch.max(pseudo_label, dim=-1)
                    pseudo_label = pseudo_label.long()
                    
                px = w_x*labels_x + (1-w_x)*px              
                ptx = px**(1/args.T) # temparature sharpening 
                        
                targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
                targets_x = targets_x.detach()       
            outputs_uxx1 = net(inputs_u11, inputs_u33)
            outputs_uyy1 = net(inputs_u22, inputs_u44)            
            outputs_u111 = net(inputs_u111, inputs_u3)
            outputs_u122 = net(inputs_u222, inputs_u4)
           
            px1 = (outputs_uxx1 + outputs_uyy1 + outputs_u111 + outputs_u122) / 4
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1-l)
            all_inputs1 = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_inputs2 = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0) 
            idx = torch.randperm(all_inputs1.size(0))

            input_a, input_b = all_inputs1, all_inputs1[idx]
            input_a1, input_b1 = all_inputs2, all_inputs2[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_input1 = l * input_a1 + (1 - l) * input_b1
            mixed_target = l * target_a + (1 - l) * target_b
                    
            logits = net(mixed_input, mixed_input1)
            logits_x = logits[:batch_size*2]
            logits_u = logits[batch_size*2:]
            
            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)

            prior = torch.ones(args.num_class)/args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            if epoch > 0:
                loss = Lx + lamb * Lu  + penalty + 0.2*(F.cross_entropy(px1, pseudo_label,reduction='none')).mean()#lambda * anchor
            else:
                loss = Lx + lamb * Lu  + penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                    %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
            sys.stdout.flush()
          
def warmup(epoch,net,optimizer,dataloader,flag=True):
    correct = 0
    total = 0

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, inputs1, labels, path, index) in enumerate(dataloader):      
        inputs, inputs1, labels = inputs.cuda(), inputs1.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs, inputs1)             
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 
        _, predicted = torch.max(outputs, 1)                       
        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()                 
        acc = 100.*correct/total

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
    sys.stdout.flush()
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))

def test(epoch,net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, targets) in enumerate(test_loader):
            inputs, inputs1, targets = inputs.cuda(), inputs1.cuda(), targets.cuda()
            predicted = []
            outputs1 = net1(inputs, inputs1)
            outputs2 = net2(inputs, inputs1)
            outputs = outputs1+outputs2 
            _, predicted = torch.max(outputs, 1)             
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def eval_train(model,all_loss,flag=True):    
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))  
    pred_target = torch.zeros(len(eval_loader.dataset))   
    class_indexs = []
    class_index = []
    noise_ids = torch.zeros(len(eval_loader.dataset))
    prob = []
    classes_loss = [[]]*args.num_class
    loss_index = [[]]*args.num_class
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, targets, index, noise_id) in enumerate(eval_loader):
            inputs, inputs1, targets = inputs.cuda(), inputs1.cuda(), targets.cuda() 
            outputs = model(inputs, inputs1) 
            loss = CE(outputs, targets)   
            class_indexs.append(index.cpu().numpy())
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
                pred_target[index[b]]=targets[b]
                noise_ids[index[b]]=noise_id[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())   
    class_indexs = np.concatenate(class_indexs) 
    all_loss.append(losses)

    if args.r<=0.8: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)

    # CWSS
    for indexs in range(input_loss.size(0)):
        for i in range(10):
            if pred_target[indexs] == i:
                classes_loss[i].append(input_loss[indexs].cpu().numpy())
                class_index.append(indexs)
                loss_index[i].append(indexs)
          
    for i in range(args.num_class):
        gmm = GaussianMixture(n_components=2,max_iter=50,tol=1e-3,reg_covar=5e-4)
        gmm.fit(np.array(classes_loss[i]).reshape(-1, 1))
        Partial_prob = gmm.predict_proba(input_loss) 
        prob.append(Partial_prob[:,gmm.means_.argmin()])
    prob = np.concatenate(prob)
    prob = prob[class_index]

    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = FeaModel(args.num_class)
    model = model.cuda()
    return model

if __name__ == '__main__':
    stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
    test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')  
    pre_data = []
    pre_data1 = []
    duilie = []
    warm_up = args.num_warmup
    loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
        root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))
    
    print('| Building net')
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True
    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler1 = CosineAnnealingLR(optimizer1, args.num_epochs, eta_min=1e-3)
    scheduler2 = CosineAnnealingLR(optimizer2, args.num_epochs, eta_min=1e-3)
    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode=='asym':
        conf_penalty = NegEntropy()

    all_loss = [[],[]] # save the history of losses from two networks
    flag=args.flag
    for epoch in range(args.num_epochs+1):   
        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train')   
        
        if epoch < warm_up:       
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader, flag=True)    
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader, flag=True) 
    
        else:
            prob1,all_loss[0]=eval_train(net1,all_loss[0],flag=True)   
            prob2,all_loss[1]=eval_train(net2,all_loss[1],flag=True)          
            pred1 = (prob1 > args.p_threshold)      
            pred2 = (prob2 > args.p_threshold)  
            args.p_threshold = args.p_threshold

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
            train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader,scheduler1,pre_data,pre_data1,duilie,flag) # train net1  
            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
            train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader,scheduler2,pre_data,pre_data1,duilie,flag) # train net2 

        test(epoch,net1, net2)  