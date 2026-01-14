from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import cv2
from scipy.io import loadmat           
def unpickle(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, transform1, mode, noise_file='', pred=[], probability=[], log='',idx=[]): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.transform1 = transform1
        self.mode = mode  
        self.transition_eoc2 = {0:2,2:1,3:0,1:3} # eoc2 class transition for asymmetric noise
        result = root_dir.split('/')
        last_path = result[-1]
        if self.mode=='test':
            if dataset=='MSTAR':
                self.test_data1 = []
                self.test_data2 = []
                self.test_label = []
                self.flg = 0
                dpath = '%s/mstar_orignal_jpg/test'%(root_dir) 
                dpath1 = '%s/mstar_scatter/test'%(root_dir)             
                per_datas = os.listdir(dpath)
                for per_data in per_datas:    
                    per_data1 = os.path.join(dpath, per_data)
                    per_data2 = os.path.join(dpath1, per_data)
                    per_classes1 = os.listdir(per_data1)
                    for per_class1 in per_classes1:
                        per_class_paths1 = os.path.join(per_data1, per_class1)
                        per_class_paths2 = os.path.join(per_data2, per_class1)
                   
                        per_class_paths2 =per_class_paths2.replace('.JPG', '.mat')
                        keypoints = loadmat(per_class_paths2)['para_estimated']
                        images = cv2.imread(per_class_paths1)
                        self.test_data1.append(images)
                        self.test_data2.append(keypoints)
                        self.test_label.append(self.flg)
                    self.flg = self.flg + 1
                self.test_data2 = torch.tensor(self.test_data2)
                self.test_data2 = self.test_data2.permute(0, 2, 1)
                        
        else:    
            train_data1=[]
            train_data2=[]
            train_label=[]
            self.flg = 0
            if dataset=='MSTAR':
                dpath = '%s/mstar_orignal_jpg/train'%(root_dir)
                dpath1 = '%s/mstar_scatter/train'%(root_dir)
                per_datas = os.listdir(dpath)
                for per_data in per_datas:
                    per_data1 = os.path.join(dpath, per_data)
                    per_data2 = os.path.join(dpath1, per_data)
                    per_classes = os.listdir(per_data1)
                    for per_class in per_classes:
                        per_class_paths1 = os.path.join(per_data1, per_class)
                        per_class_paths2 = os.path.join(per_data2, per_class)
                        images = cv2.imread(per_class_paths1)
                        per_class_paths2 =per_class_paths2.replace('.JPG', '.mat')
                        keypoints = loadmat(per_class_paths2)['para_estimated']
                        train_data1.append(images)
                        train_label.append(self.flg)
                        train_data2.append(keypoints)
                    self.flg = self.flg + 1
            example_id=[]
            if os.path.exists(noise_file):
                data = json.load(open(noise_file,"r"))
                noise_label = data["data1"]
                example_id = data["data2"]
            else:    #inject noise   
                noise_label = []
                idx = list(range(len(train_label)))
                random.shuffle(idx)
                num_noise = int(self.r*len(train_label))            
                noise_idx = idx[:num_noise]
                for i in range(len(train_label)):
                    if i in noise_idx:
                        example_id.append(0)
                        if noise_mode=='sym':
                            noiselabel = random.randint(0,self.flg-1)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':  
                            if last_path == "soc" :
                                noiselabel = self.transition[train_label[i]]
                            else:
                                noiselabel = self.transition_eoc2[train_label[i]]
                            noise_label.append(noiselabel)                    
                    else:    
                        noise_label.append(train_label[i])   
                        example_id.append(1)
                data = {"data1":noise_label,"data2":example_id}
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(data,open(noise_file,"w"))       
                
            if self.mode == 'all':
                self.noise_id = idx
                self.train_data1 = train_data1
                self.train_data2 = train_data2
                self.noise_label = noise_label
                self.example_id = example_id
                self.train_data2 = torch.tensor(self.train_data2)
                self.train_data2 = self.train_data2.permute(0, 2, 1)
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    clean = (np.array(noise_label)==np.array(train_label))                                                 
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                train_data2 = torch.tensor(train_data2).permute(0, 2, 1)
                
                self.train_data1 = [train_data1[i] for i in pred_idx] 
                self.example_id = [example_id[i] for i in pred_idx]   
                self.train_data2 = train_data2[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]  
                self.train_label = [train_label[i] for i in pred_idx]                         
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        
        if self.mode=='labeled':
            img, keypoints, target, true_target, prob, id= self.train_data1[index], self.train_data2[index], self.noise_label[index], self.train_label[index], self.probability[index], self.example_id[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img)
            keypoints1 = keypoints.float()
            keypoints2 = keypoints.float()
            return img1, img2, keypoints1, keypoints2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data1[index]
            keypoints = self.train_data2[index]
            id = self.example_id[index]
            target = self.train_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img)
            img3 = self.transform1(img)
            img4 = self.transform1(img)
            img5 = self.transform1(img)
            img6 = self.transform1(img)
            keypoints1 = keypoints.float()
            keypoints2 = keypoints.float() 
            keypoints3 = keypoints.float() 
            keypoints4 = keypoints.float() 
            return img1, img2, img3, img4, img5, img6, keypoints1, keypoints2, keypoints3, keypoints4
        elif self.mode=='all':
            img, keypoints, target, noise_id = self.train_data1[index], self.train_data2[index], self.noise_label[index], self.example_id[index]
            img = Image.fromarray(img)
            keypoints = keypoints.float()
            img = self.transform(img)            
            return img, keypoints, target, index , noise_id       
        elif self.mode=='test':
            img, keypoints, target = self.test_data1[index], self.test_data2[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)  
            keypoints = keypoints.float()          
            return img, keypoints, target
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data1)
        else:
            return len(self.test_data1) 

class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.idx = list(range(2746))
        random.shuffle(self.idx)

        if self.dataset=='MSTAR':
            self.transform_train = transforms.Compose([
                    transforms.CenterCrop(64),
                    transforms.ToTensor()
                ]) 
            self.transform_train_l = transforms.Compose([
                    transforms.CenterCrop(64),
                    transforms.ToTensor()
                ])
            self.transform_train_2 = transforms.Compose([
                    transforms.CenterCrop(96),
                    transforms.RandomCrop(64),
                    transforms.ColorJitter(contrast=0.3, saturation=0.3),
                    transforms.ToTensor()
                ])
            self.transform_test = transforms.Compose([
                    transforms.CenterCrop(64),
                    transforms.ToTensor()
                ])    
        
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, transform1=self.transform_train_2, mode="all",noise_file=self.noise_file,idx=self.idx)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, transform1=self.transform_train_2, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, transform1=self.transform_train_2, mode="unlabeled", noise_file=self.noise_file, pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, transform1=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, transform1=self.transform_train_2, mode='all', noise_file=self.noise_file,idx=self.idx)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False)          
            return eval_loader        