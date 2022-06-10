import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_toad import TOAD_fc_mtl_concat
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils_mtl_concat import EarlyStopping,  Accuracy_Logger
from utils.file_utils import save_pkl, load_pkl
from sklearn.metrics import roc_auc_score, roc_curve, auc
import h5py
from models.resnet_custom import resnet50_baseline
import math
from sklearn.preprocessing import label_binarize

#define the model
def initiate_model(args, ckpt_path=None):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    model = TOAD_fc_mtl_concat(**model_dict)    

    model.relocate()
    print_network(model)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)  #create the model

    print('Init Loaders')
    loader = get_simple_loader(dataset)   #[img, label, site, sex]   batchsize default：1
    results_dict = summary(model, loader, args)   #    inference_results = {'patient_results': patient_results, 'cls_test_error': cls_test_error, 'cls_auc': cls_auc, 'cls_aucs': cls_aucs, 'loggers': (cls_logger), 'df':df}

    print('cls_test_error: ', results_dict['cls_test_error'])
    print('cls_auc: ', results_dict['cls_auc'])

    return model, results_dict

# Code taken from pytorch/examples for evaluating topk classification on on ImageNet
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def summary(model, loader, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_logger = Accuracy_Logger(n_classes=args.n_classes)  #acc, correct, count
    #ite_logger = Accuracy_Logger(n_classes=2)
    model.eval()
    cls_test_error = 0.
    cls_test_loss = 0.
    site_test_error = 0.
    site_test_loss = 0.

    '''
    all_cls_probs->  [batch size,n_classes]   会输出属于每一类别的概率
    all_cls_labels-> [[batch size]            true label only one
    自定义的数据集类MyDataset对象data的长度和 DataLoader对象data_loader的长度，我们会发现：data_loader的长度是data的长度除以batch_size。
    ref：https://blog.csdn.net/weixin_45901519/article/details/115672355
    '''
    all_cls_probs = np.zeros((len(loader), args.n_classes))  #loader-->[img, label, site, sex]  len(loader)->batch size；
    all_cls_labels = np.zeros(len(loader))
    all_sexes = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id'] #所有的slide_id
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):  #batch size=1   len(data_loader)=len(dataset)
        data =  data.to(device)
        label = label.to(device)
        #site = site.to(device)
        #sex = sex.float().to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad(): #require_grad=false-->一是减少内存，二是可以把这个operator从computation graph中detach出来，这样就不会在BP过程中计算到。
            model_results_dict = model(data)  #组织切片提取的特征向量+sex

        '''
        logits-->   logits  = self.classifier(M[0].unsqueeze(0))   1×2  属于每一类别的概率
        Y_hat-->    Y_hat = torch.topk(logits, 1, dim = 1)[1]  #dim=0表示按照列求topn，dim=1表示按照行求topn  得到的是预测元素下标
        Y_prob-->   Y_prob = F.softmax(logits, dim = 1)        #按行softmax，行和为1     1×2  属于每一类别的概率
        '''
        logits, Y_prob, Y_hat  = model_results_dict['logits'], model_results_dict['Y_prob'], model_results_dict['Y_hat']
        del model_results_dict
        '''
        def log(self, Y_hat, Y):   
            Y_hat = int(Y_hat)
            Y = int(Y)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat == Y)
        '''
        cls_logger.log(Y_hat, label)      #统计总类别数和预测正确数目
        cls_probs = Y_prob.cpu().numpy()  #预测的属于每一类别的概率
        all_cls_probs[batch_idx] = cls_probs
        all_cls_labels[batch_idx] = label.item()

        #all_sexes[batch_idx] = sex.item()

        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'cls_prob': cls_probs, 'cls_label': label.item()}})
        '''
        def calculate_error(Y_hat, Y):
	        error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	        return error
        '''
        cls_error = calculate_error(Y_hat, label)
        cls_test_error += cls_error

    cls_test_error /= len(loader)
    '''
    >>a = np.array([[1, 5, 5, 2],
               [9, 6, 2, 8],
               [3, 7, 9, 1]])
    >>np.argmax(a,axis=0)
    >>array([1, 2, 2, 1], dtype=int64)
    >>np.argmax(a,axis=1)
    >>array([1, 0, 2], dtype=int64)
    '''
    all_cls_preds = np.argmax(all_cls_probs, axis=1)
    topk=()
    if args.n_classes > 2:
        if args.n_classes > 5:
            topk = (1,3,5)
        else:
            topk = (1,3)
        topk_accs = accuracy(torch.from_numpy(all_cls_probs), torch.from_numpy(all_cls_labels), topk=topk)
        for k in range(len(topk)):
            print('top{} acc: {:.3f}'.format(topk[k], topk_accs[k].item()))

    if len(np.unique(all_cls_labels)) == 1:  # 只有一个类别时 无法计算auc
        cls_auc = -1
        cls_aucs = []
    else:
        if args.n_classes == 2:
            '''
            y_true = np.array([0, 0, 1, 1])
            y_scores = np.array([0.1, 0.4, 0.35, 0.8])
            roc_auc_score(y_true, y_scores)
            0.75
            '''
            cls_auc = roc_auc_score(all_cls_labels, all_cls_probs[:, 1])
            cls_aucs = []
        else:
            cls_aucs = []
            binary_labels = label_binarize(all_cls_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_cls_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_cls_probs[:, class_idx])
                    cls_aucs.append(auc(fpr, tpr))
                else:
                    cls_aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_cls_labels, classes=[i for i in range(args.n_classes)])
                valid_classes = np.where(np.any(binary_labels, axis=0))[0]
                binary_labels = binary_labels[:, valid_classes]
                valid_cls_probs = all_cls_probs[:, valid_classes]
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), valid_cls_probs.ravel())
                cls_auc = auc(fpr, tpr)
            else:
                cls_auc = np.nanmean(np.array(cls_aucs))
    
    '''
        cls_probs = Y_prob.cpu().numpy()  #预测的属于每一类别的概率
        all_cls_preds = np.argmax(all_cls_probs, axis=1)
        all_cls_labels[batch_idx] = label.item()
        
    slide_id,sex,Y,Y_hat,p_0,p_1
    C3L-02647-24,1.0,1.0,1,0.010934877209365368,0.9890650510787964
    C3L-04733-22,0.0,1.0,1,2.2340067516779527e-05,0.999977707862854

    '''
    results_dict = {'slide_id': slide_ids,  'Y': all_cls_labels, 'Y_hat': all_cls_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_cls_probs[:,c]})


    df = pd.DataFrame(results_dict)
    '''
    patient_results--->patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'cls_prob': cls_probs, 'cls_label': label.item()}})
    cls_test_error---->cls_error = calculate_error(Y_hat, label)
                       cls_test_error += cls_error
    cls_auc:  ---->    cls_auc = roc_auc_score(all_cls_labels, all_cls_probs[:, 1])
    cls_aucs: ---->    cls_aucs = []
    cla_logger: ---->  acc, correct, count

    '''
    inference_results = {'patient_results': patient_results, 'cls_test_error': cls_test_error,
                     'cls_auc': cls_auc, 'cls_aucs': cls_aucs, 'loggers': (cls_logger), 'df':df}

    for k in range(len(topk)):
        inference_results.update({'top{}_acc'.format(topk[k]): topk_accs[k].item()})

    return inference_results
