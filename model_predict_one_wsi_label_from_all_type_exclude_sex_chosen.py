
from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_mtl_concat import Generic_MIL_MTL_Dataset, save_splits
import h5py
from utils.eval_utils_mtl_concat import *

# Training settings
parser = argparse.ArgumentParser(description='TOAD Evaluation Script')
parser.add_argument('--data_root_dir', type=str, help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. ' +
                         'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--drop_out', action='store_true', default=False,
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=1, help='number of folds (default: 1)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False,
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'],
                    default='test')  # 定义了所加载的数据，默认情况下为test
parser.add_argument('--task', type=str, choices=['study_v2_mtl_sex'])

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoding_size = 1024
# save_exp_code：dummy_mtl_sex_s1_eval
# models_exp_code：dummy_mtl_sex_s1
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))  # save the result   eval_results/EVAL_dummy_mtl_sex_s1_eval
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))  # load the model          results/dummy_mtl_sex_s1

os.makedirs(args.save_dir, exist_ok=True)
# 若split_dir 为None
if args.splits_dir is None:
    args.splits_dir = args.models_dir  # /results/dummy_mtl_sex_s1

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,  #
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'drop_out': args.drop_out,
            'micro_avg': args.micro_average}

# {'task': 'study_v2_mtl_sex', 'split': 'test', 'save_dir': './eval_results/EVAL_dummy_mtl_sex_s1_eval', 'models_dir': 'results/dummy_mtl_sex_s1', 'drop_out': True, 'micro_avg': False}
with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code),
          'w') as f:  # ./eval_results/EVAL_dummy_mtl_sex_s1_eval/eval_experiment_dummy_mtl_sex_s1_eval.txt
    print(settings, file=f)
f.close()

print(settings)
'''
    pt file:  features, label, site, sex
    h5 file:  features, label, site, sex, coords
 '''
if args.task == 'study_v2_mtl_sex':
    args.n_classes = 8
    dataset = Generic_MIL_MTL_Dataset(csv_path='dataset_csv/one_wsi_exclude_sex_chosen.csv',
                                      data_dir=os.path.join(args.data_root_dir,'/home/daichuangchuang/Nature/Lclam/FEATURES_DIRECTORY_one_wsi_chosen/pt_files'),
                                      shuffle=False,
                                      print_info=True,
                                      label_dicts=[{'Lung': 0, 'Skin': 1, 'Kidney': 2, 'Uterus Endometrium':3, 'Pancreas':4, 'Soft Tissue':5, 'Head Neck':6, 'Brain':7}],
                                      label_cols=['label'],
                                      patient_strat=False)
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)  # start=0, end=1, folds=0
else:
    folds = range(args.fold, args.fold + 1)
ckpt_paths = os.path.join(args.models_dir, 's_0_checkpoint.pt')  # results/dummy_mtl_sex_s1/s_0_checkpoint.pt


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

if __name__ == "__main__":
        model = initiate_model(args, ckpt_paths)  # create the model
        loader = get_simple_loader(dataset)  # [img, label, site, sex]   batchsize default：1
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        slide_ids = loader.dataset.slide_data['slide_id']  # 所有的slide_id
        #print("slide_id","\t","sex","\t","pred_0","\t","p_0","\t","pred_1","\t","p_1")
        print("slide_id","pred_0","p_0","pred_1","p_1","pred_2","p_2","pred_3","p_3","pred_4","p_4","pred_5","p_5","pred_6","p_6","pred_7","p_7",sep="\t")
        for batch_idx, (data, label) in enumerate(loader):  # batch size=1   len(data_loader)=len(dataset)
            data = data.to(device)
            label = label.to(device)
            #site = site.to(device)
            #sex = sex.float().to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():  # require_grad=false-->一是减少内存，二是可以把这个operator从computation graph中detach出来，这样就不会在BP过程中计算到。
                model.eval()
                model_results_dict = model(data)
                '''
                logits-->   logits  = self.classifier(M[0].unsqueeze(0))   1×2  属于每一类别的概率
                Y_hat-->    Y_hat = torch.topk(logits, 1, dim = 1)[1]  #dim=0表示按照列求topn，dim=1表示按照行求topn  得到的是预测元素下标
                Y_prob-->   Y_prob = F.softmax(logits, dim = 1)        #按行softmax，行和为1     1×2  属于每一类别的概率
                '''
            logits, Y_prob, Y_hat = model_results_dict['logits'], model_results_dict['Y_prob'], model_results_dict['Y_hat']
            #print(logits,Y_prob,Y_hat)
            print(slide_id,"Lung",Y_prob.cpu().numpy()[0][0],"Skin",Y_prob.cpu().numpy()[0][1],"Kidney",Y_prob.cpu().numpy()[0][2],"Uterus Endometrium",Y_prob.cpu().numpy()[0][3],"Pancreas",Y_prob.cpu().numpy()[0][4],"Soft Tissue",Y_prob.cpu().numpy()[0][5],"Head Neck",Y_prob.cpu().numpy()[0][6],"Brain",Y_prob.cpu().numpy()[0][7],sep="\t")
