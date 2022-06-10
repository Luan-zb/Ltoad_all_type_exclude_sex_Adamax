import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.nn.functional as F

import numpy as np
import pdb
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.
		从给定的索引列表中按顺序采样元素，无需替换
	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)
'''
>>> import torch
>>> A=torch.ones(2,3) #2x3的张量（矩阵）                                     
>>> A
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
>>> B=2*torch.ones(4,3)#4x3的张量（矩阵）                                    
>>> B
tensor([[ 2.,  2.,  2.],
        [ 2.,  2.,  2.],
        [ 2.,  2.,  2.],
        [ 2.,  2.,  2.]])
>>> C=torch.cat((A,B),0)#按维数0（行）拼接
>>> C
tensor([[ 1.,  1.,  1.],
         [ 1.,  1.,  1.],
         [ 2.,  2.,  2.],
         [ 2.,  2.,  2.],
         [ 2.,  2.,  2.],
         [ 2.,  2.,  2.]])
'''
def collate_MIL_mtl_concat(batch):
	'''

	img--->number(patches number)*1024
	在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[idx]取值。当实例对象做P[idx]运算时，就会调用类中的__getitem__()方法
	'''
	img = torch.cat([item[0] for item in batch], dim = 0)   #按行进行concat操作
	label = torch.LongTensor([item[1] for item in batch])
	#site = torch.LongTensor([item[2] for item in batch])
	#sex = torch.LongTensor([item[3] for item in batch])
	return [img, label]

'''
pytorch 的数据加载到模型的操作顺序是这样的：
① 创建一个 Dataset 对象
② 创建一个 DataLoader 对象
③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练

dataset = MyDataset()
dataloader = DataLoader(dataset)
num_epoches = 100
for epoch in range(num_epoches):
    for img, label in dataloader:
        ....
'''
def get_simple_loader(dataset, batch_size=1):
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	'''
	dataset:dataset from which to load the data.
	batch_size:how many samples per batch to load (default: 1).
	sampler:defines the strategy to draw samples from the dataset. If specified, ``shuffle`` must be False.
            自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
	collate_fn:collate_fn (callable, optional): merges a list of samples to form a mini-batch.
			将一个list的sample组成一个mini-batch的函数-->[img, label, site, sex]
	'''
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL_mtl_concat, **kwargs)
	return loader 

'''
get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
'''
def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader
		shuffle为默认值 False时，sampler是SequentialSampler，就是按顺序取样,
		shuffle为True时，sampler是RandomSampler， 就是按随机取样

		https://www.cnblogs.com/marsggbo/p/11541054.html
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL_mtl_concat, **kwargs)	
			else:  #training   default:`len(dataset)`
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL_mtl_concat, **kwargs)
		else:  #validation
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL_mtl_concat, **kwargs)
	
	else: #testing
		'''
		numpy.random.choice(a, size=None, replace=True, p=None)
		从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
		replace:True表示可以取相同数字，False表示不可以取相同数字
		数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
		'''
		ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset)*0.01), replace = False)   #直选了1%作为测试集
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL_mtl_concat, **kwargs )
		'''
		def __len__(self):
			return len(self.slide_data)
		'''
	return loader

#ref:https://blog.csdn.net/qq_36401512/article/details/105045157
def get_optim(model, args):
	#frozen the parameters needs lambda and filter function
	'''
	class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	'''
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg) # 记住一定要加上filter()，不然会报错
		'''
		参数p赋值的元素从列表model.parameters()里取。所以只取param.requires_grad = True（模型参数的可导性是true的元素），就过滤掉为false的元素。
		'''
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	elif args.opt == 'adamax':
		optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()  #统计参数量
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)

	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			remaining_ids = possible_indices

			if val_num[c] > 0:
				val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
				remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
				all_val_ids.extend(val_ids)

			if custom_test_ids is None and test_num[c] > 0: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

'''
import torch
outputs=torch.FloatTensor([[1,2,3]])
targets=torch.FloatTensor([[0,2,3]])
print(targets.float().eq(outputs.float()).float().mean())
tensor(0.6667)

'''
def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
