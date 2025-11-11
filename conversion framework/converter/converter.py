import torch
from torch import fx
from torch import nn
from tqdm import tqdm
from typing import Tuple
import numpy as np
import os
from neurons import *
from utils import MyatSequential,MyConv,MyLinear,MyPass
from collections.abc import Iterable
class Converter(nn.Module):
    def __init__(self, neuron, args, T=0, step_mode='s', fuse_flag=False):
        super().__init__()
        self.neuron = neuron
        self.neuron_identity=args.neuron_name_identity
        self.fuse_flag = fuse_flag
        self.step_mode = step_mode
        self.T=T
        self.args=args
        self.identity_change=(args.identity_change.lower()=='true')

    def forward(self, ann: nn.Module):
        snn = Converter.replace_relu_by_neuron(ann,self.neuron,step_mode=self.step_mode,T=self.T,args=self.args)
        if self.identity_change:
            snn = Converter.replace_identity_by_neuron(snn,self.neuron_identity,step_mode=self.step_mode,T=self.T,args=self.args)
        if self.fuse_flag:
            snn = fx.symbolic_trace(snn)
            snn_fused = self.fuse(snn, fuse_flag=self.fuse_flag)
            return snn_fused
        else:
            return snn
    
    
    @staticmethod
    def replace_by_maxpool_neuron(model,neuron=maxpool_neuron,T=8,step_mode='s',coding_type='diff_rate'):
        # here the maxpool neuron is made originally
        for name, module in model._modules.items():
            if "maxpool" in module.__class__.__name__.lower():
                model._modules[name] = neuron(maxpool=module,T=T,step_mode=step_mode,coding_type=coding_type)
            elif hasattr(module, "_modules"):
                # 递归替换
                model._modules[name] = Converter.replace_by_maxpool_neuron(module,neuron,T,step_mode=step_mode,coding_type=coding_type)
        return model 
    
    @staticmethod
    def change_maxpool_before_relu(model):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = Converter.change_maxpool_before_relu(module)
            if 'relu' in module.__class__.__name__.lower() or ("threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu'):
                tmp_name = name
            if 'maxpool' in module.__class__.__name__.lower():
                tmp = model._modules[tmp_name]
                model._modules[tmp_name] = nn.Identity()
                model._modules[name] = nn.Sequential(model._modules[name],tmp)
                print("change a maxpool before relu")
        return model
    
    @staticmethod
    def replace_relu_by_neuron(model, neuron, args, step_mode='s', T=0):
        # 实为replace_relu_by_neuron
        if neuron=='IF':
            return Converter.replace_relu_by_IF(model, step_mode=step_mode, T=T, neuron = IF,args=args)
        elif neuron=='IF_with_neg':
            return Converter.replace_relu_by_IF(model, step_mode=step_mode, T=T, neuron = IF_with_neg,args=args)
        elif neuron == 'IF_diff':
            return Converter.replace_relu_by_IF(model, step_mode=step_mode, T=T, neuron = IF_diff,args=args)
        elif neuron == 'IF_with_neg_line':
            return Converter.replace_relu_by_IF(model, step_mode=step_mode, T=T, neuron = IF_with_neg_line,args=args)
        elif neuron == 'IF_diff_line':
            return Converter.replace_relu_by_IF(model, step_mode=step_mode, T=T, neuron = IF_diff_line,args=args)
        elif neuron=='LIF':
            return Converter.replace_relu_by_LIF(model, step_mode=step_mode, T=T, tau=args.tau, neuron=LIF,args=args)
        elif neuron=='LIF_with_neg':
            return Converter.replace_relu_by_LIF(model, step_mode=step_mode, T=T, tau=args.tau, neuron=LIF_with_neg,args=args)
        elif neuron=='LIF_diff':
            return Converter.replace_relu_by_LIF(model, step_mode=step_mode, T=T, tau=args.tau, neuron=LIF_diff,args=args)
        elif neuron=='MTH':
            return Converter.replace_relu_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='MTH_with_neg':
            return Converter.replace_relu_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH_with_neg, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='MTH_diff':
            return Converter.replace_relu_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH_diff, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='MTH_with_neg_line':
            return Converter.replace_relu_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH_with_neg_line, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='MTH_diff_line':
            return Converter.replace_relu_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH_diff_line, num_thresholds = args.num_thresholds,args=args)
        
        elif neuron=='H_MT_version_1':
            return Converter.replace_relu_by_HMT(model, step_mode=step_mode, T=T, neuron=H_MT_version_1, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='H_MT_version_2':
            return Converter.replace_relu_by_HMT(model, step_mode=step_mode, T=T, neuron=H_MT_version_2, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='H_MT_version_IF':
            return Converter.replace_relu_by_HMT(model, step_mode=step_mode, T=T, neuron=H_MT_version_IF, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='pos_H_MT_version_1':
            return Converter.replace_relu_by_HMT(model, step_mode=step_mode, T=T, neuron=pos_H_MT_version_1, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='pos_H_MT_version_2':
            return Converter.replace_relu_by_HMT(model, step_mode=step_mode, T=T, neuron=pos_H_MT_version_2, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='pos_H_MT_storage_version_1':
            return Converter.replace_relu_by_HMT(model, step_mode=step_mode, T=T, neuron=pos_H_MT_storage_version_1, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='H_MT_version_1_no_commu':
            return Converter.replace_relu_by_HMT(model, step_mode=step_mode, T=T, neuron=H_MT_version_1_no_commu, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='pos_H_MT_version_IF':
            return Converter.replace_relu_by_HMT(model, step_mode=step_mode, T=T, neuron=pos_H_MT_version_IF, num_thresholds = args.num_thresholds,args=args)
        else:
            print("Unsupported Neuron Name")
    
    def replace_identity_by_neuron(model, neuron, args, step_mode='s', T=0):
        print(neuron)
        if neuron=='IF':
            return Converter.replace_identity_by_IF(model, step_mode=step_mode, T=T, neuron = IF,args=args)
        elif neuron=='IF_with_neg':
            return Converter.replace_identity_by_IF(model, step_mode=step_mode, T=T, neuron = IF_with_neg,args=args)
        elif neuron == 'IF_diff':
            return Converter.replace_identity_by_IF(model, step_mode=step_mode, T=T, neuron = IF_diff,args=args)
        elif neuron == 'IF_with_neg_line':
            return Converter.replace_identity_by_IF(model, step_mode=step_mode, T=T, neuron = IF_with_neg_line,args=args)
        elif neuron == 'IF_diff_line':
            return Converter.replace_identity_by_IF(model, step_mode=step_mode, T=T, neuron = IF_diff_line,args=args)
        elif neuron=='LIF':
            return Converter.replace_identity_by_LIF(model, step_mode=step_mode, T=T, tau=args.tau, neuron=LIF,args=args)
        elif neuron=='LIF_with_neg':
            return Converter.replace_identity_by_LIF(model, step_mode=step_mode, T=T, tau=args.tau, neuron=LIF_with_neg,args=args)
        elif neuron=='LIF_diff':
            return Converter.replace_identity_by_LIF(model, step_mode=step_mode, T=T, tau=args.tau, neuron=LIF_diff,args=args)
        elif neuron=='MTH':
            return Converter.replace_identity_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='MTH_with_neg':
            return Converter.replace_identity_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH_with_neg, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='MTH_diff':
            return Converter.replace_identity_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH_diff, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='MTH_with_neg_line':
            return Converter.replace_identity_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH_with_neg_line, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='MTH_diff_line':
            return Converter.replace_identity_by_MTH(model, step_mode=step_mode, T=T, neuron=MTH_diff_line, num_thresholds = args.num_thresholds,args=args)
        
        elif neuron=='H_MT_version_1':
            return Converter.replace_identity_by_HMT(model, step_mode=step_mode, T=T, neuron=H_MT_version_1, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='H_MT_version_2':
            return Converter.replace_identity_by_HMT(model, step_mode=step_mode, T=T, neuron=H_MT_version_2, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='H_MT_version_IF':
            return Converter.replace_identity_by_HMT(model, step_mode=step_mode, T=T, neuron=H_MT_version_IF, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='pos_H_MT_storage_version_1':
            return Converter.replace_identity_by_HMT(model, step_mode=step_mode, T=T, neuron=pos_H_MT_storage_version_1, num_thresholds = args.num_thresholds,args=args)
        elif neuron=='H_MT_version_1_no_commu':
            return Converter.replace_identity_by_HMT(model, step_mode=step_mode, T=T, neuron=H_MT_version_1_no_commu, num_thresholds = args.num_thresholds,args=args)
        else:
            print("Unsupported Neuron Name")

    @staticmethod
    def replace_relu_by_IF(model,step_mode,T,neuron,args):
        for name, module in model._modules.items():
            if module.__class__.__name__.lower()=="relu":
                # relu
                model._modules[name] = neuron(T=T, thresh=1.0, step_mode=step_mode)
            elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu':
                # hook
                thre = model._modules[name].scale
                thre = thre*(thre>=0).float()
                model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode)
            elif hasattr(module, "_modules"):
                model._modules[name] = Converter.replace_relu_by_IF(module,step_mode,T,neuron,args=args)
        return model 
    
    def replace_identity_by_IF(model,step_mode,T,neuron,args):
        for name, module in model._modules.items():
            if module.__class__.__name__.lower()=="identity":
                # identity
                model._modules[name] = neuron(T=T, thresh=1.0, step_mode=step_mode)
            elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='identity':
                # hook
                thre = model._modules[name].scale
                thre = thre*(thre>=0).float()
                model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode)
            elif hasattr(module, "_modules"):
                model._modules[name] = Converter.replace_identity_by_IF(module,step_mode,T,neuron,args=args)
        return model 
    
    @staticmethod
    def replace_relu_by_LIF(model,step_mode,T,tau,neuron,args):
        for name, module in model._modules.items():
            if module.__class__.__name__.lower()=="relu":
                model._modules[name] = neuron(T=T, thresh=1.0, tau = tau, step_mode=step_mode)
            elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu':
                thre = model._modules[name].scale
                thre = thre*(thre>=0).float()
                model._modules[name] = neuron(T=T, thresh=thre, tau = tau, step_mode=step_mode)
            elif hasattr(module, "_modules"):
                model._modules[name] = Converter.replace_relu_by_LIF(module,step_mode,T,tau,neuron,args=args)
        return model
    
    def replace_identity_by_LIF(model,step_mode,T,tau,neuron,args):
        for name, module in model._modules.items():
            if module.__class__.__name__.lower()=="identity":
                model._modules[name] = neuron(T=T, thresh=1.0, tau = tau, step_mode=step_mode)
            elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='identity':
                thre = model._modules[name].scale
                thre = thre*(thre>=0).float()
                model._modules[name] = neuron(T=T, thresh=thre, tau = tau, step_mode=step_mode)
            elif hasattr(module, "_modules"):
                model._modules[name] = Converter.replace_identity_by_LIF(module,step_mode,T,tau,neuron,args=args)
        return model
    
    @staticmethod
    def replace_relu_by_MTH(model,step_mode,T,neuron,num_thresholds,args):
        for name, module in model._modules.items():
            print("name ahead is",module.__class__.__name__.lower())
            if module.__class__.__name__.lower()=="relu":# no use
                model._modules[name] = neuron(T=T, thresh=1.0, step_mode=step_mode,num_thresholds=num_thresholds)
            elif args.model_name in ['fcos_resnet50_fpn', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2']:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear','conv2d']:
                    model._modules[name] = module.out
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu':
                    thre = model._modules[name].scale
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds)
                elif "threhook" in module.__class__.__name__.lower():
                    model._modules[name] = exp_comp_neuron(module.out,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_relu_by_MTH(module,step_mode,T,neuron,num_thresholds,args)
            elif args.model_name in ['vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu':
                    print("module.__class__.__name__.lower() is",module.__class__.__name__.lower())
                    print("name is",name)
                    thre = model._modules[name].scale
                    #print(thre)
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds)
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear','conv2d']:
                    model._modules[name] = module.out
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_relu_by_MTH(module,step_mode,T,neuron,num_thresholds,args)
            else:
                if "threhook" in module.__class__.__name__.lower(): #and (module.out.__class__.__name__.lower()=='relu' or module.out.__class__.__name__.lower()=='myat'):
                #if "threhook" in module.__class__.__name__.lower() and (module.out.__class__.__name__.lower()=='relu' or module.out.__class__.__name__.lower()=='myat'):    
                    print(f"threhook's out is {module.out.__class__.__name__.lower()}")
                    scale = 8
                    thre = model._modules[name].scale
                    # print(thre,module.out.__class__.__name__.lower())
                    # thre = thre*(thre>=0).float()*scale
                    if module.out.__class__.__name__.lower()=='relu':
                        thre = thre*(thre>=0).float()
                        model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds)
                    elif module.out.__class__.__name__.lower()=='myat':
                        thre = thre*(thre>=0).float()
                        thre2 = model._modules[name].scale2
                        thre2 = thre2*(thre2>=0).float()#*scale
                        model._modules[name] = MyatSequential(neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds),
                                                            neuron(T=T, thresh=thre2, step_mode=step_mode,num_thresholds=num_thresholds),
                                                            AtNeuron(T=args.time,step_mode=args.step_mode,coding_type=args.coding_type))
                    elif module.out.__class__.__name__.lower() in ['linear','conv2d']:
                        model._modules[name] = nn.Sequential(neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds),
                                                             module.out)
                    # elif module.out.__class__.__name__.lower() in ['conv2d']:
                    #     model._modules[name] = module.out
                    else:
                        if module.out.__class__.__name__.lower() not in ['identity']:
                            # may not change the identity module, 
                            # though exp_comp_neuron-version identity is identical to identity, but brings extra cost
                            # however, such extra cost is also small
                            model._modules[name] = exp_comp_neuron(module.out,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
                elif module.__class__.__name__.lower() == 'mymul':
                    model._modules[name] = MulNeuron(T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_relu_by_MTH(module,step_mode,T,neuron,num_thresholds,args)
        return model
    
    def replace_identity_by_MTH(model,step_mode,T,neuron,num_thresholds,args):
        for name, module in model._modules.items():
            print("name ahead is",module.__class__.__name__.lower())
            if module.__class__.__name__.lower()=="identity":# useless
                #print("oh yes I do converte an Identity!")
                model._modules[name] = neuron(T=T, thresh=1.0, step_mode=step_mode,num_thresholds=num_thresholds)
            elif args.model_name in ['fcos_resnet50_fpn', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2']:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear','conv2d']:
                    model._modules[name] = module.out
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='identity':
                    thre = model._modules[name].scale
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds)
                elif "threhook" in module.__class__.__name__.lower():
                    model._modules[name] = exp_comp_neuron(module.out,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_identity_by_MTH(module,step_mode,T,neuron,num_thresholds,args)
            elif args.model_name in ['vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='identity':
                    #print("oh yes I do converte an Identity!")
                    thre = model._modules[name].scale
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds)
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear','conv2d']:
                    #print("oh yes I do converte an Identity!")
                    model._modules[name] = module.out
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_identity_by_MTH(module,step_mode,T,neuron,num_thresholds,args)
            else:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='identity':
                    #print("oh yes I do converte an Identity!")
                    thre = model._modules[name].scale
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds)
                # as for ViT, only use linear neuron to transform, so many things are repeated in replace_relu & replace_identity
                if hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_identity_by_MTH(module,step_mode,T,neuron,num_thresholds,args)
        return model
    
    def replace_relu_by_HMT(model,step_mode,T,neuron,num_thresholds,args):
        for name, module in model._modules.items():
            print("name ahead is",module.__class__.__name__.lower())
            if module.__class__.__name__.lower()=="relu":# no use
                model._modules[name] = neuron(T=T, thresh=1.0, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop)
            elif args.model_name in ['fcos_resnet50_fpn', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2']:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear','conv2d']:
                    model._modules[name] = module.out
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu':
                    thre = model._modules[name].scale
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop)
                elif "threhook" in module.__class__.__name__.lower():
                    model._modules[name] = exp_comp_neuron(module.out,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_relu_by_HMT(module,step_mode,T,neuron,num_thresholds,args)
            elif args.model_name in ['vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu':
                    print("module.__class__.__name__.lower() is",module.__class__.__name__.lower())
                    print("name is",name)
                    thre = model._modules[name].scale
                    #print(thre)
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop)
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear']:
                    model._modules[name] = MyLinear(module.out)
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['conv2d']:
                    model._modules[name] = MyConv(module.out)
                elif "dropout" in module.__class__.__name__.lower() or "avgpool2d" in module.__class__.__name__.lower() or "flatten" in module.__class__.__name__.lower():
                    model._modules[name] = MyPass(module)
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_relu_by_HMT(module,step_mode,T,neuron,num_thresholds,args)
            else:
                if "threhook" in module.__class__.__name__.lower(): #and (module.out.__class__.__name__.lower()=='relu' or module.out.__class__.__name__.lower()=='myat'):
                #if "threhook" in module.__class__.__name__.lower() and (module.out.__class__.__name__.lower()=='relu' or module.out.__class__.__name__.lower()=='myat'):    
                    print(f"threhook's out is {module.out.__class__.__name__.lower()}")
                    scale = 8
                    thre = model._modules[name].scale
                    # print(thre,module.out.__class__.__name__.lower())
                    # thre = thre*(thre>=0).float()*scale
                    if module.out.__class__.__name__.lower()=='relu':
                        thre = thre*(thre>=0).float()
                        model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop)
                    elif module.out.__class__.__name__.lower()=='myat':
                        thre = thre*(thre>=0).float()
                        thre2 = model._modules[name].scale2
                        thre2 = thre2*(thre2>=0).float()#*scale
                        model._modules[name] = MyatSequential(neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop),
                                                            neuron(T=T, thresh=thre2, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop),
                                                            AtNeuron(T=args.time,step_mode=args.step_mode,coding_type=args.coding_type))
                    elif module.out.__class__.__name__.lower() in ['linear','conv2d']:
                        model._modules[name] = nn.Sequential(neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop),
                                                             module.out)
                    # elif module.out.__class__.__name__.lower() in ['conv2d']:
                    #     model._modules[name] = module.out
                    else:
                        if module.out.__class__.__name__.lower() not in ['identity']:
                            # may not change the identity module, 
                            # though exp_comp_neuron-version identity is identical to identity, but brings extra cost
                            # however, such extra cost is also small
                            model._modules[name] = exp_comp_neuron(module.out,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
                elif module.__class__.__name__.lower() == 'mymul':
                    model._modules[name] = MulNeuron(T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_relu_by_HMT(module,step_mode,T,neuron,num_thresholds,args)
        return model
    
    def replace_identity_by_HMT(model,step_mode,T,neuron,num_thresholds,args):
        for name, module in model._modules.items():
            print("name ahead is",module.__class__.__name__.lower())
            if module.__class__.__name__.lower()=="identity":
                #print("oh yes I do converte an Identity!")
                model._modules[name] = neuron(T=T, thresh=1.0, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop)
            elif args.model_name in ['fcos_resnet50_fpn', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2']:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear','conv2d']:
                    model._modules[name] = module.out
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='identity':
                    thre = model._modules[name].scale
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop)
                elif "threhook" in module.__class__.__name__.lower():
                    model._modules[name] = exp_comp_neuron(module.out,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_identity_by_HMT(module,step_mode,T,neuron,num_thresholds,args)
            elif args.model_name in ['vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='identity':
                    #print("oh yes I do converte an Identity!")
                    thre = model._modules[name].scale
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop)
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear']:
                    model._modules[name] = MyLinear(module.out)
                elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['conv2d']:
                    model._modules[name] = MyConv(module.out)
                elif "dropout" in module.__class__.__name__.lower() or "avgpool2d" in module.__class__.__name__.lower() or "flatten" in module.__class__.__name__.lower():
                    model._modules[name] = MyPass(module)
                elif hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_identity_by_HMT(module,step_mode,T,neuron,num_thresholds,args)
            else:
                if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='identity':
                    #print("oh yes I do converte an Identity!")
                    thre = model._modules[name].scale
                    thre = thre*(thre>=0).float()
                    model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds,sop=args.sop)
                # as for ViT, only use linear neuron to transform, so many things are repeated in replace_relu & replace_identity
                if hasattr(module, "_modules"):
                    model._modules[name] = Converter.replace_identity_by_HMT(module,step_mode,T,neuron,num_thresholds,args)
        return model
    
    @staticmethod
    def fuse(fx_model: torch.fx.GraphModule, fuse_flag: bool = True) -> torch.fx.GraphModule:
        def matches_module_pattern(pattern: Iterable[type], node: fx.Node, modules: dict[str, any]) -> bool:
            if len(node.args) == 0:
                return False
            nodes: Tuple[any, fx.Node] = (node.args[0], node)
            for expected_type, current_node in zip(pattern, nodes):
                if not isinstance(current_node, fx.Node):
                    return False
                if current_node.op != 'call_module':
                    return False
                if not isinstance(current_node.target, str):
                    return False
                if current_node.target not in modules:
                    return False
                if type(modules[current_node.target]) is not expected_type:
                    return False
            return True

        def replace_node_module(node: fx.Node, modules: dict[str, any],
                                new_module: torch.nn.Module):
            def parent_name(target: str) -> Tuple[str, str]:
                """
                Splits a qualname into parent path and last atom.
                For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
                """
                *parent, name = target.rsplit('.', 1)
                return parent[0] if parent else '', name

            assert (isinstance(node.target, str))
            parent_name, name = parent_name(node.target)
            modules[node.target] = new_module
            setattr(modules[parent_name], name, new_module)

        if not fuse_flag:
            return fx_model
        patterns = [(nn.Conv1d, nn.BatchNorm1d),
                    (nn.Conv2d, nn.BatchNorm2d),
                    (nn.Conv3d, nn.BatchNorm3d)]

        modules = dict(fx_model.named_modules())

        for pattern in patterns:
            for node in fx_model.graph.nodes:
                if matches_module_pattern(pattern, node,
                                          modules):
                    if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                        continue
                    conv = modules[node.args[0].target]
                    bn = modules[node.target]
                    fused_conv = fuse_conv_bn_eval(conv, bn)
                    replace_node_module(node.args[0], modules,
                                        fused_conv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)
        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()  # remove unused bn modules
        fx_model.recompile()
        return fx_model

    #not used
    @staticmethod
    def replace_by_ifnode(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        * :ref:`API in English <Converter.replace_by_ifnode-en>`

        .. _Converter.replace_by_ifnode-cn:

        :param fx_model: origin model
        :type fx_model: torch.fx.GraphModule
        :return: model after conversion.
        :rtype: torch.fx.GraphModule

        ``replace_by_ifnode`` replace ReLU with spiking neuron like IF
        """
        hook_cnt = -1
        for node in fx_model.graph.nodes:
            if node.op != 'call_module':
                continue
            if type(fx_model.get_submodule(node.target)) is VoltageHook:
                if type(fx_model.get_submodule(node.args[0].target)) is nn.ReLU:
                    hook_cnt += 1
                    hook_node = node
                    relu_node = node.args[0]
                    if len(relu_node.args) != 1:
                        raise NotImplementedError('The number of relu_node.args should be 1.')
                    s = fx_model.get_submodule(node.target).scale.item()
                    target0 = 'snn tailor.' + str(hook_cnt) + '.0'  # voltage_scaler
                    target1 = 'snn tailor.' + str(hook_cnt) + '.1'  # IF_node
                    target2 = 'snn tailor.' + str(hook_cnt) + '.2'  # voltage_scaler
                    m0 = VoltageScaler(1.0 / s)
                    m1 = neuron.IFNode(v_threshold=1., v_reset=None)
                    m2 = VoltageScaler(s)
                    node0 = Converter._add_module_and_node(fx_model, target0, hook_node, m0,
                                                           relu_node.args)
                    node1 = Converter._add_module_and_node(fx_model, target1, node0, m1
                                                           , (node0,))
                    node2 = Converter._add_module_and_node(fx_model, target2, node1, m2, args=(node1,))

                    relu_node.replace_all_uses_with(node2)
                    node2.args = (node1,)
                    fx_model.graph.erase_node(hook_node)
                    fx_model.graph.erase_node(relu_node)
                    fx_model.delete_all_unused_submodules()
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    @staticmethod
    def _add_module_and_node(fx_model: fx.GraphModule, target: str, after: fx.Node, m: nn.Module,
                             args: Tuple) -> fx.Node:
        fx_model.add_submodule(target=target, m=m)
        with fx_model.graph.inserting_after(n=after):
            new_node = fx_model.graph.call_module(module_name=target, args=args)
        return new_node