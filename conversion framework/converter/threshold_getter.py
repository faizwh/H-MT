import torch
import torch.nn as nn
from torch import fx
from torch import nn
from tqdm import tqdm
from typing import Tuple
import numpy as np
import os
import utils
class Threshold_Getter(nn.Module):
    def __init__(self, dataloader, mode='99.9%', level = 'layer', device=None, momentum=0.1, output_fx=False,do_data_collection=False):
        super().__init__()
        self.dataloader = dataloader
        self.mode = mode
        self.level = level
        self.device = device
        self.momentum = momentum
        self.output_fx = output_fx
        self.do_data_collection = do_data_collection
        #print(self.do_data_collection)
    def forward(self, model: nn.Module):
        if self.device is None:
            self.device = next(model.parameters()).device
        if self.output_fx:
            model = fx.symbolic_trace(model).to(self.device)
            model.eval()
            model_with_hook = Threshold_Getter.set_voltagehook_under_graph(model, mode=self.mode, momentum=self.momentum).to(self.device)
        else:
            model.eval()
            model_with_hook = Threshold_Getter.replace_nonlinear_by_hook(model, mode=self.mode, momentum=self.momentum, 
                                                                         level=self.level,do_data_collection=self.do_data_collection).to(self.device)
            model_with_hook = Threshold_Getter.replace_identity_by_hook(model_with_hook, 
                                                                        mode=self.mode, momentum=self.momentum, level=self.level,
                                                                        do_data_collection=self.do_data_collection).to(self.device)
        for batch_idx, (imgs, _) in enumerate(tqdm(self.dataloader)):
            # if batch_idx>10:
            #     break
            if isinstance(imgs,tuple):
                imgs = list(img.to(self.device) for img in imgs)
            else:
                imgs = imgs.to(self.device)
            model_with_hook(imgs)
        return model_with_hook
    



    @staticmethod
    def get_scale_from_var(model,T=64,a=0,b=64):
        for name, module in model._modules.items():
            if module.__class__.__name__.lower()=="threhook":
                print(module.out.__class__.__name__.lower())
                if module.out.__class__.__name__.lower()=="relu":
                    model._modules[name].get_scale_from_var(T=T,a=a,b=b)
                elif module.out.__class__.__name__.lower()=="identity":
                    model._modules[name].get_scale_from_var(T=T,a=-b,b=b)
                elif module.out.__class__.__name__.lower()=="myat":
                    model._modules[name].get_scale_from_var(T=T,a=-b,b=b)
                elif module.out.__class__.__name__.lower()=="conv2d":
                        model._modules[name].get_scale_from_var(T=T,a=-b,b=b)
                elif module.out.__class__.__name__.lower()=="linear":
                        model._modules[name].get_scale_from_var(T=T,a=-b,b=b)
                else:
                    model._modules[name].scale = torch.ones_like(model._modules[name].mean)
                print(f"the name here is {module.out.__class__.__name__.lower()}")
            elif hasattr(module, "_modules"):
                Threshold_Getter.get_scale_from_var(module, T=T,a=a,b=b)
        return model
    
    # used only for HMT with % / max
    def get_scale_for_HMT(model,n=64,change_mode="%"):
        for name, module in model._modules.items():
            if module.__class__.__name__.lower()=="threhook":
                model._modules[name].get_scale_for_HMT(n=n,change_mode=change_mode)
            elif hasattr(module, "_modules"):
                Threshold_Getter.get_scale_for_HMT(module,n=n,change_mode=change_mode)
        return model
    
    @staticmethod
    def replace_identity_by_hook(model, momentum, mode, level,do_data_collection=False):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                Threshold_Getter.replace_identity_by_hook(module, mode=mode, momentum=momentum, level=level,
                                                          do_data_collection=do_data_collection)
            if module.__class__.__name__.lower() in ["identity"]:
                # 所有这些全给挂上钩子
                #print(f"hang a hook, the do_data_collection is {do_data_collection}")
                model._modules[name] = ThreHook(mode=mode, momentum=momentum, out_layer=module, level=level,
                                                do_data_collection=do_data_collection)
                print("Replace "+module.__class__.__name__.lower()+" By ThreHook")
        return model


    @staticmethod
    def replace_nonlinear_by_hook(model, momentum, mode, level,do_data_collection=False):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                Threshold_Getter.replace_nonlinear_by_hook(module, mode=mode, momentum=momentum, level=level,
                                                           do_data_collection=do_data_collection)
            if module.__class__.__name__.lower() in ["relu",'gelu','silu','layernorm','groupnorm','softmax','myat','linear','conv2d']:
                # hang hook for all these
                #print(f"hang a hook, the do_data_collection is {do_data_collection}")
                model._modules[name] = ThreHook(mode=mode, momentum=momentum, out_layer=module, level=level,
                                                do_data_collection=do_data_collection)
                # print("Replace "+module.__class__.__name__.lower()+" By ThreHook")
        return model

    @staticmethod
    def set_voltagehook_under_graph(fx_model, mode='Max', momentum=0.1):
        hook_cnt = -1
        for node in fx_model.graph.nodes:
            if node.op != 'call_module':
                continue
            if type(fx_model.get_submodule(node.target)) is nn.ReLU:
                hook_cnt += 1
                target = 'snn tailor.' + str(hook_cnt) + '.0'  # voltage_hook
                m = ThreHook(momentum=momentum, mode=mode)
                new_node = Threshold_Getter._add_module_and_node(fx_model, target, node, m, (node,))
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model
    
    @staticmethod
    def _add_module_and_node(fx_model: fx.GraphModule, target: str, after: fx.Node, m: nn.Module, args: Tuple) -> fx.Node:
        fx_model.add_submodule(target=target, m=m)
        with fx_model.graph.inserting_after(n=after):
            new_node = fx_model.graph.call_module(module_name=target, args=args)
        return new_node
    
    @staticmethod
    def save_model(model, model_path, mode_fx, code_path=None):
        if mode_fx:
            return Threshold_Getter.save_fx_model(model, model_path+'.pt')
        else:
            return Threshold_Getter.save_module_model(model, model_path+'.pth')
        
    @staticmethod
    def load_model(model_path, mode_fx, code_path=None,model=None):
        if mode_fx:
            return Threshold_Getter.load_fx_model(model_path+'.pt')
        else:
            if model is None:
                print("Must give a model to load state dict")
                exit(0)
            return Threshold_Getter.load_module_model(model=model, model_path=model_path+'.pth')
    
    @staticmethod
    def save_module_model(model, model_path):
        model_dir = os.path.dirname(model_path)
        print(model_dir)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)  # Create the directory if it doesn't exist
        torch.save(model.state_dict(), model_path)
                   
    @staticmethod
    def load_module_model(model, model_path):
        state_dict = torch.load(model_path,weights_only=True)
        model.load_state_dict(state_dict)
        return model
    
    @staticmethod
    def save_fx_model(fx_model, model_path, code_path=None):
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)  # Create the directory if it doesn't exist
        if code_path is not None:
            with open(code_path, "w") as f:
                f.write(fx_model.code)
        torch.save(fx_model, model_path)
        
    @staticmethod   
    def load_fx_model(model_path, code_path=None):
        code = None
        model = torch.load(model_path,weights_only=True)
        if code_path is not None:
            with open(code_path, "r") as f:
                code = f.read()
            return model, code
        return model

    
def merge_dims(tensor, dims):
    """
    Merge all specified dimensions into the first dimension.
    """
    dims = sorted(dims)  # Ensure that dims are sorted in ascending order.
    shape = list(tensor.shape)
    merged_size = 1
    for dim in dims:
        merged_size *= shape[dim]

    # build up new_shape
    new_shape = [merged_size] + [shape[i] for i in range(len(shape)) if i not in dims]
    # permute and reshape
    # print(tensor.shape)
    permuted_tensor = tensor.permute(*dims, *[i for i in range(tensor.ndim) if i not in dims]).contiguous()
    # print(permuted_tensor.shape)
    return permuted_tensor.view(*new_shape).contiguous(), shape, dims

def restore_dims(percentile_tensor, original_shape, dims):
    """
    Based on the original shape and dims, restore the percentile results, with the size of the merged dimensions set to 1
    """
    dims = sorted(dims)
    new_shape = [1 if i in dims else original_shape[i] for i in range(len(original_shape))]
    return percentile_tensor.view(*new_shape).contiguous()

#compute k1 for iteration
def calculate_better(mean, var, scale, n=64, a=0,b=64):
    device = mean.device
    dtype = mean.dtype
    
    sigma = torch.sqrt(var)  
    theta = scale            
    pi = torch.tensor(torch.pi, dtype=dtype, device=device)  # π 
    
    i = torch.arange(1, n + 1, dtype=dtype, device=device).view(*([1] * mean.dim()), n)

    coeff = ((2 * i - 1) / (2 * n)) * theta.unsqueeze(-1)
    
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    erf_input = (coeff - mean.unsqueeze(-1)) / (sqrt2 * sigma.unsqueeze(-1))
    erf_term = torch.erf(erf_input)
    
    term1_sum = torch.sum((1 / n) * erf_term, dim=-1)
    

    term2_sum = torch.sum(((2 * i - 1) / n**2) * erf_term, dim=-1)
    
    exp_input = -((coeff - mean.unsqueeze(-1))**2) / (2 * var.unsqueeze(-1))
    exp_term = torch.exp(exp_input)
    term3_sum = torch.sum((1 / n) * exp_term, dim=-1)
    
    denominator = 1 - term2_sum
    
    part1 = (mean / theta) * (1 - term1_sum) / denominator
    sqrt_pi_over_2 = torch.sqrt(pi / 2)
    part2 = (sigma / (sqrt_pi_over_2 * theta)) * term3_sum / denominator
    
    k1 = part1 + part2
    return k1

def calculate_better_harmony(mean, var, scale, n=64, a=0,b=64):
    assert a<=0 and isinstance(a,int),"a isn't an integer less than 0, can't do optimization with calculate_better_harmony"
    assert b>=0 and isinstance(b,int),"b isn't an integer greater than 0, can't do optimization with calculate_better_harmony"
    device = mean.device
    dtype = mean.dtype
    
    epsilon = 1e-8
    sigma = torch.sqrt(var+epsilon)
    theta = scale           
    pi = torch.tensor(torch.pi, dtype=dtype, device=device)  # π
    
    i = torch.arange(a+1, b+1, dtype=dtype, device=device).view(*([1] * mean.dim()), b-a)

    coeff = ((2 * i - 1) / (2 * n)) * theta.unsqueeze(-1)

    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    
    erf_input = (coeff - mean.unsqueeze(-1)) / (sqrt2 * sigma.unsqueeze(-1))
    
    erf_term = torch.erf(erf_input)
    
    term1_sum = torch.sum(erf_term, dim=-1)
    
    term2_sum = torch.sum((2 * i - 1) * erf_term, dim=-1)
    
    exp_input = -((coeff - mean.unsqueeze(-1))**2) / (2 * (var+epsilon).unsqueeze(-1))
    
    exp_term = torch.exp(exp_input)
    
    term3_sum = torch.sum(n * exp_term, dim=-1)
    
    denominator = a**2 + b**2 - term2_sum
    
    part1 = (mean / theta) * n * (a + b - term1_sum) / denominator
    
    sqrt_pi_over_2 = torch.sqrt(pi / 2)
    part2 = (sigma / (sqrt_pi_over_2 * theta)) * term3_sum / denominator
    
    k1 = part1 + part2
    
    return k1

class ThreHook(nn.Module):
    def __init__(self, scale=1.0, mode='Max', momentum=0.1, out_layer='Identitiy',level='layer',do_data_collection=False):
        super().__init__()
        self.register_buffer('scale', None)
        self.register_buffer('scale2', None)
        self.register_buffer('var', None)
        self.register_buffer('mean', None)
        self.register_buffer('var2', None)
        self.register_buffer('mean2', None)
        self.mode = mode 
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        self.register_buffer('num_batches_tracked2', torch.tensor(0))
        self.momentum = momentum
        self.out = out_layer

        self.scale=None
        self.scale_percent=None
        self.scale_max=None
        self.scale2_percent=None
        self.scale2_max=None

        self.level=level 
        self.do_data_collection=do_data_collection

    def get_scale_for_HMT(self, n=64,change_mode="%"):
        # self.scale = torch.ones_like(self.mean)*0.1
        # self.scale = self.mean+4*self.var
        # print(self.mean,self.var,self.scale)
        if change_mode=="%":
            if self.scale_percent is not None:
                self.scale = self.scale_percent.clone() 
            #self.scale = self.scale.clone() 
        elif change_mode.lower()=="max":
            if self.scale_max is not None:
                self.scale = self.scale_max.clone()
        else:
            print("not support this change_mode, remember this is only for quantile and max")
            return
        #print(f"in get_scale_for_HMT, self.scale is None?",self.scale is None)
        nan_mask = torch.isnan(self.scale)
        if nan_mask.any():
            print("in get_scale_for_HMT")
            print(f"original scale already have NAN elements")
        
        tmp = self.scale.clone().detach()
        #print(self.scale)
        self.scale = self.scale/(2**n-1)*2.
        #print(self.scale)
        nan_mask = torch.isnan(self.scale)
        if nan_mask.any():
            print("in get_scale_for_HMT")
            print(f"Found {nan_mask.sum().item()} NaN values, replacing with original thresholds\n")
            tmp_values = tmp[nan_mask].to(self.scale.dtype)
            self.scale[nan_mask] = tmp_values
        else:
            print(f"Found 0 NaN values, replacing with original thresholds\n")


    #iteration and update
    def get_scale_from_var(self, T=64,a=0,b=64):
        # self.scale = torch.ones_like(self.mean)*0.1
        # self.scale = self.mean+4*self.var
        # print(self.mean,self.var,self.scale)
        """
         the normalization (for initialization), i.e. /(b/T), is very very important to overcome extreme cases when too much NAN occurs!
         but it's specially designed for HMT/MTH, as the theorem doesn't need some necessary relationship between (a,b)&n(or the T here)
         and we wonder if this normalization is necessary afterwards
        """
        self.scale = torch.ones_like(self.mean)#/((b-a)/T) 
        tmp = self.scale.clone().detach()
        nan_mask = torch.isnan(self.scale)
        if nan_mask.any():
            print("in get_scale_from_val")
            print(f"original scale already have NAN elements")

        for idx in range(1500):
            #self.scale = self.scale*calculate_better(self.mean,self.var,self.scale,n=T)
            k1 = calculate_better_harmony(self.mean,self.var,self.scale,n=T,a=a,b=b)
            #nan_mask = torch.isnan(k1)
            #k1[nan_mask] = 1.
            self.scale = self.scale* k1

        nan_mask = torch.isnan(self.scale)
        if nan_mask.any():
            print("in get_scale_from_val")
            print(f"Found {nan_mask.sum().item()} NaN values, replacing with original thresholds\n")
            tmp_values = tmp[nan_mask].to(self.scale.dtype)
            self.scale[nan_mask] = tmp_values
            #self.scale[nan_mask] = 0
            nan_mask = torch.isnan(self.scale)
        else:
            print(f"Found 0 NaN values, replacing with original thresholds\n")

        if self.out.__class__.__name__.lower() == 'myat':
            self.scale2 = torch.ones_like(self.mean2)#/((b-a)/T)
            tmp = self.scale2.clone().detach()
            nan_mask = torch.isnan(self.scale2)
            if nan_mask.any():
                print("in get_scale_from_val")
                print(f"original scale already have NAN elements")
            for idx in range(1500):
                #self.scale = self.scale*calculate_better(self.mean,self.var,self.scale,n=T)
                k1_2 = calculate_better_harmony(self.mean2,self.var2,self.scale2,n=T,a=a,b=b)
                self.scale2 = self.scale2* k1_2
            nan_mask = torch.isnan(self.scale2)
            if nan_mask.any():
                print("in get_scale_from_val")
                print(f"Found {nan_mask.sum().item()} NaN values, replacing with original thresholds\n")
                tmp_values = tmp[nan_mask].to(self.scale2.dtype)
                self.scale2[nan_mask] = tmp_values
                nan_mask = torch.isnan(self.scale2)
            else:
                print(f"Found 0 NaN values, replacing with original thresholds\n")
            
    def forward(self, x, *args):
        if self.level== 'layer':
            dims = [i for i in range(x.ndim)]
        elif self.level == 'channel':
            dims = [i for i in range(x.ndim) if i != 1]
        elif self.level=='neuron':
            dims = [0]
        err_msg = 'You have used a non-defined Method to get Threshold.'
        
        if self.out.__class__.__name__.lower() == 'myat':
            # Conduct statistical analysis on the input
            if self.mode[-1] == '%':
                if self.do_data_collection:  # quantile
                    try:
                        percentile_val = float(self.mode[:-1])
                    except ValueError:
                        raise ValueError(f"Invalid percentile value in mode: {self.mode}")
                    if not (0 < percentile_val < 100):
                        raise ValueError(f"Percentile value must be between 0 and 100, got {percentile_val}")

                    merged_x, original_shape, dims = merge_dims(x.clone().detach(), dims)
                    s_t = torch.quantile(merged_x, percentile_val / 100.0, dim=0, keepdim=True).detach()
                    s_t = restore_dims(s_t, original_shape, dims)
                    
                    merged_x2, original_shape2, dims2 = merge_dims(args[0].clone().detach(), dims)
                    s_t2 = torch.quantile(merged_x2, percentile_val / 100.0, dim=0, keepdim=True).detach()
                    s_t2 = restore_dims(s_t2, original_shape2, dims2)
                    
                    if self.scale_percent is None:
                        self.scale_percent = s_t.clone()
                        self.scale2_percent = s_t2.clone()
                    else:
                        self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                        self.scale_percent = (1 - self.momentum) * self.scale_percent + self.momentum * s_t
                        self.scale2_percent = (1 - self.momentum) * self.scale2_percent + self.momentum * s_t2
                    self.scale=self.scale_percent.clone()
                    self.scale2=self.scale2_percent.clone()
                    self.num_batches_tracked += x.shape[0]

            elif self.mode.lower() in ['max']:
                if self.do_data_collection: # max input
                    s_t = x.clone().detach().amax(dim=dims, keepdim=True).detach()
                    s_t2 = args[0].clone().detach().amax(dim=dims, keepdim=True).detach()
                    if self.scale_max is None:
                        self.scale_max = s_t.clone()
                        self.scale2_max = s_t2.clone()
                    else:
                        self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                        self.scale_max = (1 - self.momentum) * self.scale_max + self.momentum * s_t
                        self.scale2_max = (1 - self.momentum) * self.scale2_max + self.momentum * s_t2
                    self.scale=self.scale_max.clone()
                    self.scale2=self.scale2_max.clone()
                    #print("I do clone!")
                    self.num_batches_tracked += x.shape[0]

            elif self.mode.lower() in ['var']:
                if self.do_data_collection:
                    mean_t = x.mean(dim=dims, keepdim=True).detach()
                    var_t = x.var(dim=dims, keepdim=True, unbiased=False).detach()  
                    if self.mean is None:
                        self.mean = mean_t.clone()
                        self.var = var_t.clone()
                    else:
                        delta = mean_t - self.mean
                        self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                        self.mean = (1 - self.momentum) * self.mean + self.momentum * mean_t
                        self.var = (1 - self.momentum) * self.var + self.momentum * (var_t + (1-self.momentum) * delta * delta)
                    self.num_batches_tracked += x.shape[0]

                    mean2_t = args[0].mean(dim=dims, keepdim=True).detach()
                    var2_t = args[0].var(dim=dims, keepdim=True, unbiased=False).detach()
                    if self.mean2 is None:
                        self.mean2 = mean2_t.clone()
                        self.var2 = var2_t.clone()
                    else:
                        delta2 = mean2_t - self.mean2
                        self.momentum2 = args[0].shape[0]/(self.num_batches_tracked2+args[0].shape[0])
                        self.mean2 = (1 - self.momentum2) * self.mean2 + self.momentum2 * mean2_t
                        self.var2 = (1 - self.momentum2) * self.var2 + self.momentum2 * (var2_t + (1-self.momentum2) * delta2 * delta2)
                    self.num_batches_tracked2 += args[0].shape[0]
            else:
                raise NotImplementedError(err_msg)
            x = self.out(x,args[0])
            return x
        else:
            if self.mode.lower() in ['var']:
                if self.do_data_collection:
                    mean_t = x.mean(dim=dims, keepdim=True).detach()
                    var_t = x.var(dim=dims, keepdim=True, unbiased=False).detach()
                    
                    if self.mean is None:
                        self.mean = mean_t.clone()
                        self.var = var_t.clone()
                    else:
                        delta = mean_t - self.mean
                        self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                        self.mean = (1 - self.momentum) * self.mean + self.momentum * mean_t
                        self.var = (1 - self.momentum) * self.var + self.momentum * (var_t + (1-self.momentum) * delta * delta)
                    self.num_batches_tracked += x.shape[0]
                
            if self.out.__class__.__name__.lower() not in ['linear','conv2d']:
                x = self.out(x)
            if self.mode[-1] == '%':
                if self.do_data_collection:
                    try:
                        percentile_val = float(self.mode[:-1])
                    except ValueError:
                        raise ValueError(f"Invalid percentile value in mode: {self.mode}")
                    if not (0 < percentile_val < 100):
                        raise ValueError(f"Percentile value must be between 0 and 100, got {percentile_val}")

                    merged_x, original_shape, dims = merge_dims(x.clone().detach(), dims)
                    s_t = torch.quantile(merged_x, percentile_val / 100.0, dim=0, keepdim=True).detach()
                    s_t = restore_dims(s_t, original_shape, dims)
                    if self.scale_percent is None:
                        self.scale_percent = s_t.clone()
                    else:
                        self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                        self.scale_percent = (1 - self.momentum) * self.scale_percent + self.momentum * s_t
                    self.num_batches_tracked += x.shape[0]
                    self.scale=self.scale_percent.clone()
                    if (self.scale_percent is None):
                        print("self.scale_percent is None now?",self.scale_percent is None)
            
            elif self.mode.lower() in ['max']: 
                if self.do_data_collection:
                    s_t = x.clone().detach().amax(dim=dims, keepdim=True).detach()
                    if self.scale_max is None:
                        self.scale_max = s_t.clone()
                    else:
                        self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                        self.scale_max = (1 - self.momentum) * self.scale_max + self.momentum * s_t
                    self.num_batches_tracked += x.shape[0]
                    self.scale=self.scale_max.clone()
            else:
                if self.mode.lower() not in ['var']:
                    raise NotImplementedError(err_msg)
                
            if self.out.__class__.__name__.lower() in ['linear','conv2d']:
                x = self.out(x)
            return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in state_dict:
            if key.endswith('.scale'):
                saved_scale = state_dict[key]
                if self.scale is None or saved_scale.shape != self.scale.shape:
                    self.scale = saved_scale.new_zeros(*saved_scale.shape)
            if key.endswith('.scale2'):
                saved_scale2 = state_dict[key]
                if self.scale2 is None or saved_scale2.shape != self.scale2.shape:
                    self.scale2 = saved_scale2.new_zeros(*saved_scale2.shape)
            if key.endswith('.var'):
                saved_var = state_dict[key]
                if self.var is None or saved_var.shape != self.var.shape:
                    self.var = saved_var.new_zeros(*saved_var.shape)
            if key.endswith('.mean'):
                saved_mean = state_dict[key]
                if self.mean is None or saved_mean.shape != self.mean.shape:
                    self.mean = saved_mean.new_zeros(*saved_mean.shape)
            if key.endswith('.var2'):
                saved_var2 = state_dict[key]
                if self.var2 is None or saved_var2.shape != self.var2.shape:
                    self.var2 = saved_var2.new_zeros(*saved_var2.shape)
            if key.endswith('.mean2'):
                saved_mean2 = state_dict[key]
                if self.mean2 is None or saved_mean2.shape != self.mean2.shape:
                    self.mean2 = saved_mean2.new_zeros(*saved_mean2.shape)
            
            """
            if key.endswith('.scale_percent'):
                saved_scale_percent = state_dict[key]
                if self.scale_percent is None or saved_scale_percent.shape != self.scale_percent.shape:
                    self.scale_percent = saved_scale_percent.new_zeros(*saved_scale_percent.shape)
            if key.endswith('.scale2_percent'):
                saved_scale2_percent = state_dict[key]
                if self.scale2_percent is None or saved_scale2_percent.shape != self.scale2_percent.shape:
                    self.scale2_percent = saved_scale2_percent.new_zeros(*saved_scale2_percent.shape)
            if key.endswith('.scale_max'):
                saved_scale_max = state_dict[key]
                if self.scale_max is None or saved_scale_max.shape != self.scale_max.shape:
                    self.scale_max = saved_scale_max.new_zeros(*saved_scale_max.shape)
            if key.endswith('.scale2_max'):
                saved_scale2_max = state_dict[key]
                if self.scale2_max is None or saved_scale2_max.shape != self.scale2_max.shape:
                    self.scale2_max = saved_scale2_max.new_zeros(*saved_scale2_max.shape)
            """
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)