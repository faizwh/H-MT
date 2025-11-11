import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import MergeTemporalDim, ExpandTemporalDim

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama=1.0):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class IF(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s'):
        super(IF, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        if step_mode=='s':
            self.T = 0
        else:
            self.T = T
    def reset(self):
        if self.step_mode == 's':
            self.T = 0
    def forward(self, x):
        if self.step_mode == 'm':
            x = self.expand(x)
            mem = 0.5 * self.thresh
            spike_pot = []
            thre = self.thresh
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre) * thre
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
            return x
        else:# self.step_mode == 's':
            if self.T==0:
                self.mem = 0.5 * self.thresh
            self.mem = self.mem + x
            spike = self.act(self.mem - self.thresh) * self.thresh
            self.mem = self.mem - spike
            self.T = self.T + 1
            return spike
        
class IF_with_neg(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s'):
        super(IF_with_neg, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        if step_mode=='s':
            self.T = 0
        else:
            self.T = T
    def reset(self):
        if self.step_mode == 's':
            self.T = 0
    def forward(self, x):
        if self.step_mode == 'm':
            x = self.expand(x)
            mem = 0.5 * self.thresh
            spike_pot = []
            thre = self.thresh
            cumulative_out = torch.zeros_like(x[0])
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre) * thre
                
                spike_neg = self.act(-mem) * ((cumulative_out-thre)>=0).float() * thre
                spike = spike-spike_neg
                cumulative_out = cumulative_out + spike
                
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
            return x
        else:# self.step_mode == 's':
            if self.T==0:
                self.mem = 0.5 * self.thresh
                self.cumulative_out = torch.zeros_like(x)
                
            self.mem = self.mem + x
            spike = self.act(self.mem - self.thresh) * self.thresh
            
            spike_neg = self.act(-self.mem) * ((self.cumulative_out>=self.thresh).float()) * self.thresh
            spike = spike - spike_neg
            self.cumulative_out = self.cumulative_out + spike
            
            self.mem = self.mem - spike
            self.T = self.T + 1
            return spike
        
class IF_diff(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s'):
        super(IF_diff, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T+1)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        if step_mode=='s':
            self.T = -1
        else:
            self.T = T
    def reset(self):
        if self.step_mode == 's':
            self.T = -1
    def forward(self, x):
        if self.step_mode == 'm':
            x = self.expand(x)
            mem = 0.5 * self.thresh
            spike_pot = []
            thre = self.thresh
            spike_pot.append(torch.zeros_like(x[0]))
            exp_in = x[0].clone().detach()
            exp_out = torch.zeros_like(x[0])
            cumulative_out = torch.zeros_like(x[0])
            for t in range(self.T):
                # print(t,thre[0][0][0][0],(x[t+1] - x[0])[0][0][0][0],exp_in[0][0][0][0],exp_out[0][0][0][0])
                mem = mem + x[t+1] - x[0] + exp_in - exp_out
                cumulative_out = cumulative_out + exp_out
                spike = self.act(mem - thre) * thre
                spike_neg = self.act(-mem) * ((cumulative_out-thre)>=0).float() * thre
                spike = spike-spike_neg
                # spike = spike * (spike.abs()/(t+1)>=exp_out.abs()/16).float()
                
                cumulative_out = cumulative_out + spike 
                mem = mem - spike
                spike_pot.append(spike)
                exp_in = exp_in + (x[t+1] - x[0])/(t+1)
                exp_out = exp_out + spike/(t+1)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
            return x
        else:# self.step_mode == 's':
            if self.T==-1:
                self.bias = x.clone()
                self.exp_in = x.clone()
                self.exp_out = torch.zeros_like(x)
                self.cumulative_out = torch.zeros_like(x)
                self.T = self.T+1
                return torch.zeros_like(x)
            if self.T==0:
                self.mem = 0.5 * self.thresh
            self.mem = self.mem + x - self.bias + self.exp_in - self.exp_out
            spike = self.act(self.mem - self.thresh) * self.thresh
            
            self.cumulative_out = self.cumulative_out + self.exp_out
            spike_neg = self.act(-self.mem) * ((self.cumulative_out-self.thresh)>=0).float() * self.thresh
            spike = spike-spike_neg
            self.cumulative_out = self.cumulative_out + spike
            
            self.mem = self.mem - spike
            # self.mem = self.mem - (spike!=0).float()*self.mem
            
            self.exp_in = self.exp_in + (x - self.bias)/(self.T+1)
            self.exp_out = self.exp_out + spike/(self.T+1)
            # self.exp_out = self.exp_out + (spike!=0).float()*self.mem/(self.T+1)
            
            self.T = self.T + 1
            return spike
        # 12 -11 0 0 0
        # 12 0 0 0 0 
        
class LIF(nn.Module):
    def __init__(self, T=0, thresh=1.0, tau=0.99, step_mode='s'):
        super(LIF, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        self.tau = tau
        if step_mode == 's':
            self.T = 0
        else:
            self.T = T

    def reset(self):
        if self.step_mode == 's':
            self.T = 0

    def forward(self, x):
        if self.step_mode == 'm':
            x = self.expand(x)
            mem = 0.5 * self.thresh / self.tau
            spike_pot = []
            thre = self.thresh
            for t in range(self.T):
                mem = self.tau * mem + x[t, ...]  # 加入泄漏
                spike = self.act(mem - thre) * thre
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
            return x
        else:
            if self.T == 0:
                self.mem = 0.5 * self.thresh / self.tau
            self.mem = self.tau * self.mem + x  # 加入泄漏
            spike = self.act(self.mem - self.thresh) * self.thresh
            self.mem = self.mem - spike
            self.T = self.T + 1
            return spike

class LIF_with_neg(nn.Module):
    def __init__(self, T=0, thresh=1.0, tau=0.99, step_mode='s'):
        super(LIF_with_neg, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        self.tau = tau
        if step_mode == 's':
            self.T = 0
        else:
            self.T = T

    def reset(self):
        if self.step_mode == 's':
            self.T = 0

    def forward(self, x):
        if self.step_mode == 'm':
            x = self.expand(x)
            mem = 0.5 * self.thresh / self.tau
            spike_pot = []
            thre = self.thresh
            cumulative_out = torch.zeros_like(x[0])
            for t in range(self.T):
                mem = self.tau * mem + x[t, ...]  # 加入泄漏
                spike = self.act(mem - thre) * thre
                
                spike_neg = self.act(-mem) * ((cumulative_out-thre)>=0).float() * thre
                spike = spike-spike_neg
                cumulative_out = cumulative_out + spike
                
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
            return x
        else:
            if self.T == 0:
                self.mem = 0.5 * self.thresh / self.tau
                self.cumulative_out = torch.zeros_like(x)
                self.mul = 1
            self.mem = self.tau * self.mem + x  # 加入泄漏
            spike = self.act(self.mem - self.thresh) * self.thresh
            
            spike_neg = self.act(-self.mem) * ((self.cumulative_out-self.thresh*self.mul)>=0).float() * self.thresh
            spike = spike-spike_neg
            self.cumulative_out = self.cumulative_out + spike*self.mul
            self.mul = self.mul/self.tau
            
            # 输入增加，更早发射脉冲，阈值代表更小的值
            self.mem = self.mem - spike
            self.T = self.T + 1
            return spike
        
class LIF_diff(nn.Module):
    def __init__(self, T=0, thresh=1.0, tau=1.00, step_mode='s'):
        super(LIF_diff, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T+1)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        self.tau = tau
        if step_mode == 's':
            self.T = -1
        else:
            self.T = T

    def reset(self):
        if self.step_mode == 's':
            self.T = -1

    def forward(self, x):
        if self.step_mode == 'm':
            x = self.expand(x)
            mem = 0.5 * self.thresh / self.tau
            spike_pot = []
            thre = self.thresh
            exp_in = torch.zeros_like(x[0])
            exp_out = torch.zeros_like(x[0])
            cumulative_out = torch.zeros_like(x[0])
            mul = 1
            for t in range(self.T):
                mem = self.tau * mem + x[t+1] - x[0] + x[0]/mul  + exp_in/mul - exp_out/mul
                spike = self.act(mem - thre) * thre
                
                cumulative_out = cumulative_out + exp_out
                spike_neg = self.act(-mem) * ((cumulative_out-thre)>=0).float() * thre
                spike = spike-spike_neg
                cumulative_out = cumulative_out + spike
                
                mem = mem - spike
                
                exp_in = exp_in + (x[t+1] - x[0])*mul/(t+1)
                exp_out = exp_out + spike*mul/(t+1)
                
                mul = mul/self.tau
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
            return x
        else:
            if self.T==-1:
                self.bias = x.clone()
                self.exp_in = torch.zeros_like(x)
                self.exp_out = torch.zeros_like(x)
                self.T = self.T+1
                return torch.zeros_like(x)
            if self.T == 0:
                self.mem = 0.5 * self.thresh / self.tau
                self.cumulative_out = torch.zeros_like(x)
                self.mul = 1
            self.mem = self.tau * self.mem + x - self.bias + self.bias/self.mul  + self.exp_in - self.exp_out
            spike = self.act(self.mem - self.thresh) * self.thresh
            
            self.cumulative_out = self.cumulative_out + self.exp_out
            spike_neg = self.act(-self.mem) * ((self.cumulative_out-self.thresh)>=0).float() * self.thresh
            spike = spike-spike_neg
            
            # 输入增加，更早发射脉冲，阈值代表更小的值
            self.mem = self.mem - spike
            
            self.exp_in = (self.exp_in + (x - self.bias)/(self.T+1)) * self.tau
            self.exp_out = (self.exp_out + spike/(self.T+1)) * self.tau
            self.cumulative_out = (self.cumulative_out + spike)* self.tau
            
            # self.mul = self.mul/self.tau
            self.T = self.T + 1
            return spike
        
        
class MTH(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        super(MTH, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        self.num_thresholds = num_thresholds
        if step_mode=='s':
            self.T = 0
        else:
            self.T = T
    def reset(self):
        if self.step_mode == 's':
            self.T = 0
    def forward(self, x):
        if self.step_mode == 'm':
            time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
            num_additional_dims = x.dim()
            for _ in range(num_additional_dims):
                time_steps = time_steps.unsqueeze(-1)
            thre_shaped = self.thresh.unsqueeze(0) / time_steps  # 形状: (num_thresholds, 1, ...)
            thre_threshold = thre_shaped * 0.75  # 等同于 thre_shaped * 3/4
            x = self.expand(x)
            mem = torch.zeros_like(x[0])
            spike_pot = []
            for t in range(self.T):
                mem = mem + x[t]  # 更新膜电位
                pos_false_spikes = (mem.unsqueeze(0) >= thre_threshold).float()  # 形状: (num_thresholds, channels, ...)
                pos_true = pos_false_spikes.clone()
                pos_true[1:] -= pos_false_spikes[:-1]
                spike = torch.sum(thre_shaped * pos_true, dim=0)  # 形状: (channels, ...)
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)  # 形状: (T, channels, ...)
            x = self.merge(x)
            return x
        else:
            if self.T==0:
                self.mem = torch.zeros_like(x)
                time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
                num_additional_dims = x.dim()
                for _ in range(num_additional_dims):
                    time_steps = time_steps.unsqueeze(-1)
                self.thre_shaped = self.thresh.unsqueeze(0) / time_steps  # 形状: (num_thresholds, 1, ...)
            self.mem = self.mem + x
            pos_false_spikes = (self.mem.unsqueeze(0) >= (self.thre_shaped*0.75)).float()
            pos_true = pos_false_spikes.clone()
            pos_true[1:] -= pos_false_spikes[:-1]
            spike = torch.sum(self.thre_shaped * pos_true, dim=0)
            self.mem = self.mem - spike
            self.T = self.T + 1
            return spike
        
class MTH_with_neg(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        super(MTH_with_neg, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        self.num_thresholds = num_thresholds
        if step_mode=='s':
            self.T = 0
        else:
            self.T = T
    def reset(self):
        if self.step_mode == 's':
            self.T = 0
    def forward(self, x):
        if self.step_mode == 'm':
            time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
            num_additional_dims = x.dim()
            for _ in range(num_additional_dims):
                time_steps = time_steps.unsqueeze(-1)
            thre_shaped = self.thresh.unsqueeze(0) / time_steps  # 形状: (num_thresholds, 1, ...)
            thre_threshold = thre_shaped * 0.75  # 等同于 thre_shaped * 3/4
            x = self.expand(x)
            mem = torch.zeros_like(x[0])
            spike_pot = []
            cumulative_out = torch.zeros_like(x[0])
            for t in range(self.T):
                mem = mem + x[t]  # 更新膜电位
                pos_false_spikes = (mem.unsqueeze(0) >= thre_threshold).float()  # 形状: (num_thresholds, channels, ...)
                neg_false_spikes = (mem.unsqueeze(0) <= -thre_threshold).float() * (cumulative_out.unsqueeze(0) >= thre_shaped).float()
                
                pos_true = pos_false_spikes.clone()
                pos_true[1:] -= pos_false_spikes[:-1]
                neg_true = neg_false_spikes.clone()
                neg_true[1:] -= neg_false_spikes[:-1]
                
                spike = torch.sum(thre_shaped * pos_true, dim=0) - torch.sum(thre_shaped * neg_true, dim=0)
                mem = mem - spike
                cumulative_out = cumulative_out + spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)  # 形状: (T, channels, ...)
            x = self.merge(x)
            return x
        else:
            if self.T==0:
                self.mem = torch.zeros_like(x)
                time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
                num_additional_dims = x.dim()
                for _ in range(num_additional_dims):
                    time_steps = time_steps.unsqueeze(-1)
                self.thre_shaped = self.thresh.unsqueeze(0) / time_steps  # 形状: (num_thresholds, 1, ...)
                self.cumulative_out = torch.zeros_like(x)
            self.mem = self.mem + x
            pos_false_spikes = (self.mem.unsqueeze(0) >= (self.thre_shaped*0.75)).float()
            neg_false_spikes = (self.mem.unsqueeze(0) <= -(self.thre_shaped*0.75)).float() * (self.cumulative_out.unsqueeze(0) >= self.thre_shaped).float()
            
            pos_true = pos_false_spikes.clone()
            pos_true[1:] -= pos_false_spikes[:-1]
            neg_true = neg_false_spikes.clone()
            neg_true[1:] -= neg_false_spikes[:-1]
            
            spike = torch.sum(self.thre_shaped * pos_true, dim=0) - torch.sum(self.thre_shaped * neg_true, dim=0)
            self.mem = self.mem - spike
            self.cumulative_out = self.cumulative_out + spike
            self.T = self.T + 1
            return spike
        
class MTH_diff(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        super(MTH_diff, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T+1)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        self.num_thresholds = num_thresholds
        if step_mode=='s':
            self.T = -1
        else:
            self.T = T
    def reset(self):
        if self.step_mode == 's':
            self.T = -1
    def forward(self, x):
        if self.step_mode == 'm':
            time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
            num_additional_dims = x.dim()
            for _ in range(num_additional_dims):
                time_steps = time_steps.unsqueeze(-1)
            thre_shaped = self.thresh.unsqueeze(0) / time_steps  # 形状: (num_thresholds, 1, ...)
            thre_threshold = thre_shaped * 0.75  # 等同于 thre_shaped * 3/4
            x = self.expand(x)
            mem = torch.zeros_like(x[0])
            spike_pot = []
            spike_pot.append(torch.zeros_like(x[0]))
            cumulative_out = torch.zeros_like(x[0])
            bias = x[0].clone()
            exp_in = x[0].clone()
            exp_out = torch.zeros_like(x[0])
            
            for t in range(self.T):
                
                mem = mem + x[t+1] - bias + exp_in - exp_out # 更新膜电位
                cumulative_out = cumulative_out + exp_out
                pos_false_spikes = (mem.unsqueeze(0) >= thre_threshold).float()  # 形状: (num_thresholds, channels, ...)
                neg_false_spikes = (mem.unsqueeze(0) <= -thre_threshold).float() * (cumulative_out.unsqueeze(0) >= thre_shaped).float()
                
                pos_true = pos_false_spikes.clone()
                pos_true[1:] -= pos_false_spikes[:-1]
                neg_true = neg_false_spikes.clone()
                neg_true[1:] -= neg_false_spikes[:-1]
                
                spike = torch.sum(thre_shaped * pos_true, dim=0) - torch.sum(thre_shaped * neg_true, dim=0)
                # spike = spike * (spike.abs()/(t+1)>=exp_out.abs()/64).float()
                
                mem = mem - spike
                cumulative_out = cumulative_out + spike
                
                exp_in = exp_in + (x[t+1] - bias)/(t+1)
                exp_out = exp_out + spike/(t+1)
                
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)  # 形状: (T, channels, ...)
            x = self.merge(x)
            return x
        else:
            # print(x.shape,self.thresh.shape,self.thresh)
            if self.T==-1:
                self.bias = x.clone().detach()
                # self.exp_in = torch.zeros_like(x)
                self.exp_in = x.clone().detach()
                self.exp_out = torch.zeros_like(x)
                self.T = self.T+1
                return torch.zeros_like(x)
            if self.T==0:
                self.mem = torch.zeros_like(x)
                time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
                num_additional_dims = x.dim()
                for _ in range(num_additional_dims):
                    time_steps = time_steps.unsqueeze(-1)
                self.thre_shaped = self.thresh.unsqueeze(0) / time_steps  # 形状: (num_thresholds, 1, ...)
                self.cumulative_out = torch.zeros_like(x)
                
            self.mem = self.mem + x - self.bias + self.exp_in - self.exp_out
            self.cumulative_out = self.cumulative_out + self.exp_out
            
            pos_false_spikes = (self.mem.unsqueeze(0) >= (self.thre_shaped*0.75)).float()
            neg_false_spikes = (self.mem.unsqueeze(0) <= -(self.thre_shaped*0.75)).float() * (self.cumulative_out.unsqueeze(0) >= self.thre_shaped).float()
            
            pos_true = pos_false_spikes.clone()
            pos_true[1:] -= pos_false_spikes[:-1]
            neg_true = neg_false_spikes.clone()
            neg_true[1:] -= neg_false_spikes[:-1]
            
            spike = torch.sum(self.thre_shaped * pos_true, dim=0) - torch.sum(self.thre_shaped * neg_true, dim=0)
            self.mem = self.mem - spike
            self.cumulative_out = self.cumulative_out + spike
            
            self.exp_in = self.exp_in + (x - self.bias)/(self.T+1)
            self.exp_out = self.exp_out + spike/(self.T+1)
            
            self.T = self.T + 1
            return spike

class pos_H_MT_version_1(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        # thresh，实际为lambda_min，便于直接套用replace_relu_by_MTH函数
        super(pos_H_MT_version_1, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.lambda_min = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        self.num_thresholds = num_thresholds
        self.max_bit_form = int(2**num_thresholds-1)
        self.a = int(2**num_thresholds-1)
        self.b = int(-2**num_thresholds+1)
        if step_mode=='s':
            self.T = 0
        else:
            self.T = T
    def reset(self):
        if self.step_mode == 's':
            self.T = 0
    def forward(self, x):
        # return torch.clamp(x,-self.thresh,self.thresh)
        if self.step_mode == 'm':
            time_steps = 2 ** torch.arange(self.num_thresholds-1,-1,-1,device=x.device)
            num_additional_dims = x.dim()
            for _ in range(num_additional_dims):
                time_steps = time_steps.unsqueeze(-1)
            thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  # 形状: (num_thresholds, 1, ...)
            thre_threshold = thre_shaped
            x = self.expand(x)
            pos_mem = torch.ones_like(x[0])*self.lambda_min/2
            neg_mem = torch.ones_like(x[0])*self.lambda_min/2
            spike_pot = []
            for t in range(self.T):
                pos_mem = pos_mem + x[t]  # 更新膜电位
                pos_spike_pot=[]
                for i in range(self.num_thresholds):
                    pos_spike_i = (pos_mem>=thre_threshold[i]).float()
                    pos_spike_pot.append(pos_spike_i)
                    pos_mem-=pos_spike_i*thre_threshold[i]
                
                pos_spikes=torch.sum(thre_shaped * torch.stack(pos_spike_pot), dim=0)
                
                spike = pos_spikes

                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)  # 形状: (T, channels, ...)
            x = self.merge(x)
            return x
        else:
            if self.T==0:
                self.pos_mem = torch.ones_like(x[0])*self.lambda_min/2
                self.neg_mem = torch.ones_like(x[0])*self.lambda_min/2
                time_steps = 2 ** torch.flip(torch.arange(self.num_thresholds, device=x.device),dim=0)
                num_additional_dims = x.dim()
                for _ in range(num_additional_dims):
                    time_steps = time_steps.unsqueeze(-1)
                self.thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  # 形状: (num_thresholds, 1, ...)
            pos_mem = pos_mem + x[t]  # 更新膜电位
            pos_spike_pot=[]
            for i in range(self.num_thresholds):
                pos_spike_i = (pos_mem>=thre_threshold[i]).float()
                neg_spike_i = (neg_mem>=thre_threshold[i]).float()
                pos_spike_pot.append(pos_spike_i)
                pos_mem-=pos_spike_i*thre_threshold[i]
                neg_mem-=neg_spike_i*thre_threshold[i]
                
            pos_spikes=torch.sum(thre_shaped * torch.stack(pos_spike_pot), dim=0)
                
            spike = pos_spikes

            self.T = self.T + 1
            return spike
        

class pos_H_MT_version_2(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        # thresh，实际为lambda_min，便于直接套用replace_relu_by_MTH函数
        super(pos_H_MT_version_2, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.lambda_min = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        self.num_thresholds = num_thresholds
        self.max_bit_form = int(1<<num_thresholds-1)
        if step_mode=='s':
            self.T = 0
        else:
            self.T = T
    def reset(self):
        if self.step_mode == 's':
            self.T = 0
    def threshold_bits_tensor(self,x_tensor, theta, n):
        device = x_tensor.device
        dtype = x_tensor.dtype
        
        # 计算缩放因子并转换为整数张量
        scale = 2 ** (n - 1)
        max_k = (1 << n) - 1  # 最大允许值
        
        # 缩放并截断到[0, 2^n-1]
        k = (x_tensor * scale / theta).long().clamp(min=0, max=max_k)
        
        # 生成二进制位掩码 (n位，高位在前)
        bits_mask = 1 << torch.arange(n-1, -1, -1, device=device)  # 形如 [4,2,1] 当n=3
        num_additional_dims = x_tensor.dim()
        for _ in range(num_additional_dims):
            bits_mask = bits_mask.unsqueeze(-1)
        # 通过广播计算每个位的值 (形状: [..., n])
        bits = (k.unsqueeze(0) & bits_mask).ne(0).to(dtype=torch.int32)
        return bits  # 返回形状为 [n(num_threshold),原始维度] 的张量
    
    def forward(self, x):
        # return torch.clamp(x,-self.thresh,self.thresh)
        if self.step_mode == 'm':
            time_steps = 2 ** torch.arange(self.num_thresholds-1,-1,-1,device=x.device)
            num_additional_dims = x.dim()
            for _ in range(num_additional_dims):
                time_steps = time_steps.unsqueeze(-1)
            thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  # 形状: (num_thresholds, 1, ...)
            thre_threshold = thre_shaped
            x = self.expand(x)
            pos_mem = torch.ones_like(x[0])*self.lambda_min/2
            neg_mem = torch.ones_like(x[0])*self.lambda_min/2
            spike_pot = []
            for t in range(self.T):
                pos_mem = pos_mem + x[t]  # 更新膜电位
                
                pos_spikes=torch.sum(thre_shaped*self.threshold_bits_tensor(pos_mem,self.lambda_min,self.num_thresholds),dim=0)
                
                spike = pos_spikes
                pos_mem -= spike


                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)  # 形状: (T, channels, ...)
            x = self.merge(x)
            return x
        else:
            if self.T==0:
                self.pos_mem = torch.ones_like(x[0])*self.lambda_min/2
                self.neg_mem = torch.ones_like(x[0])*self.lambda_min/2
                time_steps = 2 ** torch.flip(torch.arange(self.num_thresholds, device=x.device),dim=0)
                num_additional_dims = x.dim()
                for _ in range(num_additional_dims):
                    time_steps = time_steps.unsqueeze(-1)
                self.thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  # 形状: (num_thresholds, 1, ...)
            pos_mem = pos_mem + x[t]  # 更新膜电位
            
            pos_spikes=torch.sum(thre_shaped*self.threshold_bits_tensor(pos_mem,self.lambda_min,self.num_thresholds),dim=0)
                
            spike = pos_spikes
            pos_mem -= spike
            self.T = self.T + 1
            return spike
        


class pos_H_MT_storage_version_1(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        # thresh，实际为lambda_min，便于直接套用replace_relu_by_MTH函数
        super(pos_H_MT_storage_version_1, self).__init__()
        self.step_mode = step_mode
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        self.act = ZIF.apply
        self.lambda_min = nn.Parameter(thresh.clone().detach(), requires_grad=True)
        self.num_thresholds = num_thresholds
        self.max_bit_form = int(2**num_thresholds-1)
        self.a = int(2**num_thresholds-1)
        self.b = int(-2**num_thresholds+1)
        if step_mode=='s':
            self.T = 0
        else:
            self.T = T
    def reset(self):
        if self.step_mode == 's':
            self.T = 0
    def forward(self, x):
        # return torch.clamp(x,-self.thresh,self.thresh)
        if self.step_mode == 'm':
            time_steps = 2 ** torch.arange(self.num_thresholds-1,-1,-1,device=x.device)
            num_additional_dims = x.dim()
            for _ in range(num_additional_dims):
                time_steps = time_steps.unsqueeze(-1)
            thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  # 形状: (num_thresholds, 1, ...)
            thre_threshold = thre_shaped
            x = self.expand(x)
            pos_mem = torch.ones_like(x[0])*self.lambda_min/2
            neg_mem = torch.ones_like(x[0])*self.lambda_min/2
            judging_mem = torch.zeros_like(x[0]) # 决定负脉冲是否进行发放
            storage_mem = torch.zero_like(x[0]) # 存储未释放的负脉冲
            spike_pot = []
            for t in range(self.T):
                pos_mem = pos_mem + x[t]  # 更新膜电位
                neg_mem = neg_mem - x[t]
                pos_spike_pot=[]
                neg_spike_pot=[]
                for i in range(self.num_thresholds):
                    pos_spike_i = (pos_mem>=thre_threshold[i]).float()
                    neg_spike_i = (neg_mem>=thre_threshold[i]).float()
                    pos_spike_pot.append(pos_spike_i)
                    neg_spike_pot.append(neg_spike_i)
                    pos_mem-=pos_spike_i*thre_threshold[i]
                    neg_mem-=neg_spike_i*thre_threshold[i]
                
                pos_spikes=torch.sum(thre_shaped * torch.stack(pos_spike_pot), dim=0)
                neg_spikes=torch.sum(thre_shaped * torch.stack(neg_spike_pot), dim=0)

                # 维持正负神经元膜电位相加为lambda_min的性质，同一时刻正负神经元只会有其一能够发放脉冲
                pos_mem += neg_spikes
                neg_mem += pos_spikes
                
                # 将(未发放)的负脉冲存储到storage_mem里
                storage_mem += neg_spikes
                # 将正负整合的脉冲存储到judging_mem里
                judging_mem += pos_spikes-neg_spikes
                
                # 很朴素的想法是，当juding_mem有大于0的，说明正的释放量超越了负的，可以把存储的负的释放出来进行修正了
                # 而且不希望一次把负的全释放，导致这一个时刻的输出传到下一层还是一个极端的负的
                judging_mask = judging_mem >= 0
                storage_mem_specific = storage_mem[judging_mask].clone()
                storage_spikes_pot_specific=[]
                for i in range(self.num_thresholds):
                    storage_spike_specific_i = (storage_mem_specific>=thre_threshold[i]).float()
                    storage_spikes_pot_specific.append(storage_spike_specific_i)
                    storage_mem_specific-=storage_spike_specific_i*thre_threshold[i]
                storage_spikes_specific=torch.sum(thre_shaped * torch.stack(storage_spikes_pot_specific), dim=0)
                storage_mem[judging_mask] = storage_mem_specific
                storage_spikes_specific = torch.zeros_like(storage_mem)
                storage_spikes_specific[judging_mask] =  storage_spikes_specific

                spike = pos_spikes - storage_mem_specific
                # 但是这样做有一个隐患，如果从最开始是一个极其负的输入，后面都是正的，总的算下来加和平均还是负的，
                # 因为正的怎么累加也不会超越负的，所以还是t=1时不会释放本来该有的负脉冲，但是后面正脉冲还是一样发放
                # 所以不如改换思路，storage根据每次的pos_mem直接发放算了，但是这个还需要再考虑，看看适不适合spike形式

                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)  # 形状: (T, channels, ...)
            x = self.merge(x)
            return x
        else:
            if self.T==0:
                self.pos_mem = torch.ones_like(x[0])*self.lambda_min/2
                self.neg_mem = torch.ones_like(x[0])*self.lambda_min/2
                self.judging_mem = torch.zeros_like(x[0]) # 决定负脉冲是否进行发放
                self.storage_mem = torch.zero_like(x[0]) # 存储未释放的负脉冲
                time_steps = 2 ** torch.flip(torch.arange(self.num_thresholds, device=x.device),dim=0)
                num_additional_dims = x.dim()
                for _ in range(num_additional_dims):
                    time_steps = time_steps.unsqueeze(-1)
                self.thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  # 形状: (num_thresholds, 1, ...)
            pos_mem = pos_mem + x[t]  # 更新膜电位
            pos_spike_pot=[]
            neg_spike_pot=[]
            for i in range(self.num_thresholds):
                pos_spike_i = (pos_mem>=thre_threshold[i]).float()
                neg_spike_i = (neg_mem>=thre_threshold[i]).float()
                pos_spike_pot.append(pos_spike_i)
                neg_spike_pot.append(neg_spike_i)
                pos_mem-=pos_spike_i*thre_threshold[i]
                neg_mem-=neg_spike_i*thre_threshold[i]
                
            pos_spikes=torch.sum(thre_shaped * torch.stack(pos_spike_pot), dim=0)
            neg_spikes=torch.sum(thre_shaped * torch.stack(neg_spike_pot), dim=0)
                
            # 维持正负神经元膜电位相加为lambda_min的性质，同一时刻正负神经元只会有其一能够发放脉冲
            pos_mem += neg_spikes
            neg_mem += pos_spikes
                
            # 将(未发放)的负脉冲存储到storage_mem里
            storage_mem += neg_spikes
            # 将正负整合的脉冲存储到judging_mem里
            judging_mem += pos_spikes-neg_spikes
                
            # 很朴素的想法是，当juding_mem有大于0的，说明正的释放量超越了负的，可以把存储的负的释放出来进行修正了
            # 而且不希望一次把负的全释放，导致这一个时刻的输出传到下一层还是一个极端的负的
            judging_mask = judging_mem >= 0
            storage_mem_specific = storage_mem[judging_mask].clone()
            storage_spikes_pot_specific=[]
            for i in range(self.num_thresholds):
                storage_spike_specific_i = (storage_mem_specific>=thre_threshold[i]).float()
                storage_spikes_pot_specific.append(storage_spike_specific_i)
                storage_mem_specific-=storage_spike_specific_i*thre_threshold[i]
            storage_spikes_specific=torch.sum(thre_shaped * torch.stack(storage_spikes_pot_specific), dim=0)
            storage_mem[judging_mask] = storage_mem_specific
            storage_spikes_specific = torch.zeros_like(storage_mem)
            storage_spikes_specific[judging_mask] =  storage_spikes_specific

            spike = pos_spikes - storage_mem_specific

            self.T = self.T + 1
            return spike