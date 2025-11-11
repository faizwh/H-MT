import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import MergeTemporalDim, ExpandTemporalDim

class ZIF(torch.autograd.Function):
    # for judging whether to send out spike
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
        
class IF_with_neg_line(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s'):
        super(IF_with_neg_line, self).__init__()
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
                spike_neg = self.act(-mem) *  thre
                spike = spike-spike_neg
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0) #按照dim=0进行堆叠
            x = self.merge(x)
            return x
        else:# self.step_mode == 's':
            if self.T==0:
                self.mem = 0.5 * self.thresh
                
            self.mem = self.mem + x
            spike = self.act(self.mem - self.thresh) * self.thresh
            spike_neg = self.act(-self.mem)  * self.thresh
            spike = spike - spike_neg
            
            self.mem = self.mem - spike
            self.T = self.T + 1
            return spike

#这一块是time differential的neuron
class IF_diff_line(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s'):
        super(IF_diff_line, self).__init__()
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
            spike_pot.append(torch.zeors_like(x[0]))
            exp_in = torch.zeors_like(x[0])
            exp_out = torch.zeors_like(x[0])
            for t in range(self.T):
                mem = mem + x[t+1] + exp_in - exp_out
                spike = self.act(mem - thre) * thre
                spike_neg = self.act(-mem) * thre
                spike = spike-spike_neg
                
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
                self.exp_in = torch.zeros_like(x)
                self.exp_out = torch.zeros_like(x)
                self.T = self.T+1
                return torch.zeros_like(x)
            if self.T==0:
                self.mem = 0.5 * self.thresh
            self.mem = self.mem + x + self.exp_in - self.exp_out
            spike = self.act(self.mem - self.thresh) * self.thresh
            spike_neg = self.act(-self.mem) * self.thresh
            spike = spike-spike_neg
            
            self.mem = self.mem - spike
            
            self.exp_in = self.exp_in + (x - self.bias)/(self.T+1)
            self.exp_out = self.exp_out + spike/(self.T+1)
            
            self.T = self.T + 1
            return spike
        
class MTH_with_neg_line(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        super(MTH_with_neg_line, self).__init__()
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
        # return torch.clamp(x,-self.thresh,self.thresh)
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
                pos_false_spikes = (mem.unsqueeze(0) >= thre_threshold).float()# 形状: (num_thresholds, channels, ...)
                neg_false_spikes = (mem.unsqueeze(0) <= -thre_threshold).float()
                
                pos_true = pos_false_spikes.clone()
                pos_true[1:] -= pos_false_spikes[:-1]#筛出唯一想要的true #当然这里进行筛除也是历史遗留的判断问题；
                # 相当于pos_true[i]-pos_true[i-1]，当且仅当pos_true[i]是第一个为true的时候，才会有pos_true[i]-pos_true[i-1]=1
                neg_true = neg_false_spikes.clone()
                neg_true[1:] -= neg_false_spikes[:-1]#同上
                
                spike = torch.sum(thre_shaped * pos_true, dim=0) - torch.sum(thre_shaped * neg_true, dim=0)
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
            neg_false_spikes = (self.mem.unsqueeze(0) <= -(self.thre_shaped*0.75)).float()
            
            pos_true = pos_false_spikes.clone()
            pos_true[1:] -= pos_false_spikes[:-1]
            neg_true = neg_false_spikes.clone()
            neg_true[1:] -= neg_false_spikes[:-1]
            
            spike = torch.sum(self.thre_shaped * pos_true, dim=0) - torch.sum(self.thre_shaped * neg_true, dim=0)
            self.mem = self.mem - spike
            self.T = self.T + 1
            return spike
        
# class MTH_diff_line(nn.Module):
#     def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
#         super(MTH_diff_line, self).__init__()
#         self.step_mode = step_mode
#         self.expand = ExpandTemporalDim(T+1)
#         self.merge = MergeTemporalDim()
#         self.act = ZIF.apply
#         self.thresh = nn.Parameter(thresh.clone().detach(), requires_grad=True)
#         self.num_thresholds = num_thresholds
#         if step_mode=='s':
#             self.T = -1
#         else:
#             self.T = T
#     def reset(self):
#         if self.step_mode == 's':
#             self.T = -1
    
#     def forward(self, x):
#         if self.step_mode == 'm':
#             time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
#             num_additional_dims = x.dim()
#             for _ in range(num_additional_dims):
#                 time_steps = time_steps.unsqueeze(-1)
#             thre_shaped = self.thresh.unsqueeze(0) / time_steps  # 形状: (num_thresholds, 1, ...)
#             thre_threshold = thre_shaped * 0.75  # 等同于 thre_shaped * 3/4
#             x = self.expand(x)
#             mem = torch.zeros_like(x[0])
#             spike_pot = []
#             spike_pot.append(torch.zeros_like(x[0]))
#             bias = x[0].clone()
#             exp_in = x[0].clone()
#             exp_out = torch.zeros_like(x[0])
            
#             for t in range(self.T):
                
#                 mem = mem + x[t+1] - bias + exp_in - exp_out # 更新膜电位
#                 pos_false_spikes = (mem.unsqueeze(0) >= thre_threshold).float()  # 形状: (num_thresholds, channels, ...)
#                 neg_false_spikes = (mem.unsqueeze(0) <= -thre_threshold).float()
                
#                 pos_true = pos_false_spikes.clone()
#                 pos_true[1:] -= pos_false_spikes[:-1]
#                 neg_true = neg_false_spikes.clone()
#                 neg_true[1:] -= neg_false_spikes[:-1]
                
#                 spike = torch.sum(thre_shaped * pos_true, dim=0) - torch.sum(thre_shaped * neg_true, dim=0)
#                 # spike = spike * (spike.abs()/(t+1)>=exp_out.abs()/16).float()
                
#                 mem = mem - spike
                
#                 exp_in = exp_in + (x[t+1] - bias)/(t+1)
#                 exp_out = exp_out + spike/(t+1)
                
#                 spike_pot.append(spike)
#             x = torch.stack(spike_pot, dim=0)  # 形状: (T, channels, ...)
#             x = self.merge(x)
#             return x
#         else:
#             if self.T==-1:
#                 self.bias = x.clone().detach()
#                 self.exp_in = x.clone().detach()
#                 self.exp_out = torch.zeros_like(x)
#                 self.T = self.T+1
#                 return torch.zeros_like(x)
#             if self.T==0:
#                 self.mem = torch.zeros_like(x)
#                 time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
#                 num_additional_dims = x.dim()
#                 for _ in range(num_additional_dims):
#                     time_steps = time_steps.unsqueeze(-1)
#                 self.thre_shaped = self.thresh.unsqueeze(0) / time_steps  # 形状: (num_thresholds, 1, ...)
                
#             self.mem = self.mem + x - self.bias + self.exp_in - self.exp_out
            
#             pos_false_spikes = (self.mem.unsqueeze(0) >= (self.thre_shaped*0.75)).float()
#             neg_false_spikes = (self.mem.unsqueeze(0) <= -(self.thre_shaped*0.75)).float()
            
#             pos_true = pos_false_spikes.clone()
#             pos_true[1:] -= pos_false_spikes[:-1]
#             neg_true = neg_false_spikes.clone()
#             neg_true[1:] -= neg_false_spikes[:-1]
            
#             spike = torch.sum(self.thre_shaped * pos_true, dim=0) - torch.sum(self.thre_shaped * neg_true, dim=0)
#             self.mem = self.mem - spike
            
#             self.exp_in = self.exp_in + (x - self.bias)/(self.T+1)
#             self.exp_out = self.exp_out + spike/(self.T+1)
            
#             self.T = self.T + 1
#             return spike

@torch.jit.script
def ms_forward_core_jit_(mem: torch.Tensor, x: torch.Tensor, bias: torch.Tensor, exp_in: torch.Tensor, exp_out: torch.Tensor, thre_threshold: torch.Tensor, thre_shaped: torch.Tensor, t: int):
    mem = mem + x - bias + exp_in - exp_out  # 更新膜电位
    pos_false_spikes = (mem.unsqueeze(0) >= thre_threshold).float()  # 形状: (num_thresholds, channels, ...)
    neg_false_spikes = (mem.unsqueeze(0) <= -thre_threshold).float()

    pos_true = pos_false_spikes.clone()
    pos_true[1:] -= pos_false_spikes[:-1]
    neg_true = neg_false_spikes.clone()
    neg_true[1:] -= neg_false_spikes[:-1]

    spike = torch.sum(thre_shaped * pos_true, dim=0) - torch.sum(thre_shaped * neg_true, dim=0)
    # spike = spike * (spike.abs()/(t+1)>=exp_out.abs()/16).float()

    mem = mem - spike

    exp_in = exp_in + (x - bias) / (t + 1)
    exp_out = exp_out + spike / (t + 1)
    return mem, exp_in, exp_out, spike


def ms_forward_core_jit(mem: torch.Tensor, x: torch.Tensor, bias: torch.Tensor, exp_in: torch.Tensor, exp_out: torch.Tensor, thre_threshold: torch.Tensor, thre_shaped: torch.Tensor, T: int):
    spike_pot = []
    for t in range(T):
        mem, exp_in, exp_out, spike = ms_forward_core_jit_(mem, x[t+1], bias, exp_in, exp_out, thre_threshold, thre_shaped, t)
        spike_pot.append(spike)
    return torch.stack(spike_pot)


class MTH_diff_line(nn.Module):        
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        super(MTH_diff_line, self).__init__()
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
            # if hasattr(self, 'time_steps') and self.time_steps.numel() == self.num_thresholds:
            #     thre_threshold = self.thre_threshold
            # else:
            thre_threshold = self.thresh
            time_steps = 2 ** torch.arange(self.num_thresholds, device=x.device)
            num_additional_dims = x.dim()
            for _ in range(num_additional_dims):
                time_steps = time_steps.unsqueeze(-1)

            self.time_steps = time_steps

            thre_shaped = self.thresh.unsqueeze(0) / time_steps  # 形状: (num_thresholds, 1, ...)
            thre_threshold = thre_shaped * 0.75  # 等同于 thre_shaped * 3/4

            self.thre_threshold = thre_threshold



            x = self.expand(x)
            bias = x[0].clone()
            exp_in = x[0].clone()

            if hasattr(self, 'mem') and self.mem.shape == x.shape[1:]:
                mem = self.mem
                exp_out = self.exp_out
                spike_pot0 = self.spike_pot0
            else:
                mem = torch.zeros_like(x[0])
                exp_out = torch.zeros_like(x[0])
                spike_pot0 = torch.zeros_like(x[0])
                self.mem = mem
                self.exp_out = exp_out
                self.spike_pot0 = spike_pot0



            spike_pot = ms_forward_core_jit(mem, x, bias, exp_in, exp_out, thre_threshold, thre_shaped, self.T)

            x = torch.cat((spike_pot0.unsqueeze(0), spike_pot), dim=0)  # 形状: (T, channels, ...)
            x = self.merge(x)
            return x


        else:
            if self.T==-1:
                self.bias = x.clone().detach()
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
                
            self.mem = self.mem + x - self.bias + self.exp_in - self.exp_out
            
            pos_false_spikes = (self.mem.unsqueeze(0) >= (self.thre_shaped*0.75)).float()
            neg_false_spikes = (self.mem.unsqueeze(0) <= -(self.thre_shaped*0.75)).float()
            
            pos_true = pos_false_spikes.clone()
            pos_true[1:] -= pos_false_spikes[:-1]
            neg_true = neg_false_spikes.clone()
            neg_true[1:] -= neg_false_spikes[:-1]
            
            spike = torch.sum(self.thre_shaped * pos_true, dim=0) - torch.sum(self.thre_shaped * neg_true, dim=0)
            self.mem = self.mem - spike
            
            self.exp_in = self.exp_in + (x - self.bias)/(self.T+1)
            self.exp_out = self.exp_out + spike/(self.T+1)
            
            self.T = self.T + 1
            return spike
        
# H-MT 尝试
class H_MT_version_1(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        # thresh, actually lambda_min
        super(H_MT_version_1, self).__init__()
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
            thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  # (num_thresholds, 1, ...)
            thre_threshold = thre_shaped
            x = self.expand(x)
            pos_mem = torch.ones_like(x[0])*self.lambda_min/2
            neg_mem = torch.ones_like(x[0])*self.lambda_min/2
            spike_pot = []
            for t in range(self.T):
                pos_mem = pos_mem + x[t]  
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
                
                spike = pos_spikes-neg_spikes
                # the communication
                pos_mem += neg_spikes
                neg_mem += pos_spikes

                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)  #(T, channels, ...)
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
                self.thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  #(num_thresholds, 1, ...)
            pos_mem = pos_mem + x[t]  
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
                
            spike = pos_spikes-neg_spikes
            # the communication
            pos_mem += neg_spikes
            neg_mem += pos_spikes
            self.T = self.T + 1
            return spike



class H_MT_version_2(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        # thresh，actually lambda_min
        super(H_MT_version_2, self).__init__()
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
        
        scale = 2 ** (n - 1)
        max_k = (1 << n) - 1
        
        k = (x_tensor * scale / theta).long().clamp(min=0, max=max_k)
        
        bits_mask = 1 << torch.arange(n-1, -1, -1, device=device)
        num_additional_dims = x_tensor.dim()
        for _ in range(num_additional_dims):
            bits_mask = bits_mask.unsqueeze(-1)
        
        bits = (k.unsqueeze(0) & bits_mask).ne(0).to(dtype=torch.int32)
        return bits
    
    def forward(self, x):
        # return torch.clamp(x,-self.thresh,self.thresh)
        if self.step_mode == 'm':
            time_steps = 2 ** torch.arange(self.num_thresholds-1,-1,-1,device=x.device)
            num_additional_dims = x.dim()
            for _ in range(num_additional_dims):
                time_steps = time_steps.unsqueeze(-1)
            thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  #(num_thresholds, 1, ...)
            thre_threshold = thre_shaped
            x = self.expand(x)
            pos_mem = torch.ones_like(x[0])*self.lambda_min/2
            neg_mem = torch.ones_like(x[0])*self.lambda_min/2
            spike_pot = []
            for t in range(self.T):
                pos_mem = pos_mem + x[t]
                neg_mem = neg_mem - x[t]
                
                pos_spikes=torch.sum(thre_shaped*self.threshold_bits_tensor(pos_mem,self.lambda_min,self.num_thresholds),dim=0)
                neg_spikes=torch.sum(thre_shaped*self.threshold_bits_tensor(neg_mem,self.lambda_min,self.num_thresholds),dim=0)
                
                spike = pos_spikes-neg_spikes
                pos_mem -= spike
                neg_mem += spike

                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
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
                self.thre_shaped = self.lambda_min.unsqueeze(0) * time_steps
            pos_mem = pos_mem + x[t]
            neg_mem = neg_mem - x[t]
            
            pos_spikes=torch.sum(thre_shaped*self.threshold_bits_tensor(pos_mem,self.lambda_min,self.num_thresholds),dim=0)
            neg_spikes=torch.sum(thre_shaped*self.threshold_bits_tensor(neg_mem,self.lambda_min,self.num_thresholds),dim=0)
                
            spike = pos_spikes-neg_spikes
            pos_mem -= spike
            neg_mem += spike
            self.T = self.T + 1
            return spike
        
class H_MT_version_IF(nn.Module):
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        # thresh, actually lambda_min
        super(H_MT_version_IF, self).__init__()
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
            # 形状 (num_threshods,[shape of input])
            # so there are num_thresholds neurons assembled together 
            pos_mem = (torch.ones_like(x[0])*self.lambda_min/2).unsqueeze(0).repeat((self.num_thresholds,)+(1,)*x[0].dim())
            neg_mem = (torch.ones_like(x[0])*self.lambda_min/2).unsqueeze(0).repeat((self.num_thresholds,)+(1,)*x[0].dim())
            spike_pot = []
            for t in range(self.T):
                pos_mem = pos_mem + x[t]
                neg_mem = neg_mem - x[t]
                pos_spike_pot=[]
                neg_spike_pot=[]
                for i in range(self.num_thresholds):
                    pos_spike_i = (pos_mem[i]>=thre_threshold[i]).float()
                    pos_spike_pot.append(pos_spike_i)
                    pos_mem-=pos_spike_i*thre_threshold[i]
                    neg_spike_i = (neg_mem[i]>=thre_threshold[i]).float()
                    neg_spike_pot.append(neg_spike_i)
                    neg_mem-=neg_spike_i*thre_threshold[i]
                
                pos_spikes=torch.sum(thre_shaped * torch.stack(pos_spike_pot), dim=0)
                neg_spikes=torch.sum(thre_shaped * torch.stack(neg_spike_pot), dim=0)

                spike = pos_spikes - neg_spikes
                pos_mem += neg_spikes
                neg_mem += pos_spikes

                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)  # (T, channels, ...)
            x = self.merge(x)
            return x
        else:
            if self.T==0:
                # (num_thresholds,[shape of input])
                # so there are num_thresholds neurons assembled together 
                self.pos_mem = (torch.ones_like(x[0])*self.lambda_min/2).unsqueeze(0).repeat((self.num_thresholds,)+(1,)*x[0].dim())
                self.neg_mem = torch.ones_like(x[0])*self.lambda_min/2
                time_steps = 2 ** torch.flip(torch.arange(self.num_thresholds, device=x.device),dim=0)
                num_additional_dims = x.dim()
                for _ in range(num_additional_dims):
                    time_steps = time_steps.unsqueeze(-1)
                self.thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  # (num_thresholds, 1, ...)
            self.pos_mem = self.pos_mem + x[t]
            pos_spike_pot=[]
            for i in range(self.num_thresholds):
                pos_spike_i = (self.pos_mem[i]>=thre_threshold[i]).float()
                neg_spike_i = (self.neg_mem[i]>=thre_threshold[i]).float()
                pos_spike_pot.append(pos_spike_i)
                self.pos_mem-=pos_spike_i*thre_threshold[i]
                self.neg_mem-=neg_spike_i*thre_threshold[i]
                
            pos_spikes=torch.sum(thre_shaped * torch.stack(pos_spike_pot), dim=0)
                
            spike = pos_spikes

            self.T = self.T + 1
            return spike
        
class H_MT_version_1_no_commu(nn.Module):
    # the version without communication
    def __init__(self, T=0, thresh=1.0, step_mode='s',num_thresholds=8):
        # thresh, actually lambda_min
        super(H_MT_version_1_no_commu, self).__init__()
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
            thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  #(num_thresholds, 1, ...)
            thre_threshold = thre_shaped
            x = self.expand(x)
            pos_mem = torch.ones_like(x[0])*self.lambda_min/2
            neg_mem = torch.ones_like(x[0])*self.lambda_min/2
            spike_pot = []
            for t in range(self.T):
                pos_mem = pos_mem + x[t]
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
                
                spike = pos_spikes-neg_spikes

                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)  # (T, channels, ...)
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
                self.thre_shaped = self.lambda_min.unsqueeze(0) * time_steps  # (num_thresholds, 1, ...)
            pos_mem = pos_mem + x[t] 
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
                
            spike = pos_spikes-neg_spikes
            self.T = self.T + 1
            return spike