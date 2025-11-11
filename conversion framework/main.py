import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from utils import seed_all, get_logger, get_modules
from datasets import datapool
from models import modelpool
from train_val_functions import valpool
from converter import Threshold_Getter,Converter
from forwards import forward_replace

def get_args():
    parser = argparse.ArgumentParser(description='Conversion Frame')

    # Model configuration
    parser.add_argument('--model_name', default='vgg16_bn', type=str, help='Model class name')
    parser.add_argument('--load_name', '-load', type=str, help='Path to the model state_dict file')
    parser.add_argument('--mode', choices=['test_ann', 'get_threshold', 'test_snn', 'train_snn'], default='test_ann', type=str, help='Mode of operation')
    parser.add_argument('--sop', action='store_true', help="whether to static sop")
    parser.add_argument('--save_name', '-save', default='checkpoint', type=str, help='Name for saving the model')

    # Threshold configuration
    parser.add_argument('--threshold_mode', '-thre', default='99.9%', type=str, help='Threshold mode')
    parser.add_argument('--threshold_level', default='layer', choices=['layer', 'channel', 'neuron'], type=str, help='Threshold level')
    parser.add_argument('--fx', action='store_true', help="Whether to use fx output graph")

    # Neuron conversion configuration
    parser.add_argument('--neuron_name', '-neuron', choices=['IF', 'IF_with_neg', 'IF_diff', 'IF_line','IF_diff_line'
                                                             'LIF', 'LIF_with_neg', 'LIF_diff',
                                                             'MTH', 'MTH_with_neg', 'MTH_diff', 'MTH_line','MTH_diff_line'], default='IF', type=str, 
                                                             help='Neuron model name-----for relu actually')
    
    parser.add_argument('--neuron_name_identity', '-neuron_iden', choices=["H_MT_version_1","H_MT_version_2"], 
                                                             default='H_MT_version_1', type=str, 
                                                             help='Neuron model name-----for identity actually')
    
    parser.add_argument('--identity_change', '-iden', choices=['True','False'], default='False', type=str, help='whether to change the identity modules')
    parser.add_argument('--tau', default=0.98, type=float, help='Parameter tau')
    parser.add_argument('--num_thresholds', default=3, type=int, help='num_thresholds')
    parser.add_argument('--step_mode', choices=['s', 'm'], default='s', type=str, help='Step_mode')
    parser.add_argument('--coding_type', '-coding', choices=['rate', 'leaky_rate', 'diff_rate', 'diff_leaky_rate'], default='rate', type=str, help='Coding type')
    parser.add_argument('--fuse', action='store_true', help="Whether to fuse")
    
    # Task configuration
    parser.add_argument('--task', choices=['classification','object_detection'], default='classification', type=str, help='Task type')
    
    # Dataset configuration
    parser.add_argument('--dataset', '-data', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('--dataset_path', default='../data', type=str, help='Dataset path')
    parser.add_argument('--batchsize', '-b', default=25, type=int, metavar='N', help='Batch size')

    # Device configuration
    parser.add_argument('--device', '-dev', default='0', type=str, help='CUDA device ID (default: 0)')
    # Device configuration only for imagenet
    # eg.torchrun --nproc_per_node=1 main.py --logger --dataset imagenet --batchsize 64 --distributed
    parser.add_argument('--distributed', action='store_true', help="Enable distributed (default: False)")
    
    # Logger configuration
    parser.add_argument('--logger', action='store_true', help="Enable logging (default: False)")
    parser.add_argument('--logger_path', type=str, default="logs/log.txt", help="Path to save the log file")

    # Training and Testing configuration
    parser.add_argument('--seed', default=2024, type=int, help='Random seed for training initialization')
    parser.add_argument('--time', '-T', type=int, default=0, help='SNN simulation time')

    # YAML configuration
    parser.add_argument('--config', default='configs/config.yaml', type=str, help="Path to the YAML configuration file")

    # Parse arguments
    args = parser.parse_args()

    # Load configuration from YAML if specified
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)  # 从文件中加载配置
        for key, value in config.items():
            setattr(args, key, value)  # 将配置文件中的键值对添加到 args 中

    
    # Set CUDA device
    # 设置新的环境变量，这是可行的
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    return args, device

def load_model_from_dict(model, model_path, device):
    state_dict = torch.load(os.path.join(model_path), map_location=torch.device('cpu'),weights_only=True)
    for model_key in ['model','module']:
        if model_key in state_dict:
            state_dict = state_dict[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def load_model_from_model(model, model_path, device):
    model = torch.load(model_path,weights_only=True)
    return model

def main():
    args, device = get_args()
    # Set the random seed
    seed_all(args.seed)
    #print(args.coding_type)
    logger = get_logger(args.logger,args.logger_path)
    # train_loader is None if testing
    train_loader, test_loader = datapool(args)
    model = modelpool(args)
    get_modules(111,model)
    # Perform training or testing based on args.mode
    if args.mode == 'test_ann':
        print("Test ANN Mode")
        model = load_model_from_dict(model, args.load_name, device)
        print(model)
        print("Successfully load ann state dict")
        model.to(device)
        model.eval()
        val = valpool(args)
        print(type(test_loader))
        val(model, test_loader, device, args)
    elif args.mode == 'get_threshold':
        print("Get Threshold for SNN Neuron Mode")
        model = load_model_from_dict(model, args.load_name, device)#加载模型
        model.to(device)
        model.eval()
        model = Converter.change_maxpool_before_relu(model)
        do_data_collection=True
        #Threshold_Getter的forward会进行replace操作，然后再过一遍数据集，是的，这个转化过程在replace完还要过一轮数据集进行数据的统计收集，确实很可怕了
        model_converter = Threshold_Getter(dataloader=test_loader, mode=args.threshold_mode, level=args.threshold_level, 
                                           device=device, momentum=0.1, output_fx=args.fx,do_data_collection=do_data_collection)#替换模块
        model_with_threshold = model_converter(model)#计算阈值

        Threshold_Getter.save_model(model=model_with_threshold, model_path=args.save_name, mode_fx=args.fx)# 保存模型/模型状态字典
        print("Successfully Save Model with Threshold")
    elif args.mode == 'test_snn':# 暂时不支持输入fx
        print("Test SNN Mode")
        do_data_collection=False
        model = Converter.change_maxpool_before_relu(model)
        model = Converter.replace_by_maxpool_neuron(model,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
        # 重新调整为带钩子的
        model = Threshold_Getter.replace_nonlinear_by_hook(model=model, momentum=0.1, mode=args.threshold_mode, level=args.threshold_level,
                                                           do_data_collection=do_data_collection)
        model = Threshold_Getter.replace_identity_by_hook(model=model, momentum=0.1, mode=args.threshold_mode, level=args.threshold_level,
                                                          do_data_collection=do_data_collection)
        # 然后把参数load进来
        model = load_model_from_dict(model, args.load_name, device)
        
        # 若使用max或%方法来选取阈值，则在前面Threshold_Getter的forward里已经完成数据收集了，直接就用scale就行了
        # 但是使用max或%方法，对HMT的设计需要是把最后更新计算出的Threshold除掉2**n-1，才是合适的
        if  args.threshold_mode[-1]=="%":
            if args.neuron_name.startswith('H_MT'):
                model = Threshold_Getter.get_scale_for_HMT(model, n=args.num_thresholds,change_mode="%")
            elif args.neuron_name.startswith('pos_H_MT'):
                model = Threshold_Getter.get_scale_for_HMT(model, n=args.num_thresholds,change_mode="%")
            else :
                model = Threshold_Getter.get_scale_for_HMT(model, n=1,change_mode="%")
        elif  args.threshold_mode=="max":
            if args.neuron_name.startswith('H_MT'):
                model = Threshold_Getter.get_scale_for_HMT(model, n=args.num_thresholds,change_mode="max")
            elif args.neuron_name.startswith('pos_H_MT'):
                model = Threshold_Getter.get_scale_for_HMT(model, n=args.num_thresholds,change_mode="max")
            else :
                model = Threshold_Getter.get_scale_for_HMT(model, n=1,change_mode="max")
        # 若使用迭代法，在这一步完成最后的最优阈值计算
        elif args.threshold_mode=="var":
            if args.neuron_name.startswith('MTH'):
                model = Threshold_Getter.get_scale_from_var(model, T=args.time*(2**args.num_thresholds),a=0,b=args.time*(2**args.num_thresholds))
            elif args.neuron_name.startswith('H_MT'):
                model = Threshold_Getter.get_scale_from_var(model, T=args.time,a=-args.time*(2**args.num_thresholds-1),b=args.time*(2**args.num_thresholds-1))
            elif args.neuron_name.startswith('pos_H_MT'):
                model = Threshold_Getter.get_scale_from_var(model, T=args.time,a=0,b=args.time*(2**args.num_thresholds-1))
                #model = Threshold_Getter.get_scale_from_var(model, T=args.time,a=0,b=args.time*(2**args.num_thresholds))
            else:
                model = Threshold_Getter.get_scale_from_var(model, T=args.time,a=0,b=args.time)
        # 在这一步完成了把ReLu替换成脉冲神经元，因为Converter的forward就是也只是完成relu的转化，得自己把identity的转化写出来，并且塞进converter的forward部分里面
        model_converter = Converter(neuron=args.neuron_name,args=args,T=args.time,step_mode=args.step_mode,fuse_flag=args.fuse)#fuse为True会输出fx的graph模型

        model = model_converter(model)
        model = forward_replace(args, model)
        model.to(device)
        model.eval()
        val = valpool(args)# using args.coding_type
        val(model, test_loader, device, args)
    elif args.mode == 'train_snn':
        print("Train SNN Mode")
    else:
        print("Not Support This Mode")
    if args.distributed:
        dist.destroy_process_group()
    
if __name__ == "__main__":
    main()