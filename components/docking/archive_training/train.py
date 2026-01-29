import copy
import math
import os
import shutil
from functools import partial

import wandb
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, t_to_sigma_individual
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, inference_epoch_fix
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage


# 定义训练函数，包含多个输入参数：args（命令行参数），model（模型），optimizer（优化器），scheduler（调度器），
# ema_weights（指数加权平均权重），train_loader（训练数据加载器），val_loader（验证数据加载器），t_to_sigma（转换函数），
# run_dir（运行目录），val_dataset2（第二个验证数据集）。
def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, val_dataset2):

    ## 使用偏函数来定义损失函数，并传递训练时所需的各个权重参数
    loss_fn = partial(loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                      tor_weight=args.tor_weight, no_torsion=args.no_torsion, backbone_weight=args.backbone_loss_weight,
                      sidechain_weight=args.sidechain_loss_weight)

    ## 初始化一些变量，存储训练过程中最好的验证损失和对应的最佳模型信息
    best_val_loss = math.inf## 初始化最小验证损失为正无穷大
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0# 初始化最好的推理值
    best_val_secondary_value = math.inf if args.inference_earlystop_goal == 'min' else 0# # 初始化第二个验证值
    best_epoch = 0# 初始化最佳epoch
    best_val_inference_epoch = 0# 初始化最佳验证推理epoch

    ## 初始化freeze参数为0，用于后续的参数冻结
    freeze_params = 0
    ## 根据早期停止的目标值选择调度模式
    scheduler_mode = args.inference_earlystop_goal if args.val_inference_freq is not None else 'min'
     # 如果使用的是'layer_linear_warmup'调度器，计算冻结的参数数量
    if args.scheduler == 'layer_linear_warmup':
        freeze_params = args.warmup_dur * (args.num_conv_layers + 2) - 1
        print("Freezing some parameters until epoch {}".format(freeze_params))

    print("Starting training...")
    # 开始训练循环，训练轮次为args.n_epochs
    for epoch in range(args.n_epochs):
        ## 每5个epoch打印一次训练的名称
        if epoch % 5 == 0: print("Run name: ", args.run_name)

        # 根据不同的调度策略进行参数解冻
        if args.scheduler == 'layer_linear_warmup' and (epoch+1) % args.warmup_dur == 0:
             # 计算warmup步骤，逐步解冻网络中的一些层
            step = (epoch+1) // args.warmup_dur
            if step < args.num_conv_layers + 2:
                print("New unfreezing step")
                # 更新优化器和调度器
                optimizer, scheduler = get_optimizer_and_scheduler(args, model, step=step, scheduler_mode=scheduler_mode)
            elif step == args.num_conv_layers + 2:
                print("Unfreezing all parameters")
                # 解冻所有参数
                optimizer, scheduler = get_optimizer_and_scheduler(args, model, step=step, scheduler_mode=scheduler_mode)
                ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)# 初始化EMA权重
        elif args.scheduler == 'linear_warmup' and epoch == args.warmup_dur:
            print("Moving to plateu scheduler")
            # 当达到指定的warmup阶段时，切换到plateau调度器
            optimizer, scheduler = get_optimizer_and_scheduler(args, model, step=1, scheduler_mode=scheduler_mode,
                                                               optimizer=optimizer)

        logs = {}# 存储训练过程中的日志信息
         # 调用train_epoch函数执行一个epoch的训练，并返回训练损失
        train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, loss_fn, ema_weights if epoch > freeze_params else None)
        # 打印当前epoch的训练损失信息
        print("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   sc {:.4f}  lr {:.4f}"
              .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                      train_losses['tor_loss'], train_losses['sidechain_loss'], optimizer.param_groups[0]['lr']))

        # 如果当前epoch已经解冻参数，更新EMA权重
        if epoch > freeze_params:
            ema_weights.store(model.parameters())# 存储当前模型的权重
            if args.use_ema: ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference# 如果启用了EMA，复制EMA权重到模型
        # 调用test_epoch函数执行一个epoch的验证，并返回验证损失
        val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, args.test_sigma_intervals)
         # 打印当前epoch的验证损失信息
        print("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   sc {:.4f}"
              .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss'], val_losses['sidechain_loss']))

        # 如果验证推理频率不为None，并且当前epoch满足验证推理的频率条件
        if args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0:
            # 从验证数据加载器中提取一部分数据进行推理，取样的数量为num_inference_complexes与验证集大小的最小值
            inf_dataset = [val_loader.dataset.get(i) for i in range(min(args.num_inference_complexes, val_loader.dataset.__len__()))]
            # 执行推理，返回推理结果的度量指标
            inf_metrics = inference_epoch_fix(model, inf_dataset, device, t_to_sigma, args)
            # 打印推理结果，包括rmsds_lt2, rmsds_lt5, min_rmsds_lt2, min_rmsds_lt5等
            print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
            # 更新日志，记录当前推理结果
            logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

        # 如果启用了双重验证，并且当前epoch满足验证推理的频率条件
        if args.double_val and args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0:
            # 从第二个验证数据集提取一部分数据进行推理
            inf_dataset = [val_dataset2.get(i) for i in range(min(args.num_inference_complexes, val_dataset2.__len__()))]
            # 执行推理并返回度量指标
            inf_metrics2 = inference_epoch_fix(model, inf_dataset, device, t_to_sigma, args)
             # 打印第二个验证集的推理结果
            print("Epoch {}: Val inference on second validation rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics2['rmsds_lt2'], inf_metrics2['rmsds_lt5'], inf_metrics2['min_rmsds_lt2'], inf_metrics2['min_rmsds_lt5']))
            # 更新日志，记录第二个验证集的推理结果
            logs.update({'valinf2_' + k: v for k, v in inf_metrics2.items()}, step=epoch + 1)
            # 更新日志，记录两个验证集推理结果的平均值
            logs.update({'valinfcomb_' + k: (v + inf_metrics[k])/2 for k, v in inf_metrics2.items()}, step=epoch + 1)

        # 如果训练推理频率不为None，并且当前epoch满足训练推理的频率条件
        if args.train_inference_freq != None and (epoch + 1) % args.train_inference_freq == 0:
            # 从训练数据集提取一部分数据进行推理
            inf_dataset = [train_loader.dataset.get(i) for i in range(min(min(args.num_inference_complexes, 300), train_loader.dataset.__len__()))]
            # 执行推理并返回度量指标
            inf_metrics = inference_epoch_fix(model, inf_dataset, device, t_to_sigma, args)
            # 打印训练集的推理结果
            print("Epoch {}: Train inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
            # 更新日志，记录训练集的推理结果
            logs.update({'traininf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

        # 如果当前epoch超过冻结参数的设置，则更新EMA权重
        if epoch > freeze_params:
             # 如果未使用EMA，将当前的EMA权重复制到模型参数中
            if not args.use_ema: ema_weights.copy_to(model.parameters())
            # 深拷贝模型的状态字典，用于后续保存最好的模型
            ema_state_dict = copy.deepcopy(model.module.state_dict() if device.type == 'cuda' else model.state_dict())
            # 恢复EMA权重
            ema_weights.restore(model.parameters())

        # 如果启用了wandb日志功能，记录训练、验证以及学习率等信息
        if args.wandb:
            logs.update({'train_' + k: v for k, v in train_losses.items()})# 更新训练损失
            logs.update({'val_' + k: v for k, v in val_losses.items()}) # 更新验证损失
            logs['current_lr'] = optimizer.param_groups[0]['lr']# 更新当前学习率
            # 使用wandb记录日志
            wandb.log(logs, step=epoch + 1)

        # 获取当前epoch的模型状态字典
        state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()
        # 如果验证推理的早期停止指标存在，并且当前epoch满足早期停止条件，则保存最佳模型
        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            # 保存当前最好的模型
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            # 如果模型已经解冻，保存EMA模型
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))

        # 如果启用了第二个早期停止指标，并且当前epoch满足早期停止条件，则保存最佳模型
        if args.inference_secondary_metric is not None and args.inference_secondary_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_secondary_metric] <= best_val_secondary_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_secondary_metric] >= best_val_secondary_value):
            best_val_secondary_value = logs[args.inference_secondary_metric]
            # 如果模型已经解冻，保存EMA模型
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_secondary_epoch_model.pt'))

        # 如果当前验证集上的损失小于等于最佳验证损失，更新最佳验证损失
        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']# 更新最佳验证损失
            best_epoch = epoch
            # 保存当前最好的模型
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
            # 如果模型已经解冻，则保存EMA模型
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))

        # 如果设置了模型保存频率，并且当前epoch满足保存频率条件
        if args.save_model_freq is not None and (epoch + 1) % args.save_model_freq == 0:
            # 将当前的最好的模型（best_model.pt）复制到以当前epoch命名的文件中
            shutil.copyfile(os.path.join(run_dir, 'best_model.pt'),
                            os.path.join(run_dir, f'epoch{epoch+1}_best_model.pt'))

        # 如果存在学习率调度器（scheduler）
        if scheduler:
            # 如果当前epoch小于冻结参数的epoch，或者在"线性预热"调度器下，当前epoch小于预热阶段的epoch
            if epoch < freeze_params or (args.scheduler == 'linear_warmup' and epoch < args.warmup_dur):
                # 执行一步调度
                scheduler.step()
             # 如果设置了验证推理频率
            elif args.val_inference_freq is not None:
                # 使用最佳验证推理值执行调度
                scheduler.step(best_val_inference_value)
            else:
                # 否则，使用当前验证损失执行调度
                scheduler.step(val_losses['loss'])

        # 保存当前epoch的模型、优化器状态和EMA权重
        torch.save({
            'epoch': epoch, # 当前epoch
            'model': state_dict, # 当前模型的状态字典
            'optimizer': optimizer.state_dict(),# 当前优化器的状态字典
            'ema_weights': ema_weights.state_dict(),# 当前EMA权重的状态字典
        }, os.path.join(run_dir, 'last_model.pt'))# 保存为'last_model.pt'

    # 打印出最好的验证损失及其对应的epoch
    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    # 打印出最好的推理指标及其对应的epoch
    print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))


# 主训练函数
def main_function():
    args = parse_train_args()# 解析命令行参数
    # 如果提供了配置文件，通过yaml加载并更新命令行参数
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader) # 解析yaml配置文件
        arg_dict = args.__dict__# 将args转换为字典格式
        # 遍历配置字典，更新命令行参数
        for key, value in config_dict.items():
            if isinstance(value, list):# 如果配置值是列表，则将列表值附加到命令行参数中
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value# 否则直接覆盖命令行参数
        args.config = args.config.name # 保存配置文件的名称到args中
     # 确保早停目标为 'max' 或 'min'，否则会引发错误
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
     # 如果设置了验证推理频率，并且使用了调度器，确保调度器耐心度大于验证推理频率
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs # 否则训练将在一定epoch后停止
    # 如果启用了CUDNN基准模式，设置CUDNN的优化选项
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # 如果启用了WandB（Weight and Biases，用于实验追踪），初始化WandB
    if args.wandb:
        wandb.init(
            entity='',
            settings=wandb.Settings(start_method="fork"),# 设置启动方法为"fork"
            project=args.project,# 设置项目名称
            name=args.run_name,# 设置运行名称
            config=args# 将命令行参数传递给WandB
        )

    # 构建数据加载器
    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)# 定义一个偏函数，用于计算t_to_sigma
    train_loader, val_loader, val_dataset2 = construct_loader(args, t_to_sigma, device)# 构建训练和验证数据加载器
    
    # 获取模型，优化器，调度器
    model = get_model(args, device, t_to_sigma=t_to_sigma)# 获取模型实例
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min') # 获取优化器和调度器
    ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)# 设置EMA权重平滑

    # 如果设置了恢复训练，尝试从指定的检查点加载模型和优化器
    if args.restart_dir:
        try:
            dict = torch.load(f'{args.restart_dir}/{args.restart_ckpt}.pt', map_location=torch.device('cpu'))# 加载恢复检查点
            if args.restart_lr is not None: dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr# 如果指定了恢复学习率，更新学习率
            optimizer.load_state_dict(dict['optimizer'])# 加载优化器状态
            model.module.load_state_dict(dict['model'], strict=True)# 加载模型权重
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(dict['ema_weights'], device=device)# 加载EMA权重
            print("Restarting from epoch", dict['epoch']) # 输出恢复的epoch
        except Exception as e:
            print("Exception", e)
            # 如果加载恢复模型失败，则加载最佳模型进行训练
            dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
            model.module.load_state_dict(dict, strict=True)# 加载最佳模型权重
            print("Due to exception had to take the best epoch and no optimiser") # 输出异常信息
    # 如果设置了预训练模型路径，则加载预训练模型
    elif args.pretrain_dir:
        dict = torch.load(f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt', map_location=torch.device('cpu'))
        model.module.load_state_dict(dict, strict=True)# 加载预训练模型权重
        print("Using pretrained model", f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt') # 输出预训练模型路径

     # 计算模型的总参数量
    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')# 输出模型的参数数量

    # 如果启用了WandB，则记录模型参数数量
    if args.wandb:
        wandb.log({'numel': numel})

    # 记录模型参数
    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)# 设置训练日志目录
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')# 设置保存配置的yaml文件路径
    save_yaml_file(yaml_file_name, args.__dict__)# 将命令行参数保存为yaml文件
    args.device = device# 设置设备（CPU或GPU）

    # 启动训练
    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, val_dataset2)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()
