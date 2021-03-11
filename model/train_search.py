import os
import sys
import time
import glob
import logging
import torch
import numpy as np
from tqdm import tqdm
from thop import profile
from random import shuffle
import torch.nn as nn
import torch.utils
from torch.utils.tensorboard import SummaryWriter
from config_search import config
#import torchvision.datasets as dataset
#import torchvision.transforms as transform
#import torchvision.transforms.functional as TF
#from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import get_train_loader
from tools.datasets import Cityscapes
from architect import Architect
from model_search import Network_Multi_Path as Network
from model_seg import Network_Multi_Path_Infer
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from utils.init_func import init_weight
from eval import SegEvaluator
from matplotlib import pyplot as plt

def main():
    config.save = 'search-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(os.path.join(config.save, 'runs'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))

    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # setup model and architecture
    model = Network(config.num_classes, config.layers, Fch=config.Fch, width_mult_list=config.width_mult_list, prun_modes=config.prun_modes, stem_head_width=config.stem_head_width)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
    logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
    model = model.cuda()
    init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')
    architect = Architect(model, config)

    # Set up dataloaders
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'down_sampling': config.down_sampling}
    index_select = list(range(config.num_train_imgs))
    shuffle(index_select)  # shuffle to make sure balanced dataset split
    train_loader_model = get_train_loader(config, Cityscapes, portion=config.train_portion, index_select=index_select)
    train_loader_arch = get_train_loader(config, Cityscapes, portion=config.train_portion-1, index_select=index_select)

    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                             config.image_std, model, config.eval_scale_array, config.eval_flip, 0, config=config,
                             verbose=False, save_path=None, show_image=False)

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.stem.parameters())
    parameters += list(model.cells.parameters())
    parameters += list(model.refine32.parameters())
    parameters += list(model.refine16.parameters())
    parameters += list(model.head0.parameters())
    parameters += list(model.head1.parameters())
    parameters += list(model.head2.parameters())
    parameters += list(model.head02.parameters())
    parameters += list(model.head12.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    # lr policy ##############################
    lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.978)

    # training loop
    tbar = tqdm(range(config.nepochs), ncols=80)
    valid_mIoU_history = []; FPSs_history = [];
    latency_supernet_history = []; latency_weight_history = [];
    valid_names = ["8s", "16s", "32s", "8s_32s", "16s_32s"]
    arch_names = {0: "teacher", 1: "student"}
    for epoch in tbar:
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=True)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
        with torch.no_grad():
            valid_mIoUss = []; FPSs = []
            model.prun_mode = None
            for idx in range(len(model._arch_names)):
                # arch_idx
                model.arch_idx = idx
                valid_mIoUs, fps0, fps1 = infer(epoch, model, evaluator, logger)
                valid_mIoUss.append(valid_mIoUs)
                FPSs.append([fps0, fps1])
                for i in range(5):
                    # preds
                    logger.add_scalar('mIoU/val_%s_%s'%(arch_names[idx], valid_names[i]), valid_mIoUs[i], epoch)
                    logging.info("Epoch %d: valid_mIoU_%s_%s %.3f"%(epoch, arch_names[idx], valid_names[i], valid_mIoUs[i]))
                if config.latency_weight[idx] > 0:
                    logger.add_scalar('Objective/val_%s_8s_32s'%arch_names[idx], objective_acc_lat(valid_mIoUs[3], 1000./fps0), epoch)
                    logging.info("Epoch %d: Objective_%s_8s_32s %.3f"%(epoch, arch_names[idx], objective_acc_lat(valid_mIoUs[3], 1000./fps0)))
                    logger.add_scalar('Objective/val_%s_16s_32s'%arch_names[idx], objective_acc_lat(valid_mIoUs[4], 1000./fps1), epoch)
                    logging.info("Epoch %d: Objective_%s_16s_32s %.3f"%(epoch, arch_names[idx], objective_acc_lat(valid_mIoUs[4], 1000./fps1)))
            valid_mIoU_history.append(valid_mIoUss)
            FPSs_history.append(FPSs)
            latency_supernet_history.append(architect.latency_supernet)
            latency_weight_history.append(architect.latency_weight)

        save(model, os.path.join(config.save, 'weights.pt'))
    

def train(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=True):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in pbar:
        optimizer.zero_grad()

        minibatch = dataloader_model.next()
        imgs = minibatch['data']
        target = minibatch['label']
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)


        # get a random minibatch from the search queue with replacement
        pbar.set_description("[Arch Step %d/%d]" % (step + 1, len(train_loader_model)))
        minibatch = dataloader_arch.next()
        imgs_search = minibatch['data']
        target_search = minibatch['label']
        imgs_search = imgs_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)
        loss_arch = architect.step(imgs, target, imgs_search, target_search)
        if (step+1) % 10 == 0:
            logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+step)
            logger.add_scalar('arch/latency_supernet', architect.latency_supernet, epoch*len(pbar)+step)

        loss = model._loss(imgs, target, True)
        logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))
    torch.cuda.empty_cache()
    # del loss
    # if update_arch: del loss_arch


def infer(epoch, model, evaluator, logger, FPS=True):
    model.eval()
    mIoUs = []
    for idx in range(5):
        evaluator.out_idx = idx
        # _, mIoU = evaluator.run_online()
        _, mIoU = evaluator.run_online_multiprocess()
        mIoUs.append(mIoU)
    if FPS:
        fps0, fps1 = arch_logging(model, config, logger, epoch)
        return mIoUs, fps0, fps1
    else:
        return mIoUs


def arch_logging(model, args, logger, epoch):
    input_size = (1, 3, 1024, 2048)
    net = Network_Multi_Path_Infer(
        [getattr(model, model._arch_names[model.arch_idx]["alphas"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["alphas"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["alphas"][2]).clone().detach()],
        [None, getattr(model, model._arch_names[model.arch_idx]["betas"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["betas"][1]).clone().detach()],
        [getattr(model, model._arch_names[model.arch_idx]["ratios"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["ratios"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["ratios"][2]).clone().detach()],
        num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width[model.arch_idx])

    plot_op(net.ops0, net.path0, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops0_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops1, net.path1, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops1_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops2, net.path2, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops2_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)

    net.build_structure([2, 0])
    net = net.cuda()
    net.eval()
    latency0, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps0_arch%d"%model.arch_idx, 1000./latency0, epoch)
    logger.add_figure("arch/path_width_arch%d_02"%model.arch_idx, plot_path_width([2, 0], [net.path2, net.path0], [net.widths2, net.widths0]), epoch)

    net.build_structure([2, 1])
    net = net.cuda()
    net.eval()
    latency1, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps1_arch%d"%model.arch_idx, 1000./latency1, epoch)
    logger.add_figure("arch/path_width_arch%d_12"%model.arch_idx, plot_path_width([2, 1], [net.path2, net.path1], [net.widths2, net.widths1]), epoch)

    return 1000./latency0, 1000./latency1

if __name__ == '__main__':
  main() 