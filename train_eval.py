from torch_ema import ExponentialMovingAverage
import torch
import os, sys, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import wandb
import torch.nn as nn
import numpy as np
from arch.model_arch import *
from utils.data_utils import *
from utils.file_utils import *
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from data_provider import *
import torch.optim as optim
import time
import setproctitle
from torchinfo import summary
import cv2 as cv
import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
#setproctitle.setproctitle('Yuning')
wandb.init(project="JSRD_blind",id="test_ifwork",resume=True)

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-dp', default='/yuning/Denoise/DAVIS/JPEGImages/crop', help='the path of davis')
    parser.add_argument('--txt_path', '-tp', default='/yuning/Denoise/', help='the path of train/eval txt file')
    parser.add_argument('--batch_size', '-bs', default=64, type=int, help='batch size')
    parser.add_argument('--frames', '-f', default=3, type=int)
    parser.add_argument('--im_size', '-s', default=256, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float)
    parser.add_argument('--num_worker', '-nw', default=4, type=int, help='number of workers to load data by dataloader')
    parser.add_argument('--restart', '-r', action='store_true', help='whether to restart the train process')
    parser.add_argument('--eval', '-e', action='store_true', help='whether to work on the eval mode')
    parser.add_argument('--cuda', action='store_true', help='whether to train the network on the GPU, default is mGPU')
    parser.add_argument('--max_epoch', default=1000, type=int)
    return parser.parse_args()
def train(args):
    data_set = Video_Provider_davis(
            base_path=args.dataset_path,
            txt_file=os.path.join(args.txt_path, 'train.txt'),
            im_size=256,
            frames=7
        )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory = True,
        prefetch_factor=8
    )

    data_set_val_udm = Video_Provider_val_udm(
        base_path='/yuning/Denoise/udm10',
        txt_file=os.path.join('/yuning/Denoise/udm10', 'val.txt'),
        im_size=None,
        frames=32
    )
    data_loader_val_udm = DataLoader(
        dataset=data_set_val_udm,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    
    print("torchversion",torch.__version__)
    n_c = 128
    n_b = 8
    model = VSRorg(scale=4,n_c=n_c,n_b=n_b)
    # run on the GPU
    if args.cuda:
        model = nn.DataParallel(model.cuda())
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=400000)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    if args.restart:
        epoch = 0
        global_iter = 0
        best_loss = np.inf
        print('Start the train process.')
    else:
        try:
            try:        
                state = load_checkpoint('.//models//test_ifwork', is_best=False)
            except:
                print("no model to load")
            epoch = state['epoch']
            print(epoch)
            global_iter = state['global_iter']
            print(global_iter)
            best_loss = state['best_loss']
            ema.load_state_dict(state['ema'])
            optimizer.load_state_dict(state['optimizer'])
            model.module.load_state_dict(state['state_dict'])
            print('Model load OK at global_iter {}, epoch {}.'.format(global_iter, epoch))
        except:
            epoch = 0
            global_iter = 0
            best_loss = np.inf
            print('There is no any model to load, restart the train process.')
    loss_func = nn.MSELoss()
    t = time.time()
    loss_temp = 0
    psnr_temp = 0
    ssim_temp = 0
    model.train()
    
                      
    for i,(val_data,val_data1, val_gt) in enumerate(data_loader_val_udm):
                    if i == 0:
                      val_data_l_udm=val_data.cuda()
                      val_data_l1_udm=val_data1.cuda()
                      val_gt_l_udm=val_gt.cuda()
                      print("val_gt_l_udm",val_gt.shape)
                    else:
                      val_data_l_udm=torch.cat((val_data_l_udm,val_data.cuda()))
                      val_data_l1_udm=torch.cat((val_data_l1_udm,val_data1.cuda()))
                      val_gt_l_udm=torch.cat((val_gt_l_udm,val_gt.cuda()))    
   
    t = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
    for e in range(epoch, args.max_epoch):
        for iter, (data,data1,gt) in enumerate(data_loader):
            if args.cuda:
                data = data.cuda()
                data1 = data1.cuda()
                gt = gt.cuda()
            init_flag = True
            out = []
            loss_temp = 0
            for i in range(7-2):
                if init_flag == True:
                    init_h = torch.zeros((data.shape[0],n_c,data.shape[3],data.shape[4]))
                    init_o = data[:,0,:,:,:]
                    init_o = torch.nn.functional.interpolate(init_o, scale_factor=4, mode='bicubic', antialias=True)
                    try:
                        h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],init_h,init_o)
                    except:
                        print("model is not working!")
                    out.append(o)
                    init_flag = False
                else:
                    h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],h,o)
                    out.append(o)
            out = torch.stack(out, dim=1)
            gt = gt[:,2:7,...]
            loss = loss_func(gt, out)
            loss_temp = 0
            global_iter += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update()
            
            lr = scheduler.get_last_lr()
            
            psnr = 0
            ssim = 0
            if global_iter % 10 == 0:
                  psnr = calculate_psnr(out[:,-1,...], gt[:,-1,...])
                  wandb.log({"train_psnr": psnr},step=global_iter)
                  wandb.log({"loss_t": loss_temp/3},step=global_iter)
                  print(
                      'epoch: {:3d}{:6d} loss: {:.4f}, loss_t: {:.4f}, psnr: {:.4f}, time: {:.2}S. '
                      .format(e,global_iter, loss, loss_temp, psnr, time.time() - t)
                      )
                  t = time.time()
            psnr_temp += psnr
            ssim_temp += ssim
            loss_temp += loss
            ## for udm test
            if global_iter % 100 == 0:
                with torch.no_grad():
                    psnr_l = []
                    for k in range(1):
                        init_flag = True
                        out = []
                        data = val_data_l_udm[k]
                        data = data.reshape(1,data.shape[0],data.shape[1],data.shape[2],data.shape[3])
                        gt = val_gt_l_udm[k]
                        gt = gt.reshape(1,gt.shape[0],gt.shape[1],gt.shape[2],gt.shape[3])
                        for i in range(32-2):
                            if init_flag == True:
                                init_h = torch.zeros((data.shape[0],n_c,data.shape[3],data.shape[4]))
                                init_o = data[:,0,:,:,:]
                                init_o = torch.nn.functional.interpolate(init_o, scale_factor=4, mode='bicubic', antialias=True)
                                h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],init_h,init_o)
                                out.append(o)
                                init_flag = False
                            else:
                                h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],h,o)
                                out.append(o)
                        out = torch.stack(out, dim=1)
                        gt = gt[:,2:32,...]
                        psnr_v = calculate_psnr(out[0], gt[0]) 
                        psnr_v= round(psnr_v,3)
                        psnr_l.append(psnr_v)
                    mean1 = np.mean(psnr_l)
                    print("val_psnr_udm: {:.2f}".format(mean1))
                    wandb.log({"val_psnr_udm": mean1},step=global_iter)
                    
                    psnr_l = []
                    with ema.average_parameters():
                        for k in range(1):
                            init_flag = True
                            out = []
                            data = val_data_l_udm[k]
                            data = data.reshape(1,data.shape[0],data.shape[1],data.shape[2],data.shape[3])
                            gt = val_gt_l_udm[k]
                            gt = gt.reshape(1,gt.shape[0],gt.shape[1],gt.shape[2],gt.shape[3])
                            for i in range(32-2):
                                if init_flag == True:
                                    init_h = torch.zeros((data.shape[0],n_c,data.shape[3],data.shape[4]))
                                    init_o = data[:,0,:,:,:]
                                    init_o = torch.nn.functional.interpolate(init_o, scale_factor=4, mode='bicubic', antialias=True)
                                    h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],init_h,init_o)
                                    out.append(o)
                                    init_flag = False
                                else:
                                    h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],h,o)
                                    out.append(o)
                            out = torch.stack(out, dim=1)
                            gt = gt[:,2:32,...]
                            psnr_v = calculate_psnr(out[0], gt[0])
                            psnr_v= round(psnr_v,3)
                            psnr_l.append(psnr_v)
                        mean1 = np.mean(psnr_l)
                        print("val_psnr_udm_ema: {:.2f}".format(mean1))
                        wandb.log({"val_psnr_udm_ema": mean1},step=global_iter)

            if global_iter % 500 == 0:
                loss_temp /= 500
                psnr_temp /= 500
                ssim_temp /= 500
                is_best = True if loss_temp < best_loss else False
                best_loss = min(best_loss, loss_temp)
                state = {
                    'state_dict': model.module.state_dict(),
                    'epoch': e,
                    'global_iter': global_iter,
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'ema': ema.state_dict()
                }
                save_checkpoint(state, global_iter, path='./models//test_ifwork', is_best=is_best, max_keep=1000)
                t = time.time()
                loss_temp, psnr_temp, ssim_temp = 0, 0, 0


if __name__ == '__main__':
    args = args_parser()
    print(args)
    if not args.eval:
        train(args)