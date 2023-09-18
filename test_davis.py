import torch
import os, sys, shutil
import torch.nn as nn
import numpy as np
from arch.model_arch import *
from utils.data_utils import *
from utils.file_utils import *
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from data_provider_eval import *
import torch.optim as optim
import time
import setproctitle
from torchinfo import summary
import cv2 as cv
import torch.multiprocessing
from torch_ema import ExponentialMovingAverage
setproctitle.setproctitle('Yuning')
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-dp', default='D:\\Source_code\\Joint_DenoiseSR\\DAVIS-2017-test-dev-480p\\DAVIS\\JPEGImages\\480p', help='the path of davis')
    parser.add_argument('--txt_path', '-tp', default='D:\\Source_code\\Joint_DenoiseSR\\DAVIS-2017-test-dev-480p\\DAVIS', help='the path of train/eval txt file')
    parser.add_argument('--model_size', '-ms', default='medium')
    parser.add_argument('--sigma', '-s', default=10)
    return parser.parse_args()

def eval(args):
    data_set_val_davis = Video_Provider_val_davis(
        base_path=args.dataset_path,
        txt_file=os.path.join(args.txt_path, 'test-dev.txt'),
        im_size=None,
        frames=32,
        sigma=int(args.sigma)
    )
    data_loader_val_davis = DataLoader(
        dataset=data_set_val_davis,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    
    if args.model_size == "medium":
        n_b = 8
        n_c = 128
        if args.sigma == "10":
            model_name = "medium_10.pth.tar" ######## nb8 nc128 sigma10
        elif args.sigma == "20":
            model_name = "medium_20.pth.tar" ######## nb8 nc128 sigma20
        else:
            print("Wrong sigma value!")
    elif args.model_size == "large":
        n_b = 16
        n_c = 256
        if args.sigma == "10":
            model_name = "large_10.pth.tar" ######## nb16 nc256 sigma10
        elif args.sigma == "20":
            model_name = "large_20.pth.tar" ######## nb16 nc256 sigma20
        else:
            print("Wrong sigma value!")
    else:
        print("Wrong Model Size")

    model = VSRorg(scale=4,n_c=n_c,n_b=n_b)
    # run on the GPU
    model = nn.DataParallel(model.cuda())
    model.eval()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    state = load_checkpoint(model_name,'./models', is_best=False)
    model.module.load_state_dict(state['state_dict'])
    ema.load_state_dict(state['ema'])


    psnr_l = []
    ssim_l = []
    leng_l = []
    for iter, (data,data1,gt) in enumerate(data_loader_val_davis):
        with torch.no_grad():
            data = data.cuda()
            data1 = data1.cuda()
            gt = gt.cuda()
            init_flag = True
            length = data.shape[1]
            out = []
            loss_temp = 0
            for i in range(length-2):
                if init_flag == True:
                    init_h = torch.zeros((data.shape[0],n_c,data.shape[3],data.shape[4]))
                    init_o = data[:,0,:,:,:]
                    init_o = torch.nn.functional.interpolate(init_o, scale_factor=4, mode='bicubic', antialias=True)

                    with ema.average_parameters():
                        h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],init_h,init_o)
                    out.append(o)
                    init_flag = False
                else:
                    with ema.average_parameters():
                        h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],h,o)
                    out.append(o)
            out = torch.stack(out, dim=1)
            gt = gt[:,2:length,...]
            psnr_v = calculate_psnr(out[0], gt[0])
            ssim_v = calculate_ssim(out[0], gt[0])
            leng = len(out[0])
            leng_l.append(leng)
            psnr_l.append(psnr_v*leng)
            ssim_l.append(ssim_v*leng)
            print("iteration{} is done".format(iter))
            print(psnr_l)
            print("final_psnr = ", np.sum(psnr_l)/np.sum(leng_l))
    print("finished!")
    print("final_psnr = ", np.sum(psnr_l)/np.sum(leng_l))
    print("final_ssim = ", np.sum(ssim_l)/np.sum(leng_l))

                                         
if __name__ == '__main__':
    args = args_parser()
    eval(args)