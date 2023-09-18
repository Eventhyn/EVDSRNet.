import torch
import os
import torch.nn as nn
import numpy as np
from arch.model_arch import *
from utils.data_utils import *
from utils.file_utils import *
from torch.utils.data import Dataset, DataLoader
from data_provider_eval import *
import time
import setproctitle
import cv2 as cv
import torch.multiprocessing
from torch_ema import ExponentialMovingAverage
setproctitle.setproctitle('Yuning')
import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-dp', default='D:\\Source_code\\Joint_DenoiseSR\\udm10', help='the path of udm')
    parser.add_argument('--txt_path', '-tp', default='D:\\Source_code\\Joint_DenoiseSR\\udm10', help='the path of train/eval txt file')
    parser.add_argument('--model_size', '-ms', default='medium')
    parser.add_argument('--sigma', '-s', default=10)
    return parser.parse_args()
def eval(args):
    data_set_val_udm = Video_Provider_val_udm(
        base_path= args.dataset_path,
        txt_file=os.path.join(args.txt_path, 'val.txt'),
        im_size=None,
        frames=32,
        sigma=int(args.sigma)
    )
    data_loader_val_udm = DataLoader(
        dataset=data_set_val_udm,
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

    with torch.no_grad():                  
        for i,(val_data,val_data1, val_gt) in enumerate(data_loader_val_udm):
            if i == 0:
                val_data_l_udm=val_data.cuda()
                val_gt_l_udm=val_gt.cuda()
                print("val_gt_l_udm",val_gt.shape)
            else:
                val_data_l_udm=torch.cat((val_data_l_udm,val_data.cuda()))
                val_gt_l_udm=torch.cat((val_gt_l_udm,val_gt.cuda()))    
    t = time.time()       
    with torch.no_grad():
        psnr_l = []
        ssim_l = []
        for k in range(10):
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
                    with ema.average_parameters():
                        h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],init_h,init_o)
                    out.append(o)
                    init_flag = False
                else:
                    with ema.average_parameters():
                        h,o = model(data[:,i,:,:,:],data[:,i+1,:,:,:],data[:,i+2,:,:,:],h,o)
                    out.append(o)
            out = torch.stack(out, dim=1)
            out = torch.clamp(out,0,1)
            gt = gt[:,2:32,...]
         
            psnr_v = calculate_psnr(out[0], gt[0])
            ssim_v = calculate_ssim(out[0], gt[0])
            psnr_l.append(psnr_v)
            ssim_l.append(ssim_v)
            print(psnr_l)

        print(psnr_l)
        print(ssim_l)
        mean1 = np.mean(psnr_l)
        mean2 = np.mean(ssim_l)
        print("Val_PSNR_udm: {:.4f}".format(mean1))
        print("Val_SSIM_udm: {:.4f}".format(mean2))

                                         
if __name__ == '__main__':
    args = args_parser()
    eval(args)