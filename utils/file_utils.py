import os, sys, shutil
import cv2
import torch
import glob
import re
def rm_sub_files(path):
    shutil.rmtree(path)
    os.mkdir(path)

def load_checkpoint( model_name, path='./models',is_best=True):
    if is_best:
        ckpt_file = os.path.join(path, 'model_latest.pth.tar')
    elif model_name != None:
        ckpt_file = os.path.join(path,model_name)
    else:
        files = glob.glob(os.path.join(path, '{:06d}.pth.tar'))
        files.sort()
        ckpt_file = files[-1]
        print(files)
    return torch.load(ckpt_file)

def save_checkpoint(state, globel_iter, path='./models', is_best=True, max_keep=1000):
    filename = os.path.join(path, '{:06d}.pth.tar'.format(globel_iter))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

    files = sorted(os.listdir(path))
    rm_files = files[0: max(0, len(files)-max_keep)]
    for f in rm_files:
        os.remove(os.path.join(path, f))

def tryint(s):                       
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):                
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_humanly(v_list):    
    return sorted(v_list, key=str2int)

def save_image(image,addr,num):
    address = addr + "\\" + 'image' + str(num)+ '.png'
    cv2.imwrite(address,image)

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                L.append(os.path.join(root, file))
    return L

