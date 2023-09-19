import os
import numpy as np
import shutil
import cv2
main_dir = "/yuning/Denoise/DAVIS/JPEGImages/Full-Resolution/"
new_dir = "/yuning/Denoise/DAVIS/JPEGImages/crop/"
main_list = os.listdir(main_dir)
#print(main_list)
frame_per=7
crop_size=256

for i in range(len(main_list)):
    sub_path = main_dir + main_list[i]
    dst_path = new_dir + main_list[i]
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    print(sub_path)
    sub_list = os.listdir(sub_path)
    length = len(sub_list)
    num_division = length//4
    for j in range(num_division):
        print(j)
        res = []
        start = np.random.randint(0,length-frame_per+1)
        directory = dst_path+"//"+main_list[i]+"_%02d"%j
        if not os.path.exists(directory):
            os.mkdir(directory)
        for k in range(start,start+frame_per):
            ori_path = os.path.join(sub_path,'%05d.jpg'%k)
            dst = directory
            shutil.copy(ori_path,dst)
            new_loc = os.path.join(dst,'%05d.jpg'%k)
            new_name = os.path.join(dst,'%05d.jpg'%(k-start))
            os.rename(new_loc,new_name)
            res.append(new_name)
        img = cv2.imread(res[0])
        h = img.shape[0]
        w = img.shape[1]
        crop_division = h*w//(crop_size**2)
        for z in range(crop_division):
            hs = np.random.randint(0,h-crop_size+1)
            ws = np.random.randint(0,w-crop_size+1)
            directory_crop = directory + "//" + "crop_%02d"%z
            if not os.path.exists(directory_crop):
                os.mkdir(directory_crop)
                with open("train.txt","a") as f:
                    text = main_list[i] + "//" +main_list[i] + "_%02d"%j + "//" + "crop_%02d"%z+"\n"
                    f.write(text)
                    # require manully delete the first line in the txt file
            for k in range(frame_per):
                img = cv2.imread(res[k])
                cropped_image = img[hs:hs+crop_size,ws:ws+crop_size]
                cropped_image_savepath = directory_crop + "//" + "%05d.jpg"%k
                cv2.imwrite(cropped_image_savepath,cropped_image)