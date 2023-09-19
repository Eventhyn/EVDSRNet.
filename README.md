# EVDSRNet.
This is the official implementation of the paper "EFFICIENT JOINT VIDEO DENOISING AND SUPER-RESOLUTION" which is accepted at IEEE ICIP2023. The paper can be officially viewed at [IEEE Xplore](https://www.google.com).

# Overall structure
![alt text](images/Architecture.png)
In this paper, we present two versions of our proposed method, one is noted as medium and another is noted as large. 

The trade-off between the performance and computation cost is parameterized by N_b and N_c where N_b is the number of blocks in the super-resolution network and N_c is the feature dimension of these blocks. 

Our medium model adopts N_b = 8 and N_c = 128 where large model adopts N_b = 16 and N_c = 256.

Besides, our method operates in a non-blind manner, which means the level of noise is fixed for a specific model. 

We have trained and evaluated our model at two Additive White Gaussian Noise (AWGN) levels: sigma = 10 and sigma = 20.

# Performance
![alt text](images/Performance.png)
Since there is no joint super-resolution and denoising model for video input at the time when the paper is written, we chose a straightforward solution to set the baseline comparison for our method: Denoise-then-SR.

For straightforward solutions, we combine FastDVDNet+RRN as an efficient solution and VRT-De+VRT-SR as a quality-favored solution.

Our method presents significant improvement in the trade-off between performance and computation cost compared to straightforward solutions: 
* When compared to the efficient solution, our method achieves better PSNR and faster running speed while maintaining similar model sizes. 
* When compared to the quality-favored solution, our method achieves similar PSNR while running significantly faster and maintaining a much smaller model size.

# Model Weights
The model weights can be downloaded from Google Drive:
* Medium Model, sigma = 10 [Google](https://drive.google.com/file/d/1uSQgl6DwhmUR4MR4wnt0ZgVnuhwc3qAP/view?usp=sharing)
* Medium Model, sigma = 20 [Google](https://drive.google.com/file/d/11oJiXHnyJgn9EajK6cxEQqTIUNL2JHQr/view?usp=sharing)
* Large Model, sigma = 10 [Google](https://drive.google.com/file/d/11sw8PQBh6Gc3bFcCnNYxqVg2CEnwwOPE/view?usp=sharing)
* Large Model, sigma = 20 [Google](https://drive.google.com/file/d/1rxwPp7soPWnJ3qVBhSy5SUxhXsbKGlo8/view?usp=sharing)

# Evaluate
We provide scripts for evaluating our model's performance, the example running command is as follows:
```
python test_udm.py --dataset_path D:\\Source_code\\Joint_DenoiseSR\\udm10 --txt_path D:\\Source_code\\Joint_DenoiseSR\\udm10 --model_size medium --sigma 10
```
```
python test_davis.py --dataset_path D:\\Source_code\\Joint_DenoiseSR\\DAVIS-2017-test-dev-480p\\DAVIS\\JPEGImages\\480p --txt_path D:\\Source_code\\Joint_DenoiseSR\\DAVIS-2017-test-dev-480p\\DAVIS --model_size medium --sigma 10
```
* dataset_path: The directory where you put the testing dataset
* txt_path: The directory where you put the dataset information prepared for testing
* model_size: medium or large
* sigma: 10 or 20

The prepared txt file used in testing can also be downloaded from Google Drive:

* UDM: [Google](https://drive.google.com/file/d/19kamnkc7907dgGpXtgKRXG9Id52z-mrS/view?usp=sharing)
* Davis [Google](https://drive.google.com/file/d/1bvUus3UjXysoDxLokIggufkMDrSWyADD/view?usp=sharing)
# Train
We have adopted the DAVIS2017 training split as our training dataset, the training patches are extracted from an original split of DAVIS, which contains randomly cropped 7 consecutive frames of 256x256 spatial resolution. The pre-processing script is also provided, see data_prepare.py.

Following are example procedures for training our model:
```
python data_prepare.py
```
* Make sure you check and change the directory of the Davis dataset as well as the output training patches directory inside the data_prepare.py
```
python train_eval.py --dataset_path /yuning/Denoise/DAVIS/JPEGImages/crop --txt_path /yuning/Denoise/ --im_size 256 -bs 32 --cuda -lr 1e-4 -nw 4
```
* The example training code involves Wandb as a training metric monitoring library, if not wanted, please comment out the relevant code.
