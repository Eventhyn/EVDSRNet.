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
The model weights can be downloaded from Google Drive and Baidu Netdisk:
* Medium Model, sigma = 10 [Google](https://drive.google.com/file/d/1uSQgl6DwhmUR4MR4wnt0ZgVnuhwc3qAP/view?usp=sharing)/Baidu
* Medium Model, sigma = 20 [Google](https://drive.google.com/file/d/11oJiXHnyJgn9EajK6cxEQqTIUNL2JHQr/view?usp=sharing)/Baidu
* Large Model, sigma = 10 [Google](https://drive.google.com/file/d/11sw8PQBh6Gc3bFcCnNYxqVg2CEnwwOPE/view?usp=sharing)/Baidu
* Large Model, sigma = 20 [Google](https://drive.google.com/file/d/1rxwPp7soPWnJ3qVBhSy5SUxhXsbKGlo8/view?usp=sharing)/Baidu

# Evaluate
We provide scripts for evaluating our model's performance, the example running command is as follows:
```
python test_udm.py --dataset_path D:\\Source_code\\Joint_DenoiseSR\\udm10 --txt_path D:\\Source_code\\Joint_DenoiseSR\\udm10 --model_size medium --sigma 10
```
* dataset_path: The directory where you put the testing dataset
* txt_path: The directory where you put the dataset information prepared for testing
* model_size: medium or large
* sigma: 10 or 20

The prepared txt file used in testing can also be downloaded from Google Drive or BaiduNetdisk:

* UDM: [Google](https://drive.google.com/file/d/19kamnkc7907dgGpXtgKRXG9Id52z-mrS/view?usp=sharing)/Baidu

# Train
Will be released soon
