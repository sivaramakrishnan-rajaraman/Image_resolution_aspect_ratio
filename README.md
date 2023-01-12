# Does image resolution impact fine-grained Tuberculosis-consistent lesion segmentation performance in chest X-rays?

To study the impact of varying spatial resolution and aspect ratio adjustments toward segmenting TB-consistent lesions in chest X-rays

## Objective:

Dataset: Shenzhen TB CXR dataset and their corresponding pixel-wise TB consistent lesion masks (287 images, 201/29/57- train/Val/Test split). 

The best-performing Inception-V3-based UNet mode from our previous study https://www.mdpi.com/2306-5354/9/9/413 is selected.

The model performance is analyzed under the following settings:

    1. Direct resizing with original data (at resolutions 32×32, 64×64, 128×128, 256×256, 512×512, 768×768, and 1024×1024)
    2. Direct resizing with lung-cropped ROI at aforementioned resolutions.
    3. Aspect-ratio adjusted data (best of step1/step2) at aforementioned resolutions.
    4. Improve segmentation performance at the optimal resolution/ratio through a combinatorial approach of storing model snapshots, optimal test time augmentation, segmentation threshold optimization, and snapshot averaging.

## Requirements:

    h5py==3.1.0
    imageio==2.11.1
    matplotlib==3.5.1
    numpy==1.19.5
    opencv_python==4.5.4.58
    pandas==1.3.4
    Pillow==9.1.1
    scikit_image==0.18.3
    scikit_learn==1.1.1
    scipy==1.7.2
    segmentation_models==1.0.1
    skimage==0.0
    tensorflow==2.6.2
    tqdm==4.62.3
    
![Fig_1](https://user-images.githubusercontent.com/45852019/212079319-ba402ad1-b86d-4815-ab59-0d25d83d3995.png)


![image](https://user-images.githubusercontent.com/45852019/210234614-ba1aee0b-a679-46db-a388-82b2e9faf49a.png)


![image](https://user-images.githubusercontent.com/45852019/210234673-2c7f0c9a-9746-4b19-bc69-d54eaaadf233.png)


![image](https://user-images.githubusercontent.com/45852019/210234712-24cae50f-9205-47e8-993e-b499e1543289.png)


## Test-time augmentation combination

![image](https://user-images.githubusercontent.com/45852019/210234837-b6297c06-24a6-4071-be09-c6c13a882ca6.png)

## Compring the results of baseline and snapshot averaging

![image](https://user-images.githubusercontent.com/45852019/210234859-2f8dc744-2e4d-43fb-b00b-27b23b0ae0b2.png)
