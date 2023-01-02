# Does image resolution impact fine-grained Tuberculosis-consistent lesion segmentation performance in chest X-rays?

To study the impact of varying spatial resolution and aspect ratio adjustments toward segmenting TB-consistent lesions in chest X-rays

## Objective:

Dataset: Shenzhen TB CXR dataset and their corresponding pixel-wise TB consistent lesion masks (287 images, 201/29/57- train/Val/Test split). 

The best-performing Inception-V3-based UNet mode from our previous study https://www.mdpi.com/2306-5354/9/9/413 is selected.

The model performance is analyzed under the following settings:

    Direct resizing with original data (at resolutions 32×32, 64×64, 128×128, 256×256, 512×512, 768×768, and 1024×1024)

    Direct resizing with lung-cropping at afore-mentioned resolutions.

    Aspect-ratio adjusted original data at afore-mentioned resolutions.

    Aspect-ratio adjusted lung-cropped data at afore-mentioned resolutions.

Improve segmentation performance at the optimal resolution/ratio through a combinatorial approach of storing model snapshots, optimal test time augmentation, and segmentation threshold optimization.


![image](https://user-images.githubusercontent.com/45852019/210234614-ba1aee0b-a679-46db-a388-82b2e9faf49a.png)


![image](https://user-images.githubusercontent.com/45852019/210234673-2c7f0c9a-9746-4b19-bc69-d54eaaadf233.png)


![image](https://user-images.githubusercontent.com/45852019/210234712-24cae50f-9205-47e8-993e-b499e1543289.png)


## Test-time augmentation combination

![image](https://user-images.githubusercontent.com/45852019/210234837-b6297c06-24a6-4071-be09-c6c13a882ca6.png)

## Compring the results of baseline and snapshot averaging

![image](https://user-images.githubusercontent.com/45852019/210234859-2f8dc744-2e4d-43fb-b00b-27b23b0ae0b2.png)
