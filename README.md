# Surface Normal Estimation of Tilted Images via Spatial Rectifier

# Abstract

We present a spatial rectifier to estimate surface normal of tilted images. 
Tilted images are of particular interest as more visual data are captured by embodied sensors such as body-/robot-mounted cameras. Existing approaches exhibit bounded performance on predicting surface normal because they were trained by gravity-aligned images. 
Our two main hypotheses are: (1) visual scene layout is indicative of the gravity direction; and 
(2) not all surfaces are equally represented, i.e., there exists a transformation of the tilted image to produce better surface normal prediction. 
We design the spatial rectifier that is learned to transform the surface normal distribution of the tilted image such that it matches to that of gravity aligned images. 
The spatial rectifier is parametrized by a principle direction of the rectified image that maximizes the distribution match. 
When training, we jointly learn the gravity and principle direction that can synthesize images and surface normal to supervise the surface normal estimator. 
Inspired by the panoptic feature pyramid network, a new network that can access to both global and local visual features is proposed by leveraging dilated convolution in the decoder. The resulting estimator produces accurate surface normal estimation, outperforming state-of-the-art methods and data augmentation baseline. 
We evaluate our method not only on ScanNet and NYUv2 but also on a new dataset called Tilt-RGBD that includes substantial roll and pitch camera motion captured by body-mounted cameras.

# Installation Guide
For convenience, all the code in this repositority are assumed to be run inside NVIDIA-Docker. 

### For instructions on installing NVIDIA-Docker, please follow the following steps (note that this is for Ubuntu 18.04):

For more detailed instructions, please refer to [this link](https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/).
1. Install Docker

    ```
    sudo apt-get update
    
    sudo apt-get install \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common
        
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    
    sudo apt-key fingerprint 0EBFCD88
    
    sudo add-apt-repository \
       "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"
    
    sudo apt-get update
    
    sudo apt-get install docker-ce docker-ce-cli containerd.io
    ```
    
    To verify Docker installation, run:

    ```
    sudo docker run hello-world
    ```

2. Install NVIDIA-Docker

    ```
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list
      
    sudo apt-get update
    
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -
    
    sudo apt-get install nvidia-docker2
    
    sudo pkill -SIGHUP dockerd
    ```

To activate the docker environment, run the following command:

```
nvidia-docker run -it --rm --ipc=host -v /:/home nvcr.io/nvidia/pytorch:19.08-py3
```

where `/` is the directory in the local machine (in this case, the root folder), and `/home` is the reflection of that directory in the docker. 
This has also specified NVIDIA-Docker with PyTorch version 19.08 which is required to ensure the compatibility 
between the packages used in the code (at the time of submission).

Inside the docker, change the working directory to this repository: 
```
cd /home/PATH/TO/THIS/REPO/eccv2020-1886-release
```

# Datasets and pretrained models

### Datasets

We only provide NYUv2 and Tilt-RGBD datasets for evaluation purpose. For more details on downloading ScanNet dataset which contains ground-truth surface normal computed by FrameNet, 
please refer to [FrameNet's repo](https://github.com/hjwdzh/FrameNet/tree/master/src).

Please run the following commands to automatically download our provided datasets:

```
wget -nv -O nyu-normal.zip https://tilted-image-normal.s3.amazonaws.com/nyu-normal.zip && unzip -q nyu-normal.zip -d ./datasets/ && rm -rf nyu-normal.zip

wget -nv -O KinectAzure.zip https://tilted-image-normal.s3.amazonaws.com/KinectAzure.zip && unzip -q KinectAzure.zip -d ./datasets/ && rm -rf KinectAzure.zip
```

### Datasets splits
In order to obtain the exact dataset splits that we used for training/testing, please run the following command to download the `.pkl` files:

```
wget -nv -O data.zip https://tilted-image-normal.s3.amazonaws.com/data.zip && unzip -q data.zip -d ./data/ && rm -rf data.zip
```

### Pretrained models

We provide the checkpoints for all the experimental results reported in the paper with different combinations of 
network architecture, loss function, method, and training dataset. 

```
wget -nv -O checkpoints.zip https://tilted-image-normal.s3.amazonaws.com/checkpoints.zip && unzip -q checkpoints.zip -d ./checkpoints/ && rm -rf checkpoints.zip
```


# Quick Inference

Please follow the below steps to extract surface normals from some RGB images using our provided pre-trained model:
1. Make sure you have the following `.ckpt` file inside [`./checkpoints/`](./checkpoints) folder: 
`DFPN_TAL_SR.ckpt`.
You can also use this command to download ONLY this checkpoint:

```
wget -nv -O DFPN_TAL_SR.ckpt https://tilted-image-normal.s3.amazonaws.com/DFPN_TAL_SR.ckpt
```

2. Download our demo RGB images:

```
wget -nv -O demo_dataset.zip https://tilted-image-normal.s3.amazonaws.com/demo_dataset.zip && unzip -q demo_dataset.zip && rm -rf demo_dataset.zip
```

3. Run [`inference_script.sh`](./inference_script.sh) to extract the results in [`./demo_results/`](./demo_results).


# Benchmark Evaluation
We evaluate surface normal estimation on ScanNet, NYUD-v2, or Tilt-RGBD with different network architectures using our provided pre-trained models.

Run:
```
sh test_script.sh
```


Specifically, inside the bash script, multiple arguments are needed, including the path to the pre-trained model, batch size, network architecture, and test dataset (ScanNet/NYUv2/Tilt-RGBD).
Please refer to the actual code for the exact supported arguments options.

**(Note: make sure you specify the correct network architecture for your pretrained model)**

```
# Evaluation for DFPN+TAL
## Tilted Images on Tilt-RGBD
python train_test_generalized_surface_normal.py \
                     --checkpoint_path 'PATH_TO_PRETRAINED_MODEL' \
                     --operation 'evaluate' \
                     --test_dataset 'TEST_DATASET' \
                     --net_architecture 'NETWORK_ARCHITECTURE' \
                     --batch_size BATCH_SIZE
```

# Training

We train our surface normal estimation network on ScanNet dataset. 

Note that the train/test splits used for training the full spatial rectifier pipeline is not standard as we 
intentionally chose some specific parts of ScanNet in order to help the training.

As specified in [`train_script.sh`](./train_script.sh), the training pipeline consists of two stages: 

1) Continue training the canonical-view surface normal estimation network by warping the input images towards the precomputed principle direction. 

2) Using the above canonical-view surface normal estimation network and the pretrained spatial rectifier, start training the full pipeline end-to-end.


Specifically, inside the bash script, multiple arguments are needed, including the folder containing log files and checkpoints,
type of loss functions (L2, AL, or TAL), learning rate, batch size, network architecture, training/testing/validation datasets (ScanNet/NYUv2/Tilt-RGBD). 
Please refer to the actual code for the exact supported arguments options.






