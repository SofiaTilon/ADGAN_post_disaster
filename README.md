# ADGAN post disaster

### 1. Folder structure

├───data

│   └───processed

│       └───GAP

│           ├───test_anom

│			├───test_norm

│			├───train_norm

│			└───val_norm

├───notebooks

└───src

    └───models
    
        └───skip-ganomaly
        
            ├───experiments
            
            ├───lib
            
            │   ├───data
            
            │   └───models
            
            └───output


### 2. Data

10 percent of the German Asphalt Pavement (GAP) Distress dataset v1.0 is located in /data/processed/GAP.
Data is stored in "chunks", each chunk contains 32000 samples of 64x64 pixels. 
Data source: https://www.tu-ilmenau.de/en/neurob/data-sets-code/gaps/.
Citation: M. Eisenbach et al., “How to get pavement distress detection ready for deep learning? A systematic approach,” Proc. Int. Jt. Conf. Neural Networks, vol. 2017-May, pp. 2039–2047, 2017.

### 3. Model

Code adapted from https://github.com/samet-akcay/skip-ganomaly. Source: S. Akçay, A. Atapour-Abarghouei, and T. P. Breckon, “Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection,” 2019.

This model is based on Generative Adversarial Networks. GANs works as follows: one model, the Generator, aims to generate images that are indistinctable from real images. A second model, the Discriminator, aims to differentiate the fake image from the real image. The Generator aims to increase the Discriminator's loss while the Discriminator aims to decrease its loss. 

Both models are trained in conjunction and should become better over time. Different from traditional CNNs, GANs are only trained on "normal" (undamaged) patches.
In skip-ganomaly the networks are defined in /src/models/skip-ganomaly/lib/models/networks.py. Here is where you can adapt the Convolutional layers. 

##### 3.1 GANS for anomaly detection.

We assume that when the trained generator receives an anomalous input image, the reconstruction will be bad, and the difference between real and generated image will be large. 

The input images can be therefore be scored. Large scores correlate to abnormal patches while low scores correlate to normal patches. 

#### 3.2 How to run model

It is recommended to train skip-ganomaly on GPU. 

To run model using default settings (while in skip-ganomaly directory):
> python train.py

However, to run GAP, you have to specify the datapaths. Example:
> CUDA_VISIBLE_DEVICES=1 python train.py --dataset GAP --dataroot /data/tilonsm/Documents/RoadSurfaceDamageUAV/data/processed/GAP --ngpu 1

Other run options I have used (and which I would recommend) are:
> CUDA_VISIBLE_DEVICES=1 python train.py --dataset GAP --isize 64 --niter 10 --dataroot /data/tilonsm/Documents/RoadSurfaceDamageUAV/data/processed/GAP --nc 1 --ngpu 1 --outf /data/tilonsm/Documents/RoadSurfaceDamageUAV/src/models/skip-ganomaly/output --name [NAMEOFEXPERIMENT] --print_freq 50 --save_image_freq 8000 --save_test_images

To run a hyperparameter tuning experiment, run (takes a long time):
> ./experiments/run_GAP.sh 

You could also run the following hyperparameter settings which I have found to perform the best so far:
> CUDA_VISIBLE_DEVICES=1 python train.py --dataset GAP --isize 64 --niter 10 --w_adv 30 --w_con 1 --w_lat 5 --nz 256 --extralayers 10 --dataroot /data/tilonsm/Documents/RoadSurfaceDamageUAV/data/processed/f-anogan3 --nc 1 --ngpu 1 --outf /data/tilonsm/Documents/RoadSurfaceDamageUAV/src/models/skip-ganomaly/output --name [NAMEOFEXPERIMENT] --print_freq 50 --save_image_freq 8000 --save_test_images

After each epoch, the model is evaluated using the test data and the AUC scores are stored. 

#### 3.3 How to evaluate the model

Scores are stored in models/output/[NAMEOFEXPERIMENT]/train/anom/scores.pkl
I evaluate these scores using jupyter notebook. This file is located under /notebooks/.
