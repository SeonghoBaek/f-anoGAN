# f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks
## Tensorflow Implementation of f-AnoGAN ##

Train step

1. Train WGAN
   - python model.py --mode train_gan --model_path YOUR_MODEL_PATH/m.ckpt --train_data YOUR_DATA

2. Train Encoder with fixed WGAN
   - python model.py --mode train_encoder --model_path YOUR_MODEL_PATH/m.ckpt --train_data YOUR_DATA



You should modify model.py,

- Data Preprocessing: Size, Augmentation. 

- stdout message format

- Hyper Parameters

  

ref) https://www.sciencedirect.com/science/article/pii/S1361841518302640
