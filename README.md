# 🧠 MNIST Convolutional Autoencoder

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-FF0000?logo=pytorch&logoColor=white)](https://pytorch.org/)  
[![Dataset: MNIST](https://img.shields.io/badge/Dataset-MNIST-gray?logo=data:image/png;base64,)](http://yann.lecun.com/exdb/mnist/)  

---

## 🤖 Project Overview  
This project implements a **Convolutional Autoencoder** trained on the MNIST handwritten‐digits dataset. The goal is to learn compact latent representations of the input images and to reconstruct them with minimal loss — enabling unsupervised feature learning or dimensionality reduction in computer-vision tasks.

---

## ⚙️ Key Features  
- ✅ Encoder–Decoder architecture using convolutional and transpose‐convolution layers  
- ✅ Training on the MNIST dataset (28×28 grayscale images of digits)  
- ✅ Latent space exploration: compress & visualize encodings  
- ✅ Reconstruction examples: original vs autoencoder output  
- ✅ Easily extendable to denoising autoencoder, variational autoencoder (VAE), or other image datasets  

---

## 🧩 Architecture  
```text
Input (28×28 grayscale image)  
  → Conv2D layers (feature extraction)  
  → Bottleneck / Latent Vector  
  → ConvTranspose2D layers (reconstruction)  
  → Output (28×28 grayscale image)
```

## 📊 Tech Stack

| Category    | Tools                                        |
| ----------- | -------------------------------------------- |
| Language    | Python 3.x                                   |
| Framework   | PyTorch                                      |
| Dataset     | Fashion-MNIST                                |
| Optimizers  | SGD · Adam · AdamW · Shampoo                 |
| Use-Case    | Optimization Research · Autoencoders         |


## 🔧 Installation & Setup
Clone the repository:
git clone https://github.com/NirEllor/MNIST-Convolutional-Autoencoder.git
cd MNIST-Convolutional-Autoencoder

(Optional) create virtual environment:
python3 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run:
Example: train the autoencoder
python train_autoencoder.py --epochs 50 --batch_size 128

Example: visualize reconstruction results
python visualize_reconstruction.py --model_path models/ae.pth


## 📥 Dataset
The code uses the MNIST dataset, which will be automatically downloaded if not available locally via torchvision. 

## 📋 Expected Output

Reconstruction examples where input digits (e.g., “5”) become output images visually close to original
Latent vector size (e.g., 32 or 64 dimensions)
Training loss decreases steadily (e.g., MSE or BCE loss)

## 👨‍💻 Author
Nir Ellor
Full-Stack Web3 & AI Developer




