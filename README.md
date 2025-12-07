# Automatic Image Colorization using Deep Convolutional Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Status](https://img.shields.io/badge/Status-Educational-green)

## About the Project
This project implements a Deep Learning model capable of automatically colorizing grayscale images. It was developed as part of the **Digital Image Analysis** course.

The system uses a **Convolutional Autoencoder** architecture (U-Net based) to predict color channels from a single grayscale input. The model is trained on landscape datasets to learn semantic coloring (e.g., sky is blue, grass is green).

## Methodology: Why CIELAB?
Instead of using the standard RGB color space, this project utilizes the **CIELAB ($L^*a^*b^*$)** color space, which aligns better with human vision and simplifies the task for the neural network.

* **$L$ (Lightness):** The grayscale input (0-100). This is the **input** to the network.
* **$a, b$ (Color Channels):** The green-red and blue-yellow components. These are the **targets** the network learns to predict.

By isolating the lightness information, the model focuses solely on "hallucinating" the missing colors, leading to more stable training compared to RGB regression.

## Key Features
* **Deep CNN Architecture:** Custom autoencoder with bottleneck and upsampling layers.
* **GPU Acceleration:** Full CUDA support for fast training on NVIDIA GPUs.
* **Live Visualization:** Automatically saves comparison images (Grayscale vs. AI Output vs. Ground Truth) after every epoch.
* **Custom Inference:** Script to colorize any user-provided black-and-white photo.



##  Tech Stack
* **Python 3.x**
* **PyTorch** (Neural Network implementation)
* **Scikit-Image & OpenCV** (Image processing and Lab conversion)
* **Matplotlib** (Visualization)