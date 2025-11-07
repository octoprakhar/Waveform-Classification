# 🔊 1D CNN from Scratch: Waveform Classification

## Project Overview

This project implements a **1D Convolutional Neural Network (1D-CNN) from scratch** using only the fundamental Python library **NumPy**. The goal is to accurately classify synthetic time-series data: distinguishing between **Sine Waves (Label 1)** and **Square Waves (Label 0)** across varying frequencies and phases.

The primary focus of this project is to demonstrate a deep, mathematical understanding of neural network mechanics by manually coding the entire model, including the crucial backpropagation steps.

## 🧠 Core Implementation & Achievements

The entire network logic is contained within the `WaveDetectorCNN` class, meticulously defining the forward and backward passes for each layer.

### Technical Highlights:

| Component | Achievement | Insight Demonstrated |
| :--- | :--- | :--- |
| **Convolution 1D** | Implemented the sliding window and multi-filter operation for feature extraction along the time axis. | Mastery of kernel application and feature map generation. |
| **Max Pooling 1D** | Manual implementation of down-sampling the feature map. | Understanding of spatial/temporal data reduction. |
| **Max Pooling Backprop** | Crucial implementation of the **Unpooling** mechanism using a **Binary Mask** (`self.binary_max_pool`). | This is the most complex backpropagation component, proving the ability to handle gradient distribution (only to the max value location). |
| **Full Backpropagation** | Manually computed and chained the gradients (`dl_dz`, `dl_dw`, `dl_db`, etc.) through Softmax, Dense, Flatten, Pooling, ReLU, and Convolution layers. | Deep understanding of the Chain Rule and gradient flow in a CNN. |
| **Wave Generator** | Created a custom data generator to produce two distinct, frequency-varying time-series signals. | Skill in problem formulation and synthetic data preparation. |

***

## ⚙️ Model Architecture & Parameters

The network consists of a single convolutional block followed by a dense classifier.

### Self-Defined Parameters:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `sequence_size` (L) | 100 | Length of the input waveform (time steps). |
| `channel_input` (C) | 1 | Single channel (monophonic signal). |
| `filters` (F) | 8 | Number of convolutional kernels (output feature maps). |
| `kernel_size` (K) | 4 | Size of the 1D convolution window. |
| `pool_size` | 10 | Size of the Max Pooling window. |
| `stride` | 10 | Stride for the Max Pooling operation. |
| `lr` | 0.01 | Learning rate for Stochastic Gradient Descent (SGD). |

***

## 🚧 Current Status & Future Work

The model is structurally sound, but the current configuration often converges to a random guess state.

### Accuracy Note

The current model typically achieves $\mathbf{\sim 50\%}$ accuracy during training. In a binary classification problem, this indicates that the model is performing no better than random chance. This common challenge in "from-scratch" optimization is likely due to:

1.  **Exploding/Vanishing Gradients:** The simplistic $\mathbf{LR=0.01}$ is likely too high for stable convergence in a pure-NumPy environment, causing large, noisy updates.
2.  **Kernel Initialization:** The current initialization may not be optimal for propagating gradients effectively.

### Next Steps & Enhancements

* **Gradient Check:** Implement a formal numerical gradient check (e.g., finite difference approximation) to verify the correctness of the analytical backpropagation for every single layer.
* **Optimization:** Upgrade the basic SGD implementation to a more robust algorithm like **Adam** or **RMSProp**, which manages adaptive learning rates and momentum.
* **Hyperparameter Search:** Systematically test smaller learning rates (e.g., $10^{-3}$, $10^{-4}$) and less aggressive pooling settings (e.g., `pool_size=5`, `stride=2`).

***

## 🚀 How to Run

### Prerequisites

* Python (3.x)
* `numpy`
* `matplotlib` (for data visualization)
* `scikit-learn` (for evaluation metrics)

### Execution

1.  Download or clone the repository containing `sinVsSquarewave.ipynb`.
2.  Open the Jupyter Notebook: `sinVsSquarewave.ipynb`.
3.  Run all cells. The notebook will:
    * Generate the synthetic wave data.
    * Initialize the `WaveDetectorCNN` model.
    * Train the model using Stochastic Gradient Descent.
    * Evaluate performance on the test set.
