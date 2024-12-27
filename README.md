# Domain-collaborative multimodal fault diagnosis

Rotating machines are essential in industrial applications, but fault diagnosis under noisy conditions remains a challenge due to the complexity and diversity of fault types. This study introduces a **Domain-collaborative multimodal transformer** network leveraging time-frequency segmentation to improve robustness and accuracy in noisy environments.

## Key Features
- **Multimodal Input:** Vibration and current signals are converted into tensor representations of time and frequency domains using recurrence plots.
- **Hybrid Architecture:** Combines convolutional layers, bidirectional LSTM, and transformer modules for comprehensive feature extraction and analysis.
- **Self- and Cross-Attention Mechanisms:** Dynamically captures sequential dependencies and global correlations across domains.
- **Real-World Validation:** Validated on datasets from subway station motors under varying fault conditions and noise levels, achieving high classification accuracy even in extreme noise.
- **Ablation Study Insights:** Demonstrates the critical roles of the number of transformers, cross-attention, and bidirectional LSTM layers in boosting network performance.

## Dataset

The dataset covers four fault modes of rotating machines:
- Shaft misalignment
- Bearing fault
- Belt slippage
- Rotor imbalance

### Data Overview:
- **Training Data:** 45,000 samples
- **Test Data:** 5,000 samples

## Recurrence Plot (RP) Creation

Each time series and frequency signal is divided into 35 time steps, where each step consists of 20 sequential data points. The generated 4 RP images are merged as channels, and the resulting npy data is stored.

### RP Creation Code
The code used for generating the Recurrence Plot (RP) can be found at the following location:

- **Code for RP Creation**: `/Data Processing Code/Recurrence Plot Create Code.py`

### Final Data Shape:
- **Training Data:** 45,000 samples with shape `(45000, 35, 20, 20, 4)`
- **Test Data:** 5,000 samples with shape `(5000, 35, 20, 20, 4)`

## Model Training and Testing Code

The code used for both training and testing the model is located at the following path:

- **Model Training and Testing Code**: `path/to/model_training_and_testing_code`

In this file, **JOB1** corresponds to the training code, and **JOB2** corresponds to the testing code, which also includes the noise testing.
