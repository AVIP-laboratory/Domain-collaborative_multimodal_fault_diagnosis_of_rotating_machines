# Domain-collaborative multimodal fault diagnosis

Rotating machines are essential in industrial applications, but fault diagnosis under noisy conditions remains a challenge due to the complexity and diversity of fault types. This study introduces a **Domain-collaborative multimodal transformer** network leveraging time-frequency segmentation to improve robustness and accuracy in noisy environments.

## Key Features
- **Multimodal Input:** Vibration and current signals are converted into tensor representations of time and frequency domains using recurrence plots.
- **Hybrid Architecture:** Combines convolutional layers, bidirectional LSTM, and transformer modules for comprehensive feature extraction and analysis.
- **Self- and Cross-Attention Mechanisms:** Dynamically captures sequential dependencies and global correlations across domains.
- **Real-World Validation:** Validated on datasets from subway station motors under varying fault conditions and noise levels, achieving high classification accuracy even in extreme noise.
- **Ablation Study Insights:** Demonstrates the critical roles of the number of transformers, cross-attention, and bidirectional LSTM layers in boosting network performance.

## Dataset

The dataset covers four fault modes of rotating machines, including the normal state:
- Shaft misalignment
- Bearing failure
- Belt sagging
- Rotor imbalance

### Data Overview:
The dataset includes vibration and current data from three phases(R,S,T), with fault classifications for each.

- **Training Data:** 45,000 samples per phase (vibration and current data from each phase)
- **Test Data:** 5,000 samples per phase (vibration and current data from each phase)

## Recurrence Plot (RP) Creation

The four Recurrence Plot (RP) images generated for time and the four RP images generated for frequency are merged as channels, and the resulting data is stored as a numpy array.

### RP Creation Code
The code used for generating the Recurrence Plot (RP) can be found at the following location:

- **Code for RP Creation**: `/Data Processing Code/Recurrence Plot Create Code.py`

### RP Example Images
The RP example images generated from both time and frequency can be found at the following location:

- **RP Images**: `/RP example image`

### Final Data Shape:
- **Training Data:** 45,000 samples with shape `(45000, 35, 20, 20, 4)`
- **Test Data:** 5,000 samples with shape `(5000, 35, 20, 20, 4)`

The training model uses the following input data:

- **Time Train Data**: Shape `(45000, 35, 20, 20, 4)`
- **Frequency Train Data**: Shape `(45000, 35, 20, 20, 4)`

## Model Training and Testing Code

The main code for the **Domain-Collaborative Multimodal Fault Diagnosis** project is named **'Cross #6'**. Additionally, the table below summarizes the ablation study conducted on different model configurations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/32200802-216c-4d2a-aa06-ef802ad0d396" alt="Image" />
</p>

The code for both training and testing the model is organized within the **Codes** directory. Each model consists of two main components:

- **JOB1**: corresponds to the training code.
- **JOB2**: corresponds to the testing code, which also includes the noise testing.

## Model Results

The following details summarize the results from the **Domain-Collaborative Multimodal Fault Diagnosis** project. Each model's training history, noise test results, and associated `.h5` file are provided.

- **Model Result Files**: Located in `/Model results/Model name`

Inside each directory, you can find:
  - The model's `.h5` file
  - Training history graphs
  - Confusion matrices

For additional details, you can explore the respective files in the specified directory.




