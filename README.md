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

## Multimodal Data Creation

The time-series signal is divided into 35 time steps, each with 20 sequential data points, and four Recurrence Plot (RP) images are generated for each time step, one for each type of time-series data. Similarly, the frequency data is divided into 35 frequency regions, each covering a 20 Hz range, with four RP images generated for each frequency segment.

The time and frequency RP images are merged by each channel and stored.

### RP Creation Code
The code used for generating the RP can be found at the following location:

- **Code for RP Creation**: `/Data Processing Code/Recurrence Plot Create Code.py`

### RP Example Images
The RP example images generated from both time and frequency can be found at the following location:

- **RP Images**: `/RP example image`

### Final Data Shape:

The training model uses the following input data:

- **Training Data:**
  - **Time Train Data**: 45,000 time samples with shape `(45000, 35, 20, 20, 4)`
  - **Frequency Train Data**: 45,000 FFT samples with shape `(45000, 35, 20, 20, 4)`
    
- **Test Data:**
  - **Time Test Data**: 5,000 time samples with shape `(5000, 35, 20, 20, 4)`
  - **Frequency Test Data**: 5,000 FFT samples with shape `(5000, 35, 20, 20, 4)`
  
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

### Noise test result

The noise levels NL<sub>0</sub>, NL<sub>1</sub>, NL<sub>2</sub>, NL<sub>3</sub>, NL<sub>4</sub>, NL<sub>5</sub> and NL<sub>6</sub> were set to 22.8, -16.9, -13.3, -10.8, -8.8, and -7.3 dB (ref=1), respectively. The model accuracy results for each noise level are presented in the table.

<p align="center">
  <img src="https://github.com/user-attachments/assets/32a0b6f8-8f75-42e6-a6cb-e8a9ffe8c6ff" alt="Image" />
</p>


For additional details, you can explore the respective files in the specified directory.




