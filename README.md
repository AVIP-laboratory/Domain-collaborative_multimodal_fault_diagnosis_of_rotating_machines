# Domain-collaborative multimodal fault diagnosis of rotating machines

Rotating machines are essential in industrial applications, but fault diagnosis under noisy conditions remains a challenge due to the complexity and diversity of fault types. This study introduces a **multimodal transformer network** leveraging time-frequency separation to improve robustness and accuracy in noisy environments.

## Key Features
- **Multimodal Input:** Vibration and current signals are converted into tensor representations of time and frequency domains using recurrence plots.
- **Hybrid Architecture:** Combines convolutional layers, bidirectional LSTM, and transformer modules for comprehensive feature extraction and analysis.
- **Self- and Cross-Attention Mechanisms:** Dynamically captures sequential dependencies and global correlations across domains.
- **Real-World Validation:** Validated on datasets from subway station motors under varying fault conditions and noise levels, achieving high classification accuracy even in extreme noise.
- **Ablation Study Insights:** Demonstrates the critical roles of the number of transformers, cross-attention, and bidirectional LSTM layers in boosting network performance.

## Dataset

The dataset was collected from 21 rotating machines with different electrical power capacities and types. These machines were installed at three subway stations: Daejeon Station (9 motors), Gapcheon Station (3 motors), and City Hall Station (9 motors). The machines have six different power capacities: 2.2 kW, 3.7 kW, 5.5 kW, 7.5 kW, 11 kW, and 15 kW, and include various types of motors. The machines serve for supply, exhaust, and return air purposes and are located in various areas such as waiting rooms, utility rooms, HVAC rooms, and platforms.

The dataset covers four fault modes of rotating machines:
- Shaft misalignment
- Bearing fault
- Belt slippage
- Rotor imbalance

### Data Collection:
- **Vibration signals** were collected using an **accelerometer** (PCB Piezotronics, 352C34).
- **Current RST signals** were collected using a **sensor** (Fine-trans, FS20L8).

### Data Overview:
- **Training Data:** 45,000 samples
- **Test Data:** 5,000 samples

### Data Characteristics:
1. **Time Domain Data:**
   - The vibration data has a sampling frequency of 4000 Hz and a duration of 2 seconds, while the current data has a sampling frequency of 2000 Hz and a duration of 2 seconds.
   - **Vibration and 3-phase Current Data:**  `/Time Domain Data`
        
3. **Downsampled & Normalized Data:**
   - **Vibration and 3-phase Current Data (FS 2000 Hz, 1-second)**: `/Downsampled & Normalized Data`

4. **Frequency Domain Data (FFT):**
   - **Vibration and 3-phase Current Data (~900 Hz, FFT)**: `/Frequency Domain Data (FFT)`

## Noise Data

The noise levels were added to the test data mentioned earlier (vibration and 3-phase current data). The noise levels are divided into six categories. Initially, 2000 random values uniformly distributed between -0.5 and 0.5 were generated. These random values were then multiplied by coefficients to control the magnitude of the six noise levels. The coefficients range from 0.25 to 1.5 in steps of 0.25.

For each noise level, four types of data (V, R, S, T for vibration and 3-phase current data) are available, as well as their corresponding FFT data(~900Hz).

The noise levels and corresponding data files are as follows:

- **Noise Level 1** (\(NL_1 = -22.8\) dB): 
  - Time data: `/Noise Data/Noise Level 1/Time data`
  - FFT data: `/Noise Data/Noise Level 1/FFT data`
  
- **Noise Level 2** (\(NL_2 = -16.9\) dB): 
  - Time data: `/Noise Data/Noise Level 2/Time data`
  - FFT data: `/Noise Data/Noise Level 2/FFT data`

- **Noise Level 3** (\(NL_3 = -13.3\) dB): 
  - Time data: `/Noise Data/Noise Level 3/Time data`
  - FFT data: `/Noise Data/Noise Level 3/FFT data`

- **Noise Level 4** (\(NL_4 = -10.8\) dB): 
  - Time data: `/Noise Data/Noise Level 4/Time data`
  - FFT data: `/Noise Data/Noise Level 4/FFT data`

- **Noise Level 5** (\(NL_5 = -8.8\) dB): 
  - Time data: `/Noise Data/Noise Level 5/Time data`
  - FFT data: `/Noise Data/Noise Level 5/FFT data`

- **Noise Level 6** (\(NL_6 = -7.3\) dB): 
  - Time data: `/Noise Data/Noise Level 6/Time data`
  - FFT data: `/Noise Data/Noise Level 6/FFT data`
 
## Recurrence Plot (RP) Creation

Each time series and frequency signal is divided into 35 time steps, where each step consists of 20 sequential data points. The generated 4 RP images are merged as channels, and the resulting npy data is stored.

### RP Creation Code
The code used for generating the Recurrence Plot (RP) can be found at the following location:

- **Code for RP Creation**: `/Data Processing Code/Recurrence Plot Create Code.py`

### RP for Time Data
The RP images generated from the time series data are stored at:

- **RP Image Data (Time Domain)**: `/RP npy Data/Time Series`

### RP for Frequency Data
The RP images generated from the frequency domain data (FFT) are stored at:

- **RP Image Data (Frequency Domain)**: `/RP npy Data/Frequency Domain`

### Final Data Shape:
- **Training Data:** 45,000 samples with shape `(45000, 35, 20, 20, 4)`
- **Test Data:** 5,000 samples with shape `(5000, 35, 20, 20, 4)` including noise data

## Model Training and Testing Code

The code used for both training and testing the model is located at the following path:

- **Model Training and Testing Code**: `path/to/model_training_and_testing_code`

In this file, **JOB1** corresponds to the training code, and **JOB2** corresponds to the testing code, which also includes the noise testing.
