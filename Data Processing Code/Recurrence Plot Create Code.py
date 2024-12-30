import numpy as np
import pandas as pd
from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt

files = ["Vibration","Current_R","Current_S","Current_T"]

for file in files:
    csv_file = f'Test_{file}_Norm(FS 2000Hz, 1s).csv'
    data = pd.read_csv(csv_file, header=None).to_numpy()
    print(data.shape)

    # Setting parameters
    window_size = 20  # Sliding window size
    overlap_ratio = 0 # Overlap ratio
    stride = int(window_size * (1 - overlap_ratio))  # Movement interval
    time_steps = 35  # Number of time steps

    # Divide into 35 sections using sliding window
    num_samples, total_length = data.shape
    reshaped_data = np.zeros((num_samples, time_steps, window_size))
    num_samples = 1

    for sample_idx in range(num_samples):
        for timestep_idx in range(time_steps):
            start_idx = timestep_idx * stride
            end_idx = start_idx + window_size
            reshaped_data[sample_idx, timestep_idx, :] = data[sample_idx, start_idx:end_idx]

    # Initialize the recurrence plot converter
    rp = RecurrencePlot()

    # Initialize the array of recurrence plot storage
    rp_data = np.zeros((num_samples, time_steps, window_size, window_size), dtype=np.float32)

    # Generate recurrence plots
    for sample_idx in range(num_samples):
        for timestep_idx in range(time_steps):
            current_sequence = reshaped_data[sample_idx, timestep_idx]
            rp_matrix = rp.transform(current_sequence.reshape(1, -1))
            rp_data[sample_idx, timestep_idx] = rp_matrix

        # Progress status output
        if sample_idx % 1000 == 0:
            print(f"{sample_idx}/{num_samples} Sample processing complete")

    # Save results as .npy file
    output_file = f'Recurrence_{file}'
    np.save(output_file, rp_data)

    print("Recurrence plot processing and saving completed.")
    print("Final result shape:", rp_data.shape)

    # Visualizing the results
    # As an example, visualize the first timestep recurrence plot of the first sample
    plt.figure(figsize=(6, 6))
    plt.imshow(rp_data[0, 0], cmap='Blues', interpolation='none')  # First sample, first timestep
    plt.axis('off')
    plt.savefig(f'Recurrence_{file}_[0,0].png', bbox_inches='tight', pad_inches=0)
    plt.show()
