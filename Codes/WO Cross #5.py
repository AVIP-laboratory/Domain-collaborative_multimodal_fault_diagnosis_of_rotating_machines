import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import optimizers, layers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Conv2D, Input, MaxPool2D, GlobalAveragePooling2D, Dense, Concatenate,
                                     ZeroPadding2D, LSTM, TimeDistributed, BatchNormalization, ReLU,
                                     Add, LayerNormalization, Bidirectional, Dropout, GlobalAveragePooling1D,
                                     MultiHeadAttention, Reshape, Conv1D, MaxPool1D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# job 1 = Model Training
# job 2 = Model Testing

job = 1

Modle_name = 'W/O Cross #5'

Data_path = 'Project Data Path1'
Result_path = 'Project Data Path2'

if job == 1:
    # Load Data
    Time_data = np.load(f'{Data_path}/Train_time_data.npy', allow_pickle=True)
    FFT_data = np.load(f'{Data_path}/Train_FFT_data.npy', allow_pickle=True)
    Y_train = np.load(f'{Data_path}/Y_train.npy')
    print(f"Time_data shape : {Time_data.shape}\n"
          f"FFT_data shape : {FFT_data.shape}\n"
          f"Y_train shape : {Y_train.shape}")

    ## Time-ConvBlock1 & BiLSTM------------------------------------------------------------------------------------
    Input_time = Input(shape=(35, 20, 20, 4))
    layer_A = TimeDistributed(Conv2D(8, (3, 3), padding="same", activation='relu'))(Input_time)
    layer_A = TimeDistributed(Conv2D(8, (2, 2), padding="same", activation='relu', strides=2))(layer_A)
    SKIP_A1 = TimeDistributed(Conv2D(16, (1, 1), padding="same"))(layer_A)

    layer_A = TimeDistributed(Conv2D(16, (3, 3), padding="same"))(layer_A)
    CONNECTION_A1 = SKIP_A1 + layer_A
    layer_A = TimeDistributed(ReLU())(CONNECTION_A1)
    layer_A = TimeDistributed(Conv2D(16, (2, 2), padding="same", activation='relu', strides=2))(layer_A)
    SKIP_A2 = TimeDistributed(Conv2D(32, (1, 1), padding="same"))(layer_A)

    layer_A = TimeDistributed(Conv2D(32, (3, 3), padding="same"))(layer_A)
    CONNECTION_A2 = SKIP_A2 + layer_A
    layer_A = TimeDistributed(ReLU())(CONNECTION_A2)
    layer_A = TimeDistributed(Conv2D(32, (2, 2), padding="same", activation='relu', strides=2))(layer_A)

    layer_A = TimeDistributed(GlobalAveragePooling2D())(layer_A)
    layer_A = Bidirectional(LSTM(32, return_sequences=True))(layer_A)

    ## FFT-ConvBlock1 & BiLSTM--------------------------------------------------------------------------------------
    Input_FFT = Input(shape=(35, 20, 20, 4))
    layer_B = TimeDistributed(Conv2D(8, (3, 3), padding="same", activation='relu'))(Input_FFT)
    layer_B = TimeDistributed(Conv2D(8, (2, 2), padding="same", activation='relu', strides=2))(layer_B)
    SKIP_B1 = TimeDistributed(Conv2D(16, (1, 1), padding="same"))(layer_B)

    layer_B = TimeDistributed(Conv2D(16, (3, 3), padding="same"))(layer_B)
    CONNECTION_B1 = SKIP_B1 + layer_B
    layer_B = TimeDistributed(ReLU())(CONNECTION_B1)
    layer_B = TimeDistributed(Conv2D(16, (2, 2), padding="same", activation='relu', strides=2))(layer_B)
    SKIP_B2 = TimeDistributed(Conv2D(32, (1, 1), padding="same"))(layer_B)

    layer_B = TimeDistributed(Conv2D(32, (3, 3), padding="same"))(layer_B)
    CONNECTION_B2 = SKIP_B2 + layer_B
    layer_B = TimeDistributed(ReLU())(CONNECTION_B2)
    layer_B = TimeDistributed(Conv2D(32, (2, 2), padding="same", activation='relu', strides=2))(layer_B)

    layer_B = TimeDistributed(GlobalAveragePooling2D())(layer_B)
    layer_B = Bidirectional(LSTM(32, return_sequences=True))(layer_B)

    ## Time Transformer Self-Attention------------------------------------------------------------------------------
    MH_A1 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(layer_A, layer_A)
    MH_A1 = layer_A + MH_A1
    FF_A1 = Dense(256, activation='relu')(MH_A1)
    FF_A1 = Dense(64)(FF_A1)
    FF_A1 = Dropout(0.1)(FF_A1)
    FF_A1 = MH_A1 + FF_A1

    MH_A2 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(FF_A1, FF_A1)
    MH_A2 = FF_A1 + MH_A2
    FF_A2 = Dense(256, activation='relu')(MH_A2)
    FF_A2 = Dense(64)(FF_A2)
    FF_A2 = Dropout(0.1)(FF_A2)
    FF_A2 = MH_A2 + FF_A2

    # MH_A3 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(FF_A2, FF_A2)
    # MH_A3 = FF_A2 + MH_A3
    # FF_A3 = Dense(256, activation='relu')(MH_A3)
    # FF_A3 = Dense(64)(FF_A3)
    # FF_A3 = Dropout(0.1)(FF_A3)
    # FF_A3 = MH_A3 + FF_A3

    ## Spectral Transformer Self-Attention--------------------------------------------------------------------------
    MH_B1 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(layer_B, layer_B)
    MH_B1 = layer_B + MH_B1
    FF_B1 = Dense(256, activation='relu')(MH_B1)
    FF_B1 = Dense(64)(FF_B1)
    FF_B1 = Dropout(0.1)(FF_B1)
    FF_B1 = MH_B1 + FF_B1

    MH_B2 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(FF_B1, FF_B1)
    MH_B2 = FF_B1 + MH_B2
    FF_B2 = Dense(256, activation='relu')(MH_B2)
    FF_B2 = Dense(64)(FF_B2)
    FF_B2 = Dropout(0.1)(FF_B2)
    FF_B2 = MH_B2 + FF_B2

    # MH_B3 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(FF_B2, FF_B2)
    # MH_B3 = FF_B2 + MH_B3
    # FF_B3 = Dense(256, activation='relu')(MH_B3)
    # FF_B3 = Dense(64)(FF_B3)
    # FF_B3 = Dropout(0.1)(FF_B3)
    # FF_B3 = MH_B3 + FF_B3

    # ## Time-Spectral Transformer Cross-Attention--------------------------------------------------------------------
    # MH_C1 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(layer_A, layer_B)
    # MH_C1 = layer_A + MH_C1
    # FF_C1 = Dense(256, activation='relu')(C1)
    # FF_C1 = Dense(64)(FF_C1)
    # FF_C1 = Dropout(0.1)(FF_C1)
    # FF_C1 = MH_C1 + FF_C1
    #
    # MH_C2 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(FF_C1, FF_C1)
    # MH_C2 = FF_C1 + MH_C2
    # FF_C2 = Dense(256, activation='relu')(MH_C2)
    # FF_C2 = Dense(64)(FF_C2)
    # FF_C2 = Dropout(0.1)(FF_C2)
    # FF_C2 = MH_C2 + FF_C2
    #
    # # MH_C3 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(FF_C2, FF_C2)
    # # MH_C3 = FF_C2 + MH_C3
    # # FF_C3 = Dense(256, activation='relu')(MH_C3)
    # # FF_C3 = Dense(64)(FF_C3)
    # # FF_C3 = Dropout(0.1)(FF_C3)
    # # FF_C3 = MH_C3 + FF_C3

    ## Time-ConvBlock2----------------------------------------------------------------------------------------------
    Time_A1 = Conv1D(64, 3, padding="same", strides=2)(FF_A2)
    Time_A2 = Conv1D(64, 2, padding="same")(Time_A1)

    Time_A_CONNECTION1 = Time_A1 + Time_A2
    Time_A_CONNECTION1 = ReLU()(Time_A_CONNECTION1)
    Time_A3 = Conv1D(64, 3, padding="same", strides=2)(Time_A_CONNECTION1_Con1)
    Time_A4 = Conv1D(64, 2, padding="same")(Time_A3)

    Time_A_CONNECTION2 = Time_A3 + Time_A4
    Time_A_CONNECTION2 = ReLU()(Time_A_CONNECTION2)
    Time_A5 = Conv1D(64, 3, padding="same", strides=2)(Time_A_CONNECTION2)
    Time_A_GAP = GlobalAveragePooling1D()(Time_A5)

    ## Freq-ConvBlock2----------------------------------------------------------------------------------------------
    Freq_B1 = Conv1D(64, 3, padding="same", strides=2)(FF_B2)
    Freq_B2 = Conv1D(64, 2, padding="same")(Freq_B1)

    Freq_B_CONNECTION1 = Freq_B1 + Freq_B2
    Freq_B_CONNECTION1 = ReLU()(Freq_B_CONNECTION1)
    Freq_B3 = Conv1D(64, 3, padding="same", strides=2)(Freq_B_CONNECTION1)
    Freq_B4 = Conv1D(64, 2, padding="same")(Freq_B3)

    Freq_B_CONNECTION2 = Freq_B3 + Freq_B4
    Freq_B_CONNECTION2 = ReLU()(Freq_B_CONNECTION2)
    Freq_B5 = Conv1D(64, 3, padding="same", strides=2)(Freq_B_CONNECTION2)
    Freq_B_GAP = GlobalAveragePooling1D()(Freq_B5)

    # ## Time+Freq-ConvBlock2-----------------------------------------------------------------------------------------
    # TimeFreq_C1 = Conv1D(64, 3, padding="same", strides=2)(FF_C2)
    # TimeFreq_C2 = Conv1D(64, 2, padding="same")(TimeFreq_C1)
    #
    # TimeFreq_C_CONNECTION1 = TimeFreq_C1 + TimeFreq_C2
    # TimeFreq_C_CONNECTION1 = ReLU()(TimeFreq_C_CONNECTION1)
    # TimeFreq_C3 = Conv1D(64, 3, padding="same", strides=2)(TimeFreq_C_CONNECTION1)
    # TimeFreq_C4 = Conv1D(64, 2, padding="same")(TimeFreq_C3)
    #
    # TimeFreq_C_CONNECTION2 = TimeFreq_C3 + TimeFreq_C4
    # TimeFreq_C_CONNECTION2 = ReLU()(TimeFreq_C_CONNECTION2)
    # TimeFreq_C5 = Conv1D(64, 3, padding="same", strides=2)(TimeFreq_C_CONNECTION2)
    # TimeFreq_C_GAP = GlobalAveragePooling1D()(TimeFreq_C5)

    ## Dense--------------------------------------------------------------------------------------------------------
    Con = Concatenate()([Time_A_GAP, Freq_B_GAP])
    outputs = Dense(5, activation='softmax')(Con)
    # -------------------------------------------------------------------------------------------------------------

    # Define the model
    model = Model(inputs=[Input_time, Input_FFT], outputs=outputs)
    model.summary()

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # Train the model
    history=model.fit([Time_data, FFT_data], Y_train,
                      epochs=50,
                      batch_size=32,
                      validation_split=0.1,
                      validation_steps=4)

    # Save model
    model.save(f'{Result_path}/{Modle_name}.h5')

    # Visualize training results
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']

    plt.subplot(2, 1, 1)
    plt.plot(train_acc, '-bo', label='Training acc')
    plt.plot(val_acc, '-ro', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'])

    plt.subplot(2, 1, 2)
    plt.plot(train_loss, '-bo', label='train loss')
    plt.plot(val_loss, '-ro', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'])

    a1 = pd.DataFrame(train_acc)
    a2 = pd.DataFrame(val_acc)
    a3 = pd.DataFrame(train_loss)
    a4 = pd.DataFrame(val_loss)
    result = pd.concat([a1, a2, a3, a4], axis=1)
    result.to_csv(f'{Result_path}/Train_History_{Modle_name}.csv', index=False)
    plt.savefig(f'{Result_path}/Train_History_{Modle_name}.tiff')
    plt.show()

if job == 2:
    # Load the pre-trained model
    model = load_model(f'{Result_path}/{Modle_name}.h5')

    # Load test data
    Test_time_data = np.load(f'{Data_path}/Test_time_data.npy', allow_pickle=True)
    Test_FFT_data = np.load(f'{Data_path}/Test_FFT_data.npy', allow_pickle=True)
    Y_test = np.load(f'{Data_path}/Y_test.npy')
    print(f"Test_time_data shape : {Test_time_data.shape}\n"
          f"Test_FFT_data shape : {Test_FFT_data.shape}\n"
          f"Y_test shape : {Y_test.shape}")

    # Evaluate model on the test data
    score = model.evaluate([Test_time_data, Test_FFT_data], Y_test, verbose=0)
    print(f"Test loss: {score[0]}\nTest accuracy: {score[1]}")

    # Save test results to CSV
    a1 = pd.DataFrame(score)
    a1.to_csv(f'{Result_path}/{Modle_name}_Noisetest_result.csv',
              index=False, mode='a', header=False)

    # Generate and display confusion matrix
    Y_test_categorical = tf.keras.utils.to_categorical(Y_test, 5)
    y_pred = model.predict([Test_time_data, Test_FFT_data])
    YY_test = np.argmax(Y_test_categorical, axis=1)
    YY_pred = np.argmax(y_pred, axis=1)

    label = ['0', '1', '2', '3', '4']
    report = confusion_matrix(YY_test, YY_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=report, display_labels=label)
    disp.plot(cmap='Blues')
    plt.savefig(f'{Result_path}/{Modle_name}_ConfusionMatrix_NoiseX.tiff')

    ## Test for different noise levels
    noiselist = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
    for noise in noiselist:
        noise = round(noise, 2)
        print(f"noise: {noise}")

        Test_time_data = np.load(f'{Data_path}/Test_time_data_Noise{noise}.npy', allow_pickle=True)
        Test_FFT_data = np.load(f'{Data_path}/Test_FFT_data_Noise{noise}.npy', allow_pickle=True)
        Y_test = np.load(f'{Data_path}/Y_test.npy')
        print(f"Test_time_data shape : {Test_time_data.shape}\n"
              f"Test_FFT_data shape : {Test_FFT_data.shape}\n"
              f"Y_test shape : {Y_test.shape}")

        # Evaluate model on the test data
        score = model.evaluate([Test_time_data, Test_FFT_data], Y_test, verbose=0)
        print(f"Test loss: {score[0]}\nTest accuracy: {score[1]}")

        # Save test results to CSV
        a1 = pd.DataFrame(score)
        a1.to_csv(f'{Result_path}/{Modle_name}_Noisetest_result.csv',
                  index=False, mode='a', header=False)

        Y_test_categorical = tf.keras.utils.to_categorical(Y_test, 5)
        y_pred = model.predict([Time_test_data, FFTtestdata])
        YY_test = np.argmax(Y_test_categorical, axis=1)
        YY_pred = np.argmax(y_pred, axis=1)

        label = ['0', '1', '2', '3', '4']
        report = confusion_matrix(YY_test, YY_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=report, display_labels=label)
        disp.plot(cmap='Blues')
        plt.savefig(f'{Result_path}/{Modle_name}_ConfusionMatrix_Noise{noise}.tiff')

    plt.show()