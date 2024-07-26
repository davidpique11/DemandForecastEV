import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot time series
def plot_series(time, series, format="-", start=0, end=None):
    plt.figure(figsize=(25, 10))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(rotation=90)
    plt.grid(True)

# Plot train and validation loss function, MAE and MSE over the epochs 
def show_loss_accuracy_evolution(history, model_name, window_size, batch_size):

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Hubber Loss')
    ax1.plot(hist['epoch'], hist['loss'], label='Train Error')
    ax1.plot(hist['epoch'], hist['val_loss'], label = 'Val Error')
    ax1.grid()
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.plot(hist['epoch'], hist['mae'], label='Train MAE')
    ax2.plot(hist['epoch'], hist['val_mae'], label = 'Val MAE')
    ax2.grid()
    ax2.legend()

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MSE')
    ax3.plot(hist['epoch'], hist['mse'], label='Train MSE')
    ax3.plot(hist['epoch'], hist['val_mse'], label = 'Val MSE')
    ax3.grid()
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_bs{batch_size}_results/history.png')
    plt.show()

# Plot train and test predictions
def plot_predictions(train_predict, val_predict, test_predict, window_size, series, x_train, x_val):
    # We must shift the predictions so that they align on the x-axis with the original dataset. 
    # Shift train predictions for plotting:
    trainPredictPlot = np.empty_like(series)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[window_size:len(x_train), :] = train_predict
    
    # Shift validation predictions for plotting
    valPredictPlot = np.empty_like(series)
    valPredictPlot[:, :] = np.nan
    valPredictPlot[len(x_train)+(window_size):len(x_train)+len(x_val), :] = val_predict

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(series)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(x_train)+len(x_val)+(window_size):len(series), :] = test_predict
    
    # Plotting
    plt.figure(figsize=(20, 8))
    plt.plot(series, label='Actual Data') 
    plt.plot(trainPredictPlot, label='Train Predict')
    plt.plot(valPredictPlot, label='Validation Predict')
    plt.plot(testPredictPlot, label='Test Predict')
    plt.title('Predictions on the Whole Dataset')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.legend()
    plt.show()

# Plot real values and predicted from 'start' value to 'end' value
def raw_data_vs_predictions(x, predict, window_size, start, end, prediction_type): 
    plt.figure(figsize=(20, 8))
    plt.plot(x[window_size + start: window_size + end], label='Actual Data')
    plt.plot(predict[start:end, 0], label=f'{prediction_type} Prediction')
    plt.legend()
    plt.tight_layout()

# Función para realizar la predicción y calcular el error relativo
def forecast_and_evaluate(start, x_test, x_test_scaled, model, scaler, window_size, num_features, horizon, plot=True):
    # Lista para almacenar las predicciones
    prediction = []

    # Preparar el lote de datos inicial
    current_batch = x_test_scaled[start : start + window_size]
    current_batch = current_batch.reshape(1, window_size, num_features)

    # Realizar predicciones futuras
    for i in range(horizon):
        current_pred = model.predict(current_batch)[0]
        prediction.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # Invertir la transformación para obtener valores reales
    rescaled_prediction = scaler.inverse_transform(prediction)

    # Datos reales para el mismo período
    real_data = x_test[start + window_size : start + window_size + horizon]

    # Calcular la suma de las predicciones y los datos reales
    sum_predictions = np.sum(rescaled_prediction)
    sum_real_data = np.sum(real_data)

    # Calcular el error relativo
    relative_error = (np.abs(sum_predictions - sum_real_data) / sum_real_data) *100

    # Plot final con todas las predicciones y datos reales
    if plot: 
        plt.figure(figsize=(20, 8))
        plt.title('24h Horizon Forecast')
        plt.plot(real_data, label='Actual Data')
        plt.plot(rescaled_prediction, label='Predictions')
        plt.legend()
        plt.show()

    return rescaled_prediction, real_data, relative_error