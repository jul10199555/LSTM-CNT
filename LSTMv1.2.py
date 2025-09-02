import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Simulate tactile resistive sensor data with gradual touch and release
np.random.seed(42)
n_steps = 200
true_R = np.ones(n_steps) * 100  # Baseline resistance

# Gradual touch and release events
for i in range(20, 40):
    true_R[i] += (i - 20) * 2.5  # Gradually increase resistance
for i in range(40, 60):
    true_R[i] += (60 - i) * 2.5  # Gradually release resistance

for i in range(100, 120):
    true_R[i] += (i - 100) * 3.0  # Another touch event, sharper increase
for i in range(120, 140):
    true_R[i] += (140 - i) * 3.0  # Gradually release

# Add noise to simulate real-world measurements
noise_std = 5.0
R_measured = true_R + np.random.normal(0, noise_std, size=true_R.shape)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
R_measured_scaled = scaler.fit_transform(R_measured.reshape(-1, 1))

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Use the last 10 values to predict the next value
X, y = create_sequences(R_measured_scaled, seq_length)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
predicted_scaled = model.predict(X_test)
predicted = scaler.inverse_transform(predicted_scaled)

# Inverse transform the true values
y_test_original = scaler.inverse_transform(y_test)

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label="Actual Resistance", color="green")
plt.plot(predicted, label="Predicted Resistance", color="blue", linestyle="--")
plt.title("LSTM Resistance Prediction")
plt.xlabel("Time Step")
plt.ylabel("Resistance (Ohms)")
plt.legend()
plt.show()

# Predict the next value
last_sequence = R_measured_scaled[-seq_length:].reshape(1, seq_length, 1)
next_value_scaled = model.predict(last_sequence)
next_value = scaler.inverse_transform(next_value_scaled)
print("Next Predicted Resistance Value:", next_value[0][0])
