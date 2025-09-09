import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load CSV
df = pd.read_csv("CN_F_D_S6_CLS.csv")

# Assume time is in first column, resistance in column 25 (12-10p)
time = df.iloc[:, 0].values  # Time column (not used in training directly)
R_measured = df.iloc[:, 25].values  # Resistive values

#plt.plot(time, R_measured)
#plt.show()

# Ensure numpy array and reshape for scaler
#print(R_measured.shape)
#R_measured = np.array(R_measured).reshape(-1, 1)
#print(R_measured.shape)
#print(time.shape)

# ================================
# Normalize the resistance values
# ================================
scaler = MinMaxScaler(feature_range=(0, 1))
#R_measured_scaled = scaler.fit_transform(R_measured)
R_measured_scaled = scaler.fit_transform(R_measured.reshape(-1, 1))

#print(R_measured_scaled)
#plt.plot(time, R_measured_scaled)
#plt.show()


# ================================
# Sequence creation function
# ================================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Use last 10 resistance values to predict next
X, y = create_sequences(R_measured_scaled, seq_length)

#print(X)
#print(y)


# ================================
# Train / Test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Reshape for LSTM input [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ================================
# Build the LSTM model
# ================================
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ================================
# Train
# ================================
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=0
)

# ================================
# Plot training history
# ================================
plt.clf()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ================================
# Predictions
# ================================
predicted_scaled = model.predict(X_test)
predicted = scaler.inverse_transform(predicted_scaled)

# Inverse transform the true values
y_test_original = scaler.inverse_transform(y_test)

# ================================
# Plot predictions vs actual
# ================================
#plt.clf()
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label="Actual Resistance", color="green")
plt.plot(predicted, label="Predicted Resistance", color="blue", linestyle="--")
plt.title("LSTM Resistance Prediction")
plt.xlabel("Time Step (Test Sequence)")
plt.ylabel("Resistance (Ohms)")
plt.legend()
plt.show()

# ================================
# Predict the next value
# ================================
last_sequence = R_measured_scaled[-seq_length:].reshape(1, seq_length, 1)
next_value_scaled = model.predict(last_sequence)
next_value = scaler.inverse_transform(next_value_scaled)
print("Next Predicted Resistance Value:", next_value[0][0])
'''
'''