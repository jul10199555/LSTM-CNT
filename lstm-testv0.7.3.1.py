import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load CSV
df = pd.read_csv("data/CN_F_D_S6_CLS.csv", skipinitialspace=True)
#df = pd.read_csv("data/CN_F_D_S1_dR_R0.csv", skipinitialspace=True)

# Vars
#test_column = "9-7p (6016)"
test_column = "6-4p (6010)"

seq_length = 20  # Used last given values to predict the "next one"
LSTM_activation_func = 'relu' # or 'tanh'
LSTM_mem_cells = 100 # number of units (a.k.a. hidden size / memory cells)
LSTM_epochs = 50 # total of training cycles

# Assume time is in first column, resistance in column 25 (12-10p)
time = df.iloc[:, 0].values  # Time column (not used in training directly)
R_measured = df.loc[:, test_column].values  # Default = 25

print(len(R_measured))

#plt.ion()

#fig, ax = plt.subplots(2,3, figsize=(13,4))
gs = gridspec.GridSpec(2,2)
ax = pl.subplot(gs[0,0])

# quería mostrar solo los datos que se ploteaban en el prediction (subplot3)
# pero saca plots diferentes x.x
n = min(seq_length, len(R_measured))
ax.plot(time[-n:], R_measured[-n:], label=df.loc[:, test_column].name)

#ax.plot(time, R_measured, label=df.loc[:, test_column].name)
ax.set_title("Reference signal")
ax.set_xlabel("Time Step (Test Sequence)")
ax.set_ylabel(r'$\Delta R / R_0\ (\%)$')
ax.legend()

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

X, y = create_sequences(R_measured_scaled, seq_length)

#print(X)
#print(y)

# ================================
# Train / Test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.41, random_state=42
)

# Reshape for LSTM input [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ================================
# Build the LSTM model
# ================================
model = Sequential([
    Input(shape=(seq_length, 1)),
    LSTM(LSTM_mem_cells, activation=LSTM_activation_func),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ================================
# Train
# ================================
history = model.fit(
    X_train, y_train,
    epochs=LSTM_epochs,
    validation_data=(X_test, y_test),
    verbose=0
)

# ================================
# Plot training history
# ================================

#fig, ax = plt.subplots()
ax = pl.subplot(gs[0,1])
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_title('LSTM Training History')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()

'''
plt.clf()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
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

#fig, ax = plt.subplots(figsize=(10, 6))
ax = pl.subplot(gs[1,:])
ax.plot(y_test_original, label="Actual Resistance", color="green")
ax.plot(predicted, label="Predicted Resistance", color="blue", linestyle="--")
ax.set_title("LSTM Prediction")
ax.set_xlabel("Time Step (Test Sequence)")
ax.set_ylabel(r'$\Delta R / R_0\ (\%)$')
ax.legend()
#fig.tight_layout()
'''
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label="Actual Resistance", color="green")
plt.plot(predicted, label="Predicted Resistance", color="blue", linestyle="--")
plt.title("LSTM Resistance Prediction")
plt.xlabel("Time Step (Test Sequence)")
plt.ylabel("Resistance (Ohms)")
plt.legend()
plt.show()
'''
# ================================
# Predict the next value
# ================================
last_sequence = R_measured_scaled[-seq_length:].reshape(1, seq_length, 1)
next_value_scaled = model.predict(last_sequence)
next_value = scaler.inverse_transform(next_value_scaled)
print("Next Predicted Resistance Value:", next_value[0][0])
'''
'''
#plt.ioff()
pl.show() 