import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter as EKF

# Define nonlinear measurement function: Resistance as a function of temperature
def measurement_function(x):
    T, _ = x  # T: temperature
    R0 = 100  # Initial resistance at T=0°C
    alpha = 0.02  # Temperature coefficient
    return np.array([R0 * np.exp(alpha * T)])

# Jacobian of measurement function
def jacobian_measurement(x):
    T, _ = x
    R0 = 100
    alpha = 0.02
    return np.array([[R0 * alpha * np.exp(alpha * T), 0]])

# State transition function (linear for temperature)
def state_transition_function(x, dt):
    T, dT = x  # T: temperature, dT: rate of change of temperature
    return np.array([T + dT * dt, dT])

# Jacobian of state transition
def jacobian_state_transition(x, dt):
    return np.array([[1, dt], [0, 1]])

# Simulate nonlinear data
np.random.seed(42)
n_steps = 50
true_T = np.linspace(0, 50, n_steps)  # True temperature
R0 = 100  # Initial resistance at T=0°C
alpha = 0.02  # Temperature coefficient
true_R = R0 * np.exp(alpha * true_T)  # True resistance

# Add noise to measurements
noise_std = 5.0
R_measured = true_R + np.random.normal(0, noise_std, size=true_T.shape)

# Initialize EKF
ekf = EKF(dim_x=2, dim_z=1)
ekf.x = np.array([0, 0])  # Initial state: [temperature, rate of change of temperature]
ekf.F = jacobian_state_transition(ekf.x, dt=1)
ekf.P *= 1000  # Initial uncertainty
ekf.R = np.array([[noise_std**2]])  # Measurement noise
ekf.Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise

# Run EKF on the measurements
filtered_temperature = []
filtered_resistance = []
for z in R_measured:
    ekf.predict()
    ekf.update(z=np.array([z]), HJacobian=lambda x: jacobian_measurement(x), Hx=lambda x: measurement_function(x))
    filtered_temperature.append(ekf.x[0])  # Extract filtered temperature
    filtered_resistance.append(measurement_function(ekf.x)[0])  # Compute resistance from filtered state

# Predict the next value
ekf.predict()
next_predicted_temperature = ekf.x[0]
next_predicted_resistance = measurement_function(ekf.x)[0]

# Print results
print("Filtered Temperature:", filtered_temperature)
print("Filtered Resistance:", filtered_resistance)
print("Next Predicted Temperature:", next_predicted_temperature)
print("Next Predicted Resistance:", next_predicted_resistance)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(true_T, true_R, label="True Resistance (Nonlinear)", color="green")
plt.scatter(true_T, R_measured, label="Noisy Measurements", color="red", alpha=0.6)
plt.plot(true_T, filtered_resistance, label="Filtered Resistance", color="blue")
plt.axvline(x=len(filtered_temperature)-1, color='k', linestyle='--', alpha=0.5, label="Last Measurement")
plt.plot(len(filtered_temperature), next_predicted_resistance, 'bo', label="Next Prediction")
plt.xlabel("Temperature (°C)")
plt.ylabel("Resistance (Ohms)")
plt.title("Extended Kalman Filter for Nonlinear Sensor Data")
plt.legend()
plt.show()