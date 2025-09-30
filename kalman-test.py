import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter as EKF

# Define state transition function
def state_transition_function(x, dt):
    R, dR = x  # R: resistance, dR: rate of change of resistance
    return np.array([R + dR * dt, dR])

# Define measurement function (for example, dR/R0)
def measurement_function(x):
    R, _ = x
    R0 = 100  # Example fixed reference resistance
    return np.array([R / R0])

# Jacobian of state transition
def jacobian_state_transition(x, dt):
    return np.array([[1, dt], [0, 1]])

# Jacobian of measurement model
def jacobian_measurement(x):
    R, _ = x
    R0 = 100
    return np.array([[1 / R0, 0]])

# Simulate synthetic data for demonstration
np.random.seed(42)
n_steps = 30
true_R0 = 100  # Reference resistance
true_R = [true_R0 + i for i in range(n_steps)]  # True resistance
noise_std = 2.0
measurements = [r / true_R0 + np.random.normal(0, noise_std / true_R0) for r in true_R]

# Initialize EKF
ekf = EKF(dim_x=2, dim_z=1)
ekf.x = np.array([measurements[0] * true_R0, 0])  # Initial state: resistance and rate of change
ekf.F = jacobian_state_transition(ekf.x, dt=1)
ekf.H = jacobian_measurement  # Assign function for HJacobian
ekf.P *= 1000  # Initial uncertainty
ekf.R = np.array([[noise_std / true_R0]])  # Measurement noise
ekf.Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise

# Run EKF on the measurements
filtered_resistance = []
for z in measurements:
    ekf.predict()
    ekf.update(z=np.array([z]), HJacobian=lambda x: jacobian_measurement(x), Hx=lambda x: measurement_function(x))
    filtered_resistance.append(ekf.x[0])

# Predict the next value
ekf.predict()
next_predicted_resistance = ekf.x[0]

# Print results
print("Filtered Resistance:", filtered_resistance)
print("Next Predicted Resistance:", next_predicted_resistance)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(true_R, label="True Resistance", color="green")
plt.plot([z * true_R0 for z in measurements], label="Measurements", linestyle="none", marker="x", color="red")
plt.plot(filtered_resistance, label="Filtered Resistance", color="blue")
plt.axvline(x=len(filtered_resistance)-1, color='k', linestyle='--', alpha=0.5, label="Last Measurement")
plt.plot(len(filtered_resistance), next_predicted_resistance, 'bo', label="Next Prediction")
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Resistance (Ohms)")
plt.title("Extended Kalman Filter for Sensor Data")
plt.show()