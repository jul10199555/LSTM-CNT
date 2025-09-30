import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter as EKF

# Nonlinear measurement function: Resistance as a function of state
def measurement_function(x):
    R, _ = x  # R: resistance
    return np.array([R])

# Jacobian of measurement function
def jacobian_measurement(x):
    return np.array([[1, 0]])

# State transition function with resistance decay and clamping
def state_transition_function(x, dt):
    R, dR = x  # R: resistance, dR: rate of change of resistance
    R0 = 100  # Baseline resistance
    decay_rate = 0.1
    R_new = max(R - decay_rate * (R - R0) * dt, R0)  # Decay toward R0
    return np.array([R_new + dR * dt, dR])

# Jacobian of state transition
def jacobian_state_transition(x, dt):
    decay_rate = 0.1
    return np.array([[1 - decay_rate * dt, dt], [0, 1]])

# Simulate tactile data
np.random.seed(42)
n_steps = 100
true_R = np.ones(n_steps) * 100
true_R[20:40] = 150  # Simulate a touch event
true_R[60:80] = 180  # Another touch event

# Add noise to measurements
noise_std = 5.0
R_measured = true_R + np.random.normal(0, noise_std, size=true_R.shape)

# Initialize EKF
ekf = EKF(dim_x=2, dim_z=1)
ekf.x = np.array([100, 0])  # Initial state: [resistance, rate of change]
ekf.P = np.array([[10, 0], [0, 10]])  # Initial uncertainty
# ekf.R = np.array([[noise_std**2]])  # Measurement noise
ekf.R = np.array([[3.0]])  # Measurement noise
ekf.Q = np.array([[0.3, 0], [0, 0.3]]) #[0.1, 0], [0, 0.1]  # Process noise

# Run EKF on the measurements
filtered_resistance = []
for z in R_measured:
    ekf.predict()
    ekf.update(z=np.array([z]), HJacobian=lambda x: jacobian_measurement(x), Hx=lambda x: measurement_function(x))
    filtered_resistance.append(ekf.x[0])  # Extract filtered resistance

# Predict the next value
ekf.predict()
next_predicted_resistance = ekf.x[0]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(true_R, label="True Resistance (Tactile Events)", color="green")
plt.scatter(range(n_steps), R_measured, label="Noisy Measurements", color="red", alpha=0.6)
plt.plot(filtered_resistance, label="Filtered Resistance", color="blue")
plt.axvline(x=n_steps - 1, color='k', linestyle='--', alpha=0.5, label="Last Measurement")
plt.plot(n_steps, next_predicted_resistance, 'bo', label="Next Prediction")
plt.xlabel("Time Step")
plt.ylabel("Resistance (Ohms)")
plt.title("Improved Extended Kalman Filter for Tactile Resistive Sensor")
plt.legend()
plt.show()