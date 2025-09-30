import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# For reproducibility
np.random.seed(42)

# True parameters
n_steps = 40
true_velocity = 1.0   # constant velocity
true_position_init = 5.0

# Generate true positions (without noise)
true_positions = [true_position_init + i * true_velocity for i in range(n_steps)]

# Generate noisy observations (measurements)
noise_std = 2.0
measurements = true_positions + np.random.normal(0, noise_std, size=n_steps)

kf = KalmanFilter(dim_x=2, dim_z=1)

# Initial state estimate:
#   position = measurements[0], velocity = 0 (can be guessed)
kf.x = np.array([measurements[0], 0.0])  

# State transition matrix (F):
#   [1  dt]
#   [0  1 ]
# Here dt=1 (discrete steps)
kf.F = np.array([[1., 1.],
                 [0., 1.]])

# Measurement function (H):
#   We directly observe the position, so H picks out the position from the state.
kf.H = np.array([[1., 0.]])

# Covariance matrix (P) - our initial guess of state uncertainty
# Make it large to reflect initial uncertainty
kf.P = np.array([[1000.,    0. ],
                 [   0., 1000.]])

# Measurement noise (R)
# This should reflect the noise level in your measurements.
kf.R = np.array([[noise_std**2]])  # If measurements have std of 2, variance=4

# Process noise (Q)
# Q_discrete_white_noise(dim=2, dt=1, var=PROCESS_NOISE_VAR)
# The 'var' parameter is the variance in the acceleration or process noise.
kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.01)

filtered_positions = []
filtered_velocities = []

for z in measurements:
    # Predict step
    kf.predict()
    
    # Update step with the new measurement z
    kf.update(z)
    
    # Store the filtered estimates
    filtered_positions.append(kf.x[0])
    filtered_velocities.append(kf.x[1])

kf.predict()
next_predicted_position = kf.x[0]   # predicted position
next_predicted_velocity = kf.x[1]   # predicted velocity

print("Next predicted position:", next_predicted_position)
print("Next predicted velocity:", next_predicted_velocity)

plt.figure(figsize=(10, 6))
plt.plot(true_positions, 'g-', label="True positions")
plt.plot(measurements, 'rx', label="Measurements")
plt.plot(filtered_positions, 'b-', label="Kalman Filtered position")
plt.axvline(x=len(filtered_positions)-1, color='k', linestyle='--', alpha=0.5, label="Last measurement")
plt.plot(len(filtered_positions), next_predicted_position, 'bo', label="Next prediction")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Position")
plt.title("1D Kalman Filter - Constant Velocity Model")
plt.show()
