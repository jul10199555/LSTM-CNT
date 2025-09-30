import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Simulate noisy sensor data
np.random.seed(42)
n_steps = 100
true_R = np.ones(n_steps) * 100

# Add multiple simulated touch events with unit steps
true_R[20:30] = 150  # Step increase for a touch event
true_R[50:60] = 180  # Step increase for another touch event
true_R[70:80] = 160  # Step increase for another touch event

noise_std = 5.0
R_measured = true_R + np.random.normal(0, noise_std, size=n_steps)

# Kalman filter parameters
dt = 1  # Time step

# State vector: [current resistance]
x = np.array([R_measured[0]])  # Initial state
P = np.array([[10]])  # Initial uncertainty

# State transition matrix
F = np.array([[1]])  # Assume resistance remains constant (can be modified for other models)

# Measurement matrix
H = np.array([[1]])

# Process and measurement noise
Q = np.array([[0.1]])  # Process noise
R = np.array([[2]])  # Measurement noise

# Kalman filter implementation
filtered_values = []
for z in R_measured:
    # Predict step
    x = F @ x
    P = F @ P @ F.T + Q

    # Update step
    y = z - H @ x  # Measurement residual
    S = H @ P @ H.T + R  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    x = x + K @ y
    P = (np.eye(len(K)) - K @ H) @ P

    # Save filtered value
    filtered_values.append(x[0])

# Tkinter GUI
class KalmanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalman Filter GUI")

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Initialize plot
        self.ax.plot(true_R, label="True Resistance", color="green")
        self.ax.scatter(range(n_steps), R_measured, label="Noisy Measurements", color="red", alpha=0.6)
        self.filtered_line, = self.ax.plot(filtered_values, label="Filtered Resistance", color="blue")
        self.future_points = []  # Keep track of future prediction points

        self.ax.set_title("Kalman Filter: Next-Step Resistance Prediction")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Resistance (Ohms)")
        self.ax.legend()

        # Add a button to estimate the next value
        self.button = ttk.Button(self.root, text="Estimate Next Value", command=self.estimate_next)
        self.button.pack(pady=10)

        self.canvas.draw()

        # Kalman filter state for future predictions
        self.x = np.array([filtered_values[-1]])  # Start with the last filtered value
        self.P = np.array([[10]])
        self.last_time = n_steps

    def estimate_next(self):
        global F, Q

        # Predict the next value
        self.x = F @ self.x  # Predict next state
        self.P = F @ self.P @ F.T + Q  # Update uncertainty

        # Add the prediction to the plot
        point, = self.ax.plot(self.last_time, self.x[0], 'o', color="orange", label="Next Prediction" if not self.future_points else None)
        self.future_points.append(point)

        self.last_time += 1
        self.ax.legend()
        self.canvas.draw()

# Run the GUI
root = tk.Tk()
app = KalmanApp(root)
root.mainloop()
