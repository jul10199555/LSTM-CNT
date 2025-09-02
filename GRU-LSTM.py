import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn

class FramePredictor(nn.Module):
    def __init__(self, model_type='LSTM', input_size=205, hidden_size=256, num_layers=2):
        super().__init__()
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class FrameLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel Frame Loader and Predictor (5x41)")
        self.frames = []
        self.model = None
        self.train_len = 0
        self.prediction_cache = {}

        self.load_button = tk.Button(root, text="Load Excel File", command=self.load_excel)
        self.load_button.pack(pady=5)

        self.info_label = tk.Label(root, text="No file loaded.")
        self.info_label.pack(pady=5)

        self.slider = tk.Scale(root, from_=0, to=1, orient='horizontal', label="Frame Index", command=self.update_plot)
        self.slider.pack(fill='x', padx=20, pady=5)

        self.model_choice = ttk.Combobox(root, values=["LSTM", "GRU"])
        self.model_choice.set("LSTM")
        self.model_choice.pack(pady=5)

        train_frame = tk.Frame(root)
        train_frame.pack(pady=5)
        tk.Label(train_frame, text="Train Frames:").pack(side='left')
        self.train_entry = tk.Entry(train_frame, width=5)
        self.train_entry.insert(0, "6")
        self.train_entry.pack(side='left')

        tk.Label(train_frame, text="Epochs:").pack(side='left')
        self.epoch_entry = tk.Entry(train_frame, width=5)
        self.epoch_entry.insert(0, "100")
        self.epoch_entry.pack(side='left')

        self.predict_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.predict_button.pack(pady=5)

        self.frame_summary = tk.Text(root, height=6, width=60, state='disabled')
        self.frame_summary.pack(pady=5)

        self.figure, (self.ax_top, self.ax_bottom, self.ax_error) = plt.subplots(3, 1, figsize=(6, 4.5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.im_top = None
        self.im_bottom = None
        self.im_error = None
        self.cbar = None

    def load_excel(self):
        filepath = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not filepath:
            return

        try:
            xl = pd.ExcelFile(filepath)
            self.frames.clear()
            self.prediction_cache.clear()
            summary_lines = []

            for sheet in xl.sheet_names:
                df = xl.parse(sheet, header=None)
                if df.shape != (5, 41):
                    raise ValueError(f"Sheet '{sheet}' is not 5x41 but {df.shape}")
                self.frames.append(df.to_numpy())
                summary_lines.append(f"Loaded sheet '{sheet}' with shape {df.shape}")

            self.info_label.config(text=f"Loaded {len(self.frames)} frame(s) from: {os.path.basename(filepath)}")
            self.slider.config(to=len(self.frames) - 1)
            self.slider.set(0)
            self.update_summary(summary_lines)
            self.update_plot(0)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_summary(self, lines):
        self.frame_summary.config(state='normal')
        self.frame_summary.delete(1.0, tk.END)
        self.frame_summary.insert(tk.END, "\n".join(lines))
        self.frame_summary.config(state='disabled')

    def update_plot(self, idx):
        if not self.frames or self.model is None:
            return
        idx = int(float(idx))

        if idx in self.prediction_cache:
            predicted_np, loss_val, mae_val = self.prediction_cache[idx]
        else:
            full_data = np.array([f.flatten() for f in self.frames])
            context_start = max(0, idx - self.train_len)
            context = full_data[context_start:idx]
            if len(context) < self.train_len:
                padding = np.zeros((self.train_len - len(context), 205))
                context = np.vstack((padding, context))
            input_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                predicted = self.model(input_tensor)
                predicted_np = predicted.numpy().reshape((5, 41))

            real_frame = self.frames[idx]
            loss_val = np.mean((predicted_np - real_frame) ** 2)
            mae_val = np.mean(np.abs(predicted_np - real_frame))

            self.prediction_cache[idx] = (predicted_np, loss_val, mae_val)

        real_frame = self.frames[idx]
        error_frame = np.abs(real_frame - predicted_np)

        self.ax_top.clear()
        self.ax_bottom.clear()
        self.ax_error.clear()

        self.im_top = self.ax_top.imshow(real_frame, cmap='viridis', aspect='auto', vmin=0, vmax=25)
        self.ax_top.set_title(f"Real Frame {idx}")

        self.im_bottom = self.ax_bottom.imshow(predicted_np, cmap='viridis', aspect='auto', vmin=0, vmax=25)
        self.ax_bottom.set_title(f"Predicted Frame {idx} (Loss: {loss_val:.4f}, MAE: {mae_val:.4f})")

        self.im_error = self.ax_error.imshow(error_frame, cmap='hot', aspect='auto')
        self.ax_error.set_title("Prediction Error Heatmap")

        if self.cbar is None:
            self.cbar = self.figure.colorbar(self.im_top, ax=[self.ax_top, self.ax_bottom, self.ax_error])
            self.cbar.set_label("dR/R0")
        else:
            self.cbar.update_normal(self.im_top)

        self.canvas.draw()

    def train_model(self):
        if len(self.frames) < 2:
            messagebox.showerror("Error", "Need at least 2 frames to train.")
            return

        try:
            self.train_len = int(self.train_entry.get())
            epochs = int(self.epoch_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Train length and epochs must be integers.")
            return

        if self.train_len >= len(self.frames):
            messagebox.showerror("Error", "Train length must be less than total number of frames.")
            return

        model_type = self.model_choice.get()
        self.model = FramePredictor(model_type=model_type)

        full_data = np.array([f.flatten() for f in self.frames])

        X = []
        Y = []
        for i in range(self.train_len, len(full_data)):
            X.append(full_data[i - self.train_len:i])
            Y.append(full_data[i])

        train_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(Y), dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            output = self.model(train_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.prediction_cache.clear()
        self.update_plot(self.slider.get())

if __name__ == "__main__":
    root = tk.Tk()
    app = FrameLoaderApp(root)
    root.mainloop()
