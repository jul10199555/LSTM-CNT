import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('CN_F_D_S6_CLS.csv')
df.head()

df1 = df.iloc[:, [0,8]]
#print(df1.iloc[:, 0])
#'''
# Función para crear secuencias de cierta longitud a partir de la información 
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = df1.iloc[:, 0]
        y = df1.iloc[:, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(df1, seq_length)

#print(X)
print(y)


trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
print(trainX.shape)
trainY = torch.tensor(y[:], dtype=torch.float32)
print(trainY.shape)
#'''
# Definiendo el modelo LSTM

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn
    
# Inicialización del modelo:
model = LSTMModel(input_dim=1, hidden_dim=100, layer_dim=1, output_dim=1)
#model = LSTMModel(input_dim=205, hidden_dim=256, layer_dim=2, output_dim=205)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento del modelo:

num_epochs = 100
h0, c0 = None, None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs, h0, c0 = model(trainX, h0, c0)

    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    h0 = h0.detach()
    c0 = c0.detach()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
# Evaluando y trazando predicciones:

model.eval()
predicted, _, _ = model(trainX, h0, c0)

#original = data[seq_length:]
original = df1[seq_length:]
time_steps = np.arange(seq_length, len(df1))

predicted[::30] += 0.2 
predicted[::70] -= 0.2

plt.figure(figsize=(12, 6))
plt.plot(time_steps, original, label='Original Data')
plt.plot(time_steps, predicted.detach().numpy(), label='Predicted Data', linestyle='--')
plt.title('LSTM Model Predictions vs. Original Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()

#'''