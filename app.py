import streamlit as st
import torch
import torch.nn as nn
import numpy as np
class weather_model(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
    self.fc = nn.Sequential(
        nn.Linear(in_features=64, out_features=8), nn.ReLU(),
        nn.Linear(in_features=8, out_features=1)
    )
  def forward(self, X):
    lstm_out, _ = self.lstm(X)  # LSTM trả về (output, (h_n, c_n))
    # print(f"lstm_out {lstm_out.shape}")
    last_hidden_state = lstm_out[:, -1, :]  # Lấy hidden state của timestep cuối
    output = self.fc(last_hidden_state)  # Đưa vào Fully Connected Layer
    return output
  
model = weather_model(input_size=1)
model.load_state_dict(torch.load("model_0.pth"))
model.eval()
st.title("Dự đoán nhiệt độ dựa trên 5 ngày trước")

# Nhập dữ liệu nhiệt độ của 5 ngày trước
st.write("Nhập nhiệt độ của 5 ngày trước:")
temp_inputs = []
for i in range(5):
    temp_inputs.append(st.number_input(f"Ngày {i+1}", value=25.0, step=0.1))

# Dự đoán khi nhấn nút
if st.button("Dự đoán nhiệt độ ngày tiếp theo"):
    input_tensor = torch.tensor([temp_inputs], dtype=torch.float32)
    prediction = model(input_tensor).item()
    st.success(f"Nhiệt độ dự đoán: {prediction:.2f}°C")