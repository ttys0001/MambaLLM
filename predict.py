import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import math
from smamba_llm3 import MambaLLM
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

stock_symbol = 'GOOGL'
save_model_path = 'MambaLLM_GOOGL.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_columns = ["open", "high", "low", "close", "volume", "SMA10","SMA50","EMA10","BB_upper","BB_lower","MACD","RSI"]
finbert_output_dim = 768
llm_columns = [f"llm{i+1}" for i in range(finbert_output_dim)]

# Length of sequences
len_sequences = 10
n_features = len(feature_columns) + len(llm_columns)
loaded_data = np.load(f"{stock_symbol}_data.npz")
X_test = loaded_data["X_test"]
y_test = loaded_data["y_test"]
# Create TensorDatasets and DataLoaders
test_dataset = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test).float()
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle = False     # Set to false due to date feature
)

use_norm = True
d_model = 128
d_state = 32
d_ff = 128
dropout = 0.1
activation = 'gelu'
e_layers = 2
mamba_output_dim = 1000

# Model
net = MambaLLM(num_features=len(feature_columns), seq_len=len_sequences, use_norm=use_norm, d_model=d_model, d_state=d_state, d_ff=d_ff, dropout=dropout, activation=activation, e_layers=e_layers, mamba_output_dim=mamba_output_dim, finbert_output_dim=finbert_output_dim).to(device)
# Pretrained Model
net = torch.load(save_model_path)

actual_test = []
pred = []
# Test data
net.eval()
with torch.no_grad():
    for seqs, labels in test_dataloader:
        seqs, labels = seqs.to(device), labels.to(device)  # 数据移动到 GPU
        seqs = seqs.view(seqs.size(0), len_sequences, n_features)
        
        # Pass seqs to net and squeeze the result
        outputs = net(seqs).squeeze()

        # Save actual and predicted close prices from test data
        actual_test.extend(labels.cpu().numpy())
        pred.extend(outputs.cpu().numpy())

best_pred = np.array(pred)
best_actual_test = np.array(actual_test)
min_vals = 52.455711
max_vals = 149.125549 

def inverse_transform(scaled_data, min_vals, max_vals):
    return scaled_data * (max_vals - min_vals) + min_vals

close_pred1 = inverse_transform(best_pred, min_vals, max_vals).reshape(-1, 1)
close_actual_test1 = inverse_transform(best_actual_test, min_vals, max_vals).reshape(-1, 1)
mse = mean_squared_error(close_pred1, close_actual_test1)
rmse = math.sqrt(mse)
mae = mean_absolute_error(close_pred1, close_actual_test1)
mape = mean_absolute_percentage_error(close_pred1, close_actual_test1)*100
print(f'Test RMSE: {rmse:.4f}')
print(f'Test MAPE: {mape:.4f}')
print(f'Test MAE: {mae:.4f}')