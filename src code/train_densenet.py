import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder
import sys
import matplotlib.pyplot as plt

# Load input and validation files
input_file = r"D:\Bio\NeuralPredictModel\Extracted_dataset\merge_train_x200_suf.csv"
validate_file = r"D:\Bio\NeuralPredictModel\Extracted_dataset\merged_val_x200_suf.csv"

# Load the CSV data
data = pd.read_csv('%s' % input_file)
X_train = data.iloc[:, 1:].values
y_train = data.iloc[:, 0].values

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))

# Load validation data
validate = pd.read_csv('%s' % validate_file)
X_val = validate.iloc[:, 1:].values
y_val = validate.iloc[:, 0].values
y_val = encoder.transform(y_val.reshape(-1, 1))
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Giới hạn đầu ra trong khoảng [0, 1]
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def perform_autoencoder(df, encoding_dim=50, epochs=50, batch_size=32, device=None):
    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df)  # Impute missing values

    # Chuẩn hóa dữ liệu đã được impute
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_imputed)  # Chuẩn hóa dữ liệu đã impute
    
    # Chuyển dữ liệu đã chuẩn hóa thành tensor và đưa vào device (GPU/CPU)
    data_tensor = torch.FloatTensor(df_scaled).to(device)
    
    # Tạo DataLoader từ tensor
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Tạo mô hình Autoencoder
    input_dim = data_tensor.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim).to(device)
    
    # Hàm mất mát và tối ưu hóa
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    # Tiến trình huấn luyện
    best_loss = float('inf')
    for epoch in range(epochs):
        autoencoder.train()  # Đảm bảo mô hình ở chế độ huấn luyện
        
        epoch_loss = 0.0
        for batch in dataloader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')
        
        # Early stopping: dừng huấn luyện nếu không cải thiện
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        else:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Chuyển sang chế độ inference để lấy dữ liệu mã hóa (encoded data)
    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).cpu().numpy()
        
    # Trả về kết quả mã hóa dưới dạng DataFrame
    df_encoded = pd.DataFrame(encoded_data, columns=[f'Enc_{i + 1}' for i in range(encoded_data.shape[1])])
    
    return df_encoded
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Huấn luyện và mã hóa dữ liệu
X_train_series_encoded = perform_autoencoder(X_train, encoding_dim=50, epochs=100, batch_size=32, device=device)
X_val_series_encoded = perform_autoencoder(X_val, encoding_dim=50, epochs=100, batch_size=32, device=device)

X_train = X_train_series_encoded.values
X_val = X_val_series_encoded.values

# test_series_encoded = perform_autoencoder(test_series_df, encoding_dim=50, epochs=100, batch_size=32, device=device)

# Create groups: Group 0 (labels 0, 1, 2) and Group 1 (labels 3, 4, 5)
group_train = np.where(y_train[:, :3].sum(axis=1) > 0, 0, 1)  # 0: Group 0, 1: Group 1
group_val = np.where(y_val[:, :3].sum(axis=1) > 0, 0, 1)

# One-hot encode groups
group_train = tf.keras.utils.to_categorical(group_train, num_classes=2)
group_val = tf.keras.utils.to_categorical(group_val, num_classes=2)


# Define the pre-classifier model
def create_pre_classifier(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(2, activation='softmax'))  # 2 groups
    return model


# Define the group classifier model
def create_group_classifier(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Input shape
input_shape = (X_train.shape[1],)

# Build pre-classifier
pre_classifier = create_pre_classifier(input_shape)

# Build classifiers for each group
group0_classifier = create_group_classifier(input_shape, 3)  # Classes 0, 1, 2
group1_classifier = create_group_classifier(input_shape, 3)  # Classes 3, 4, 5

# Define the input layer
input_layer = Input(shape=input_shape)

# Pre-classifier output
group_output = pre_classifier(input_layer)

# Group classifiers
group0_output = group0_classifier(input_layer)
group1_output = group1_classifier(input_layer)

# Combine outputs
final_output = Concatenate()([
    group_output[:, :1] * group0_output,  # Multiply group 0 output
    group_output[:, 1:] * group1_output   # Multiply group 1 output
])

# Build the full model
model = tf.keras.Model(inputs=input_layer, outputs=final_output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(
    monitor='accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001
)

# Train the model
history = model.fit(
    X_train, y_train, epochs=100, batch_size=40, verbose=2,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, learning_rate_reduction]
)

# Plot the loss and accuracy curves for training and validation
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(history.history['loss'], color='b', label="Training loss")
# ax[0].plot(history.history['val_loss'], color='r', label="Validation loss", axes=ax[0])
# legend = ax[0].legend(loc='best', shadow=True)
# ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
# ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
# legend = ax[1].legend(loc='best', shadow=True)
# plt.savefig('%s_loss_acc.jpg' % input_file, dpi=300)

# Save the model
model.save_weights('%s_densenet_model.h5' % input_file)
print("Model saved as %s_densenet_model.h5" % input_file)
