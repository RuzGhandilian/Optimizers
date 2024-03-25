import torch
import torch.nn as nn
from RMSprop import RMSPropOptimizer
from Adagrad import AdagradOptimizer
from Network import NeuralNetwork

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=1000):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = criterion(predictions, y_test)
        print(f'Mean Squared Error: {mse.item():.4f}')


for method in ['xavier', 'kaiming', 'random', 'normal']:
    print(f"Training with weight initialization method: {method}")
    model = NeuralNetwork(X_train_scaled.shape[1], 64, 1, method)
    criterion = nn.MSELoss()
    optimizer = AdagradOptimizer(model.parameters(), lr=0.01, eps=1e-8)
    # optimizer = RMSPropOptimizer(model.parameters(), lr=0.01, eps=1e-8)
    train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor)
    print("Evaluation results:")
    evaluate_model(model, X_test_tensor, y_test_tensor)
    print("")
