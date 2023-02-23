import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# Define the PyTorch dataset for loading the data
class MedicalTranscriptionDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.transcriptions = df['transcription'].values
        self.medical_specialties = df['medical_specialty'].values

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, index):
        return self.transcriptions[index], self.medical_specialties[index]

# Define the PyTorch model
class MedicalSpecialtyClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output.squeeze(0)

# Define the training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for text, labels in iterator:
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Define the evaluation loop
def evaluate(model, iterator):
    model.eval()
    predictions = []
    labels_list = []
    with torch.no_grad():
        for text, labels in iterator:
            output = model(text)
            predictions.append(torch.argmax(output, dim=1))
            labels_list.append(labels)
    predictions = torch.cat(predictions)
    labels = torch.cat(labels_list)
    f1_macro = f1_score(labels, predictions, average='macro')
    return f1_macro

# Load the CSV file as a PyTorch dataset and split the data into training and testing sets
dataset = MedicalTranscriptionDataset('path/to/file.csv')
train_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# Define the model, optimizer, and loss function
vocab_size = 10000
embedding_dim = 128
hidden_dim = 256
output_dim = len(dataset.medical_specialties.unique())
num_layers = 2
model = MedicalSpecialtyClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_f1_macro = evaluate(model, test_loader)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Test F1-macro: {test_f1_macro:.3f}')
