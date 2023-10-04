import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.autograd import Variable
import numpy
import math
import os
from osgeo import gdal
import rasterio
import torchvision.transforms as transforms
from PIL import Image


# def load_tiff_files(path_to_files):
#     image_files = [os.path.join(path_to_files, file) for file in os.listdir(
#         path_to_files) if file.endswith('.tif')]

#     images = []
#     labels = []

#     for file in image_files:
#         dataset = gdal.Open(file)
#         img_array = np.array(dataset.ReadAsArray())
#         images.append(img_array)

#         metadata = dataset.GetMetadata()
#         # stellen Sie sicher, dass der Schlüssel korrekt ist
#         mpi_value = float(metadata['MPI_VALUE'])
#         labels.append(mpi_value)

#     return np.array(images), np.array(labels)

def load_tiff_files(path_to_files):
    image_files = [os.path.join(path_to_files, file) for file in os.listdir(
        path_to_files) if file.endswith('.tif')]

    images = []
    labels = []

    to_tensor = transforms.ToTensor()

    for file in image_files:
        # Open image with rasterio
        with rasterio.open(file) as dataset:
            # Convert to PIL image
            img_array = dataset.read().transpose((1, 2, 0))
            pil_img = Image.fromarray((img_array * 255).astype(np.uint8))

            # Convert PIL image to tensor
            img_tensor = to_tensor(pil_img)

            images.append(img_tensor)

            # Extract metadata
            mpi_value = float(dataset.tags()['MPI_VALUE'])
            labels.append(mpi_value)

    return torch.stack(images), np.array(labels)


# Pfad zu Ihren TIFF-Dateien
path_to_files = 'c:/Users/finnl/Downloads/regional_tifs_with_attributes'
images, labels = load_tiff_files(path_to_files)
# print(images.shape)
# print(labels.shape)
print(images[:5], labels[:5])
# print(labels[:5])

# Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.25, random_state=42)

# Dies wird Ihnen die Form der Daten zeigen. Die zweite Dimension ist die Anzahl der Kanäle.
# print(X_train.shape)  # Sollte (Anzahl der Bilder, 4, 40, 40) ausgeben
# print(X_test.shape)   # Sollte auch (Anzahl der Bilder, 4, 40, 40) ausgeben

# Daten in PyTorch-Tensoren umwandeln und die Dimensionen ändern
# X_train = torch.from_numpy(X_train).float()
# X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float().view(-1, 1)
y_test = torch.from_numpy(y_test).float().view(-1, 1)

# print(X_train.shape)  # Sollte (Anzahl der Bilder, 4, 40, 40) ausgeben
# print(X_test.shape)   # Sollte auch (Anzahl der Bilder, 4, 40, 40) ausgeben

# CNN-Modell definieren
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Da Ihre Daten 4 Kanäle haben, bleibt dies bei 4
        self.conv1 = nn.Conv2d(4, 8, 4, 1)
        self.pool = nn.MaxPool2d(4, 2)
        self.conv2 = nn.Conv2d(8, 16, 2, 2)
        # Anpassung der Linear Layer Größe, um mit der Modellbeschreibung übereinzustimmen
        # Dies sollte jetzt korrekt sein, wenn die Größe nach den Conv und Pooling Layern 16x27x27 ist
        self.fc1 = nn.Linear(16 * 3 * 3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()

# Loss-Funktion und Optimizer definieren
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Modell trainieren
num_epochs = 100
batch_size = 30
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
    #       1, num_epochs, loss.item()))

# Modell testen
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    test_loss = criterion(outputs, y_test)
    print('Test Loss: {:.4f}'.format(test_loss.item()))
    
# Calculate the R2 score
r2 = r2_score(y_test, outputs)
print('R2 Score:', r2)

# Gewichte und Biases ausgeben
# print(model.state_dict())

# # Einige Vorhersagen ausgeben
print('Predictions:', outputs[:5])
print('True values:', y_test[:5])
