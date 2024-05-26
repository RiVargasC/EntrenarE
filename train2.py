import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F  # Importar torch.nn.functional
from torchvision.transforms import ToTensor
from tqdm import tqdm
from RRDBNet_arch import RRDBNet
from utils import get_data_loaders

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hiperparámetros
lr_dir = 'LR'
hr_dir = 'HR'
pretrained_model_path = 'Previo/RRDB_ESRGAN_x3.pth'  # Ruta al modelo previamente entrenado
model_save_path = 'models/RRDB_ESRGAN_x3.pth'
num_epochs = 50  # Número de épocas
batch_size = 16  # Reducir el tamaño del lote para evitar problemas de memoria
learning_rate = 1e-4
num_workers = 2  # Reducir el número de trabajadores

# Obtener dataloaders
train_loader = get_data_loaders(lr_dir, hr_dir, batch_size=batch_size, num_workers=num_workers)

# Definir el modelo
model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
model = model.to(device)

# Cargar el modelo previamente entrenado si existe
if os.path.exists(pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path))
    print(f'Modelo cargado desde {pretrained_model_path}')

# Definir la función de pérdida y el optimizador
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Bucle de entrenamiento
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for lr_imgs, hr_imgs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        # Forward pass
        sr_imgs = model(lr_imgs)
        sr_imgs = F.interpolate(sr_imgs, size=hr_imgs.shape[2:], mode='cubic', align_corners=False)
        loss = criterion(sr_imgs, hr_imgs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Guardar el modelo después de cada época
    torch.save(model.state_dict(), model_save_path)
    print(f'Modelo guardado en {model_save_path}')

print('Entrenamiento completado y modelo guardado.')
