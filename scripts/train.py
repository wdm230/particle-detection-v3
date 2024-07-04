import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ParticlesDataset, collate_fn, transform
from model import get_model
import os

def create_directories(directories):
    """
    Create necessary directories if they don't already exist.

    Args:
        directories (list): List of directory paths to be created.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")

# Define the directories to be created
directories = [
    'data/img/images',
    'data/img/Annotations',
    'data/saved_model'
]

# Create the directories if they don't exist
create_directories(directories)

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

root_dir = 'data/img'
dataset = ParticlesDataset(root_dir=root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

num_classes = 2  # 1 class (particle) + background
model = get_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 30

for epoch in range(num_epochs):
    """
    Train the model for the specified number of epochs.

    Args:
        num_epochs (int): Number of epochs to train the model.
    """
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        
        if i % 10 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {losses.item()}")

    lr_scheduler.step()


torch.save(model.state_dict(), 'data/saved_model/particle_detector.pth')
print("Model saved as 'data/saved_model/particle_detector.pth'")
