import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataset import ParticlesDataset, collate_fn, transform
from model import get_model
import os
import json

def create_directories(directories):
    """
    Create necessary directories if they don't already exist.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")

def calculate_iou(pred_boxes, true_boxes):
    """
    Calculate Intersection over Union (IoU) for the predicted and true bounding boxes.
    """
    if pred_boxes.numel() == 0 or true_boxes.numel() == 0:
        return 0.0

    inter = (torch.min(pred_boxes[:, None, 2:], true_boxes[:, 2:]) -
             torch.max(pred_boxes[:, None, :2], true_boxes[:, :2])).clamp(0).prod(2)
    area_pred = (pred_boxes[:, 2:] - pred_boxes[:, :2]).prod(1)
    area_true = (true_boxes[:, 2:] - true_boxes[:, :2]).prod(1)
    union = area_pred[:, None] + area_true - inter
    iou = inter / union
    return iou.diag().mean().item()

# Define the directories to be created
directories = [
    'data/img/images',
    'data/img/Annotations',
    'data/saved_model',
    'data/training_history'
]

# Create the directories if they don't exist
create_directories(directories)

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

root_dir = 'data/img'
dataset = ParticlesDataset(root_dir=root_dir, transform=transform)

# Repeat the dataset to increase the number of iterations per epoch
repeat_factor = 5  # Adjust this factor as needed
repeated_dataset = ConcatDataset([dataset] * repeat_factor)

data_loader = DataLoader(repeated_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

num_classes = 2  # 1 class (particle) + background
model = get_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10
history = {'loss': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        # Calculate accuracy (IoU)
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            batch_iou = []
            for j, output in enumerate(outputs):
                pred_boxes = output['boxes']
                true_boxes = targets[j]['boxes']
                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    iou = calculate_iou(pred_boxes, true_boxes)
                    batch_iou.append(iou)
            if batch_iou:
                batch_accuracy = sum(batch_iou) / len(batch_iou)
                running_accuracy += batch_accuracy
        model.train()
        
        if i % 10 == 0:
            avg_loss = running_loss / (i + 1)
            avg_accuracy = running_accuracy / (i + 1)
            print(f"Epoch {epoch + 1},  Loss: {avg_loss:.4f}, Accuracy (IoU): {avg_accuracy:.4f}")

    lr_scheduler.step()

    history['loss'].append(running_loss / len(data_loader))


# Save the training history
with open('data/training_history/history.json', 'w') as f:
    json.dump(history, f)

torch.save(model.state_dict(), 'data/saved_model/particle_detector.pth')
print("Model saved as 'data/saved_model/particle_detector.pth'")
