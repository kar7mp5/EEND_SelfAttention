import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.diarization_model import SpeakerDiarizationModel
from utils.dataset import SpeakerDiarizationDataset, collate_fn  
from utils.config import config  
from tqdm import tqdm  

# Load Configuration from YAML
LEARNING_RATE = config.get("train.learning_rate")
EPOCHS = config.get("train.epochs")
BATCH_SIZE = config.get("train.batch_size")
DEVICE = torch.device(config.get("train.device") if torch.cuda.is_available() else "cpu")

# Ensure that the dataset directory exists
DATA_DIR = os.path.abspath("./dataset/")
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data folder '{DATA_DIR}' not found. Please check the path.")

# Initialize the Speaker Diarization Model
model = SpeakerDiarizationModel(
    input_dim=config.get("audio.num_mels"),
    num_speakers=config.get("dataset.num_speakers")
).to(DEVICE)

# Define the loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load the dataset and create a DataLoader instance
dataset = SpeakerDiarizationDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# List to store loss values for visualization
loss_history = []

# Train the model for the specified number of epochs
print(f"\nTraining Speaker Diarization Model on {DEVICE}\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    # Iterate over the dataset batches
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for features, labels, seq_lengths in pbar:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            # Reset gradients before backpropagation
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})  # Display real-time loss values

    # Compute the average loss for the epoch and store it
    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}\n")

# Save the trained model
MODEL_PATH = "diarization_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved successfully at {MODEL_PATH}")

# Plot and save the loss history as a PNG file
plt.figure(figsize=(8, 6))
plt.plot(range(1, EPOCHS + 1), loss_history, marker="o", linestyle="-", color="b")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.savefig("training_loss.png")
print("Training loss plot saved as 'training_loss.png'")
