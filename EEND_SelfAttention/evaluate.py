import os
import torch
from torch.utils.data import DataLoader
from models.diarization_model import SpeakerDiarizationModel
from utils.dataset import SpeakerDiarizationDataset, collate_fn
from utils.metrics import diarization_error_rate  
from utils.config import config
from tqdm import tqdm

# Load configuration from YAML
DEVICE = torch.device(config.get("train.device") if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.abspath(config.get("dataset.path"))
BATCH_SIZE = config.get("train.batch_size")

# Ensure that the dataset directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset folder '{DATA_DIR}' not found. Please check the path.")

# Load the test dataset
test_dataset = SpeakerDiarizationDataset(DATA_DIR)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Load the trained model
MODEL_PATH = "diarization_model.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first.")

model = SpeakerDiarizationModel(
    input_dim=config.get("audio.num_mels"),
    num_speakers=config.get("dataset.num_speakers")
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def evaluate(model, dataloader):
    """
    Evaluates the speaker diarization model using the test dataset.

    Args:
        model (torch.nn.Module): Trained speaker diarization model.
        dataloader (DataLoader): DataLoader instance for the test dataset.

    Returns:
        float: Average Diarization Error Rate (DER).
    """
    total_der = 0
    total_samples = 0

    with torch.no_grad():
        with tqdm(dataloader, desc="Evaluating Model") as pbar:
            for features, labels, seq_lengths in pbar:
                features, labels = features.to(DEVICE), labels.to(DEVICE)

                # Generate model predictions
                outputs = model(features)

                # Convert outputs to binary speaker presence predictions
                predictions = (outputs > 0.5).int()

                # Compute Diarization Error Rate (DER)
                batch_der = diarization_error_rate(predictions.cpu().numpy(), labels.cpu().numpy())
                total_der += batch_der
                total_samples += 1

                pbar.set_postfix({"DER": f"{batch_der:.4f}"})  # Display real-time DER values

    avg_der = total_der / total_samples
    return avg_der

# Run the evaluation process
avg_der = evaluate(model, test_dataloader)

# Print the final evaluation result
print(f"Average Diarization Error Rate (DER): {avg_der:.4f}")
