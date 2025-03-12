import os
import argparse
import torch
import numpy as np
import librosa
import librosa.display
import torchaudio
import matplotlib.pyplot as plt
from models.diarization_model import SpeakerDiarizationModel
from utils.config import config

# Parse command-line arguments to get the audio file path
parser = argparse.ArgumentParser(description="Speaker Diarization Inference")
parser.add_argument("--audio", type=str, required=True, help="Path to the input audio file")
args = parser.parse_args()

AUDIO_PATH = args.audio  # Use the provided audio file path

# Verify that the provided audio file exists
if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(f"Audio file '{AUDIO_PATH}' not found. Please check the path.")

def extract_features(audio_path):
    """
    Extracts Mel spectrogram features from an audio file.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        torch.Tensor: Extracted Mel spectrogram features with shape (seq_len, num_mels).
    """
    SAMPLE_RATE = config.get("audio.sample_rate")
    N_MEL = config.get("audio.num_mels")
    FRAME_STEP = config.get("audio.frame_step")

    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    waveform = waveform.mean(dim=0)  # Convert to mono

    mel_spec = librosa.feature.melspectrogram(
        y=waveform.numpy(), sr=SAMPLE_RATE, n_mels=N_MEL, hop_length=FRAME_STEP
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return torch.tensor(mel_spec_db).T  # Transpose to (seq_len, num_mels)

# Load the trained model
MODEL_PATH = "diarization_model.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first.")

model = SpeakerDiarizationModel(
    input_dim=config.get("audio.num_mels"),
    num_speakers=config.get("dataset.num_speakers")
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

def infer(model, audio_path):
    """
    Performs speaker diarization inference on the given audio file.

    Args:
        model (torch.nn.Module): The trained speaker diarization model.
        audio_path (str): Path to the input audio file.

    Returns:
        np.ndarray: Predicted speaker labels.
        np.ndarray: Mel spectrogram for visualization.
    """
    model.eval()
    features = extract_features(audio_path).unsqueeze(0).float()  # Add batch dimension

    with torch.no_grad():
        output = model(features)

    predicted_labels = (output.squeeze(0) > 0.5).int().numpy()
    return predicted_labels, features.squeeze(0).numpy()

def save_diarization_results(audio_path, predictions, mel_spectrogram):
    """
    Saves the speaker diarization results to a text file and generates a visualization.

    Args:
        audio_path (str): Path to the input audio file.
        predictions (np.ndarray): Predicted speaker labels.
        mel_spectrogram (np.ndarray): Mel spectrogram for visualization.
    """
    num_frames, num_speakers = predictions.shape
    frame_duration = config.get("dataset.frame_duration")

    y, sr = librosa.load(audio_path, sr=config.get("audio.sample_rate"))
    time_axis = np.linspace(0, len(y) / sr, num_frames)

    # Save results to a text file
    results_file = "inference_results.txt"
    with open(results_file, "w") as f:
        f.write("Time (s)," + ",".join([f"Speaker {i+1}" for i in range(num_speakers)]) + "\n")
        for i, time in enumerate(time_axis):
            speakers_active = ",".join(map(str, predictions[i]))
            f.write(f"{time:.2f},{speakers_active}\n")

    # Generate and save the visualization
    plt.figure(figsize=(12, 8))

    # Display the Mel spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(
        mel_spectrogram.T, sr=sr, hop_length=int(sr * frame_duration), x_axis="time", y_axis="mel"
    )
    plt.colorbar(label="dB")
    plt.title("Mel Spectrogram")

    # Display the speaker diarization results
    plt.subplot(2, 1, 2)
    colors = ['b', 'orange', 'g', 'r', 'purple']

    for speaker in range(num_speakers):
        active_frames = np.where(predictions[:, speaker] == 1)[0]
        if len(active_frames) > 0:
            plt.scatter(time_axis[active_frames], np.full_like(active_frames, speaker),
                        color=colors[speaker % len(colors)], label=f"Speaker {speaker+1}", alpha=0.7)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Speaker ID")
    plt.title("Speaker Diarization Results")
    plt.yticks(range(num_speakers), [f"Speaker {i+1}" for i in range(num_speakers)])
    plt.legend()
    plt.tight_layout()
    plt.savefig("inference_plot.png")

# Run inference
predictions, mel_spectrogram = infer(model, AUDIO_PATH)

# Save the results
save_diarization_results(AUDIO_PATH, predictions, mel_spectrogram)
