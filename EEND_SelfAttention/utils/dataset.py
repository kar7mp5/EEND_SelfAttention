import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from utils.config import config  # Load configuration

# 1️⃣ Load audio and matching .npy label files
def load_data_from_folder(folder_path):
    """
    Searches for .wav audio files and matches them with their corresponding .npy segment files.
    """
    audio_files = []
    npy_files = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".wav"):
            audio_path = os.path.join(folder_path, file)
            npy_path = os.path.join(folder_path, file.replace(".wav", ".npy"))

            if os.path.exists(npy_path):
                audio_files.append(audio_path)
                npy_files.append(npy_path)

    return audio_files, npy_files

# 2️⃣ Extract audio features (Mel Spectrogram)
def extract_features(audio_path):
    """
    Loads an audio file and converts it to Mel Spectrogram.
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

    return torch.tensor(mel_spec_db).T  # (seq_len, n_mel)

# 3️⃣ Convert .npy speaker segments into frame-based labels
def generate_labels(npy_path, num_frames):
    """
    Converts speaker intervals from .npy segment files into frame-based labels.

    Args:
        npy_path (str): Path to the .npy file containing speaker segments.
        num_frames (int): Number of frames in the feature sequence.

    Returns:
        torch.Tensor: Frame-wise speaker labels.
    """
    NUM_SPEAKERS = config.get("dataset.num_speakers")
    FRAME_DURATION = config.get("dataset.frame_duration")

    # Load segment information from .npy file
    segment_data = np.load(npy_path, allow_pickle=True).item()
    speaker_intervals = segment_data.get("segments", [])

    labels = torch.zeros((num_frames, NUM_SPEAKERS))  # (seq_len, num_speakers)

    # Convert "tankX" labels into numerical indices
    speaker_map = {}  
    current_speaker_id = 1  # Start labeling from 1 (0 is silence)

    for start, end, label in speaker_intervals:
        if label not in speaker_map:
            if len(speaker_map) >= NUM_SPEAKERS:
                continue  # Ignore extra speakers if limit is reached
            speaker_map[label] = current_speaker_id
            current_speaker_id += 1

        speaker_idx = speaker_map[label]
        start_frame = int(start / FRAME_DURATION)
        end_frame = int(end / FRAME_DURATION)

        labels[start_frame:end_frame, speaker_idx] = 1

    return labels

# 4️⃣ Dataset class using .npy instead of RTTM
class SpeakerDiarizationDataset(Dataset):
    """
    Custom PyTorch dataset for speaker diarization using NPY segment files instead of RTTM.
    """

    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): Path to the folder containing audio and NPY segment files.
        """
        self.audio_files, self.npy_files = load_data_from_folder(folder_path)
        assert len(self.audio_files) == len(self.npy_files), "Mismatch between audio and NPY files."

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Loads an audio file and its corresponding NPY label file.
        
        Returns:
            features (torch.Tensor): Extracted audio features.
            labels (torch.Tensor): Frame-wise speaker labels.
        """
        audio_path = self.audio_files[idx]
        npy_path = self.npy_files[idx]

        # Extract features from audio
        features = extract_features(audio_path)
        num_frames = features.shape[0]

        # Convert segment-based labels to frame-wise labels
        labels = generate_labels(npy_path, num_frames)

        return features.clone().detach(), labels.clone().detach()

# 5️⃣ Padding 적용을 위한 collate_fn
def collate_fn(batch):
    """
    Custom collate function for padding sequences of varying lengths.

    Args:
        batch (list): List of tuples (features, labels).

    Returns:
        torch.Tensor: Padded features (batch_size, max_seq_len, feature_dim).
        torch.Tensor: Padded labels (batch_size, max_seq_len, num_speakers).
        torch.Tensor: Sequence lengths for each sample (batch_size,).
    """
    features, labels = zip(*batch)

    # 각 샘플의 길이를 저장
    seq_lengths = torch.tensor([f.shape[0] for f in features])

    # 가장 긴 샘플을 기준으로 패딩 적용
    max_len = max(seq_lengths)

    # Zero-padding 적용
    padded_features = torch.stack([
        torch.cat([f, torch.zeros(max_len - f.shape[0], f.shape[1])], dim=0) for f in features
    ])
    padded_labels = torch.stack([
        torch.cat([l, torch.zeros(max_len - l.shape[0], l.shape[1])], dim=0) for l in labels
    ])

    return padded_features, padded_labels, seq_lengths
