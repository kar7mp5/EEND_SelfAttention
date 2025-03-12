import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from utils.config import config  # Load configuration


# 1️⃣ 특정 폴더에서 .wav 및 .rttm 파일 자동 검색 및 매칭
def load_data_from_folder(folder_path):
    """
    Automatically searches and matches .wav and .rttm files from a folder.
    """
    audio_files = []
    rttm_files = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".wav"):
            audio_path = os.path.join(folder_path, file)
            rttm_path = os.path.join(folder_path, file.replace(".wav", ".rttm"))

            if os.path.exists(rttm_path):
                audio_files.append(audio_path)
                rttm_files.append(rttm_path)

    return audio_files, rttm_files


# 2️⃣ RTTM 파일을 파싱하여 화자 발화 정보 추출
def parse_rttm(rttm_file):
    """
    Parses an RTTM file and extracts (speaker, start time, duration).
    """
    speaker_intervals = {}

    with open(rttm_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            speaker_id = parts[7]  # Speaker ID
            start_time = float(parts[3])  # Start time (seconds)
            duration = float(parts[4])  # Duration (seconds)
            end_time = start_time + duration  # End time

            if speaker_id not in speaker_intervals:
                speaker_intervals[speaker_id] = []
            speaker_intervals[speaker_id].append((start_time, end_time))

    return speaker_intervals


# 3️⃣ 음성 데이터를 로드하고 Mel Spectrogram으로 변환
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

# 2️⃣ Padding 적용을 위한 `collate_fn` 정의
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

    # Zero-padding을 적용하여 동일한 길이로 맞춤
    padded_features = torch.stack([
        torch.cat([f, torch.zeros(max_len - f.shape[0], f.shape[1])], dim=0) for f in features
    ])
    padded_labels = torch.stack([
        torch.cat([l, torch.zeros(max_len - l.shape[0], l.shape[1])], dim=0) for l in labels
    ])

    return padded_features, padded_labels, seq_lengths

# 4️⃣ 화자 정보를 프레임 단위로 변환
def generate_labels(speaker_intervals, num_frames):
    """
    Converts speaker intervals into frame-based labels.
    """
    NUM_SPEAKERS = config.get("dataset.num_speakers")
    FRAME_DURATION = config.get("dataset.frame_duration")

    labels = torch.zeros((num_frames, NUM_SPEAKERS))  # (seq_len, num_speakers)
    speaker_map = {spk: idx for idx, spk in enumerate(sorted(speaker_intervals.keys())[:NUM_SPEAKERS])}

    for speaker, intervals in speaker_intervals.items():
        if speaker not in speaker_map:
            continue

        speaker_idx = speaker_map[speaker]

        for start, end in intervals:
            start_frame = int(start / FRAME_DURATION)
            end_frame = int(end / FRAME_DURATION)
            labels[start_frame:end_frame, speaker_idx] = 1

    return labels


# 2️⃣ Padding 적용을 위한 `collate_fn` 정의
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

    # Zero-padding을 적용하여 동일한 길이로 맞춤
    padded_features = torch.stack([
        torch.cat([f, torch.zeros(max_len - f.shape[0], f.shape[1])], dim=0) for f in features
    ])
    padded_labels = torch.stack([
        torch.cat([l, torch.zeros(max_len - l.shape[0], l.shape[1])], dim=0) for l in labels
    ])

    return padded_features, padded_labels, seq_lengths


class SpeakerDiarizationDataset(Dataset):
    """
    Custom PyTorch dataset for speaker diarization.
    """

    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): Path to the folder containing audio and RTTM files.
        """
        self.audio_files, self.rttm_files = load_data_from_folder(folder_path)
        assert len(self.audio_files) == len(self.rttm_files), "Mismatch between audio and RTTM files."

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Loads and processes an audio file and its corresponding RTTM label.
        """
        audio_path = self.audio_files[idx]
        rttm_path = self.rttm_files[idx]

        features = extract_features(audio_path)
        num_frames = features.shape[0]

        speaker_intervals = parse_rttm(rttm_path)
        labels = generate_labels(speaker_intervals, num_frames)

        return features, labels
