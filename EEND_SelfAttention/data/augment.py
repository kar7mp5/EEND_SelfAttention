import librosa
import librosa.display
import numpy as np
import os
import soundfile as sf
from pydub import AudioSegment

def process_audio(file_path, save_dir, label_threshold=0.1, hop_length=512):
    """
    Load an audio file, apply STFT, label based on amplitude threshold, and save results.
    """
    y, sr = librosa.load(file_path, sr=None)
    
    # Compute STFT with hop_length for proper time alignment
    D = librosa.stft(y, hop_length=hop_length)
    magnitude, _ = librosa.magphase(D)
    
    # Labeling based on threshold
    labels = (np.mean(magnitude, axis=0) > label_threshold).astype(int)
    
    # Save processed audio
    base_name = os.path.basename(file_path).replace('.wav', '')
    processed_audio_path = os.path.join(save_dir, "processed_audio")
    os.makedirs(processed_audio_path, exist_ok=True)
    
    sf.write(os.path.join(processed_audio_path, f"{base_name}_processed.wav"), y, sr)
    np.save(os.path.join(processed_audio_path, f"{base_name}_labels.npy"), labels)
    
    return labels, sr, hop_length

def adjust_audio_length(audio, labels, target_length=20.0, frame_rate=100):
    """
    Ensure audio length is exactly 20 seconds by either padding with silence or slicing.
    """
    current_length = audio.duration_seconds
    target_frames = int(target_length * frame_rate)
    
    if current_length < target_length:
        silence_pad = AudioSegment.silent(duration=(target_length - current_length) * 1000)
        labels = np.pad(labels, (0, target_frames - len(labels)), constant_values=0)
        return audio + silence_pad, labels[:target_frames]
    elif current_length > target_length:
        return audio[:target_length * 1000], labels[:target_frames]
    
    return audio, labels[:target_frames]

def merge_audio_with_labels(audio_files, save_path, group_index, num_files, frame_rate=100):
    """
    Merge audio files while ensuring length consistency and label synchronization.
    """
    os.makedirs(save_path, exist_ok=True)
    
    audio_segments = [AudioSegment.from_wav(file) for file in audio_files]
    label_files = [file.replace('_processed.wav', '_labels.npy') for file in audio_files]
    labels_list = [np.load(label) if os.path.exists(label) else np.zeros(200) for label in label_files]
    
    min_length = min(len(labels) for labels in labels_list)
    labels_list = np.array([labels[:min_length] for labels in labels_list]).T
    
    merged_audio = audio_segments[0]
    for audio in audio_segments[1:]:
        merged_audio = merged_audio.overlay(audio)
    
    merged_audio, labels_list = adjust_audio_length(merged_audio, labels_list, target_length=20.0, frame_rate=frame_rate)
    
    valid_indices = np.any(labels_list, axis=1)
    labels_list = labels_list[valid_indices]
    
    merged_audio_path = os.path.join(save_path, f"merged_audio_{num_files}_{group_index}.wav")
    merged_audio.export(merged_audio_path, format="wav")
    
    labels_path = os.path.join(save_path, f"merged_audio_{num_files}_{group_index}_labels.npy")
    np.save(labels_path, labels_list)

def process_folder(audio_folder, save_directory):
    """
    Process all audio files and merge them into groups, ensuring correct label alignment.
    """
    os.makedirs(save_directory, exist_ok=True)
    merged_directory = os.path.join(save_directory, "merged")
    os.makedirs(merged_directory, exist_ok=True)
    
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".wav")]
    processed_files = []
    
    for file in audio_files:
        labels, sr, hop_length = process_audio(file, save_directory)
        processed_files.append(os.path.join(save_directory, "processed_audio", os.path.basename(file).replace('.wav', '_processed.wav')))
    
    group_index = 0
    for num_files in [1, 2, 3]:
        for i in range(0, len(processed_files), num_files):
            group = processed_files[i:i+num_files]
            if len(group) > 0:
                merge_audio_with_labels(group, merged_directory, group_index, num_files)
                group_index += 1

# Example usage
audio_folder = "/home/kar/Projects/EEND_SelfAttention/EEND_SelfAttention/data/test_data"
save_directory = "processed_audio"
process_folder(audio_folder, save_directory)
