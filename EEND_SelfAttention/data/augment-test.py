import librosa
import librosa.display
import numpy as np
import os
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import butter, filtfilt

def adjust_audio_length(audio, target_length=20.0):
    """
    Ensure audio length is exactly 20 seconds by either padding with silence or slicing.
    """
    print("Adjusting audio length to 20 seconds")
    current_length = audio.duration_seconds
    
    if current_length < target_length:
        print("Padding audio with silence")
        silence_pad = AudioSegment.silent(duration=(target_length - current_length) * 1000)
        return audio + silence_pad
    elif current_length > target_length:
        print("Trimming audio to 20 seconds")
        return audio[:target_length * 1000]
    
    return audio

def process_folder(audio_folder, save_directory):
    """
    Process all audio files in a folder and merge them into groups.
    """
    print(f"Processing folder: {audio_folder}")
    os.makedirs(save_directory, exist_ok=True)
    merged_directory = os.path.join(save_directory, "merged")
    os.makedirs(merged_directory, exist_ok=True)
    
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".wav")]
    processed_files = []
    
    for file in audio_files:
        detected_segments = process_audio(file, save_directory)
        processed_files.append(os.path.join(save_directory, "processed_audio", os.path.basename(file).replace('.wav', '_processed.wav')))
    
    group_index = 0
    for num_files in [1, 2, 3]:
        for i in range(0, len(processed_files), num_files):
            group = processed_files[i:i+num_files]
            if len(group) > 0:
                merged_audio_path = merge_audio(group, merged_directory, group_index, num_files)
                generate_rttm(merged_audio_path, speaker_labels=[f"tank{j}" for j in range(len(group))])
                group_index += 1

def bandpass_filter(y, sr, lowcut=300, highcut=1000):
    """
    Applies a bandpass filter to emphasize specific frequency ranges.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, y)

def get_band_energy(y, sr, low_freq=300, high_freq=1000):
    """
    Computes the energy in a specific frequency band.
    """
    fft_result = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), 1/sr)
    band_energy = np.sum(np.abs(fft_result[(freqs >= low_freq) & (freqs <= high_freq)]))
    return band_energy

def merge_audio(audio_files, save_path, group_index, num_files):
    """
    Merge multiple audio files while maintaining correct alignment.
    """
    print(f"Merging {num_files} files: {audio_files}")
    os.makedirs(save_path, exist_ok=True)
    
    audio_segments = [AudioSegment.from_wav(file) for file in audio_files]
    merged_audio = audio_segments[0]
    for audio in audio_segments[1:]:
        merged_audio = merged_audio.overlay(audio)
    
    merged_audio = adjust_audio_length(merged_audio, target_length=20.0)
    
    merged_audio_path = os.path.join(save_path, f"merged_audio_{num_files}_{group_index}.wav")
    merged_audio.export(merged_audio_path, format="wav")
    print(f"Merged audio saved: {merged_audio_path}")
    return merged_audio_path

def generate_rttm(audio_path, speaker_labels):
    """
    Generate an RTTM file for the given audio file with multiple speaker labels.
    """
    print(f"Generating RTTM for: {audio_path} with labels {speaker_labels}")
    y, sr = librosa.load(audio_path, sr=None)
    segment_duration = 0.5
    segment_samples = int(sr * segment_duration)
    detected_segments = []
    
    initial_energy = get_band_energy(y[:segment_samples], sr)
    energy_threshold = initial_energy * 0.8
    
    for i in range(0, len(y), segment_samples):
        y_segment = y[i:i + segment_samples]
        if len(y_segment) < segment_samples:
            break
        energy = get_band_energy(y_segment, sr)
        if energy > energy_threshold:
            detected_segments.append((i / sr, (i + segment_samples) / sr))
    
    rttm_path = os.path.dirname(audio_path)
    base_name = os.path.basename(audio_path).replace('.wav', '')
    rttm_file = os.path.join(rttm_path, f"{base_name}.rttm")
    with open(rttm_file, "w") as f:
        for segment, label in zip(detected_segments, speaker_labels):
            start, end = segment
            duration = end - start
            f.write(f"SPEAKER {base_name} 1 {start:.3f} {duration:.3f} <NA> <NA> {label} <NA>\n")
    print(f"RTTM file saved: {rttm_file}")

def process_audio(file_path, save_dir, segment_duration=0.5, energy_threshold_factor=0.8, amplitude_threshold=0.02, merge_threshold=0.5):
    """
    Process an audio file: apply filtering, label segments, and save processed audio.
    """
    print(f"Processing audio: {file_path}")
    y, sr = librosa.load(file_path, sr=None)
    segment_samples = int(sr * segment_duration)
    detected_segments = []
    
    initial_energy = get_band_energy(y[:segment_samples], sr)
    energy_threshold = initial_energy * energy_threshold_factor
    
    for i in range(0, len(y), segment_samples):
        y_segment = y[i:i + segment_samples]
        if len(y_segment) < segment_samples:
            break
        energy = get_band_energy(y_segment, sr)
        amplitude = np.max(np.abs(y_segment))
        if energy > energy_threshold and amplitude > amplitude_threshold:
            detected_segments.append((i / sr, (i + segment_samples) / sr))
    
    base_name = os.path.basename(file_path).replace('.wav', '')
    processed_audio_path = os.path.join(save_dir, "processed_audio")
    os.makedirs(processed_audio_path, exist_ok=True)
    sf.write(os.path.join(processed_audio_path, f"{base_name}_processed.wav"), y, sr)
    
    generate_rttm(file_path, speaker_labels=["tank0"])
    return detected_segments

# Example Usage
audio_folder = "/home/kar/Projects/EEND_SelfAttention/EEND_SelfAttention/data/test_data"
save_directory = "processed_audio"
process_folder(audio_folder, save_directory)