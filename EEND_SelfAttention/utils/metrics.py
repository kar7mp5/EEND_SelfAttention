import numpy as np

def diarization_error_rate(predicted, ground_truth):
    """
    Compute the Diarization Error Rate (DER).
    
    Args:
        predicted (np.ndarray): Model predictions (batch_size, seq_len, num_speakers).
        ground_truth (np.ndarray): True speaker labels (batch_size, seq_len, num_speakers).

    Returns:
        float: DER score.
    """
    total_frames = predicted.shape[1]  # 전체 프레임 수
    incorrect_frames = np.sum(predicted != ground_truth)  # 잘못된 프레임 수

    return incorrect_frames / (total_frames * predicted.shape[0])  # 평균 DER 계산
