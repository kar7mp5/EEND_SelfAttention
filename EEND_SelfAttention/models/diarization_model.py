import torch
import torch.nn as nn

class SpeakerDiarizationModel(nn.Module):
    """
    Self-Attention-based Speaker Diarization Model.

    Attributes:
        fc1 (nn.Linear): Initial linear layer for feature transformation.
        attention (nn.MultiheadAttention): Multi-head self-attention module.
        fc2 (nn.Linear): Output layer for speaker classification.
    """

    def __init__(self, input_dim=64, num_speakers=3):
        super(SpeakerDiarizationModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(128, num_speakers)

    def forward(self, x):
        """
        Forward pass for diarization model.

        Args:
            x (torch.Tensor): Input tensor with shape (Batch, Seq_Len, Feature_Dim).

        Returns:
            torch.Tensor: Speaker presence predictions (Batch, Seq_Len, Num_Speakers).
        """
        x = self.fc1(x)  # (Batch, Seq_Len, 128)
        attn_output, _ = self.attention(x, x, x)  # Self-attention mechanism
        x = self.fc2(attn_output)  # (Batch, Seq_Len, Num_Speakers)
        return torch.sigmoid(x)  # Apply sigmoid activation for binary classification
