# import torch
# import torch.nn as nn

# class SpeakerDiarizationModel(nn.Module):
#     """
#     Self-Attention-based Speaker Diarization Model.

#     Attributes:
#         fc1 (nn.Linear): Initial linear layer for feature transformation.
#         attention (nn.MultiheadAttention): Multi-head self-attention module.
#         fc2 (nn.Linear): Output layer for speaker classification.
#     """

#     def __init__(self, input_dim=64, num_speakers=3):
#         super(SpeakerDiarizationModel, self).__init__()

#         self.fc1 = nn.Linear(input_dim, 128)
#         self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
#         self.fc2 = nn.Linear(128, num_speakers)

#     def forward(self, x):
#         """
#         Forward pass for diarization model.

#         Args:
#             x (torch.Tensor): Input tensor with shape (Batch, Seq_Len, Feature_Dim).

#         Returns:
#             torch.Tensor: Speaker presence predictions (Batch, Seq_Len, Num_Speakers).
#         """
#         x = self.fc1(x)  # (Batch, Seq_Len, 128)
#         attn_output, _ = self.attention(x, x, x)  # Self-attention mechanism
#         x = self.fc2(attn_output)  # (Batch, Seq_Len, Num_Speakers)
#         return torch.sigmoid(x)  # Apply sigmoid activation for binary classification

import torch
import torch.nn as nn

class SpeakerDiarizationModel(nn.Module):
    """
    Speaker Diarization Model with self-attention, BiLSTM, multiple convolutional layers, and normalization techniques.
    """
    def __init__(self, input_dim=64, num_speakers=3, hidden_dim=128, num_heads=4, cnn_filters=64):
        super(SpeakerDiarizationModel, self).__init__()

        # Convolutional feature extraction with multiple layers
        self.conv1 = nn.Conv1d(input_dim, cnn_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters)
        self.conv3 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_filters)
        self.conv_activation = nn.GELU()

        # Initial linear transformation
        self.fc1 = nn.Linear(cnn_filters, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()

        # BiLSTM for sequential modeling
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True, dropout=0.3)
        
        # Multi-head Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Fully connected layer for classification
        self.fc2 = nn.Linear(hidden_dim, num_speakers)
        self.dropout = nn.Dropout(0.4)  # Increased dropout for better generalization

    def forward(self, x):
        """
        Forward pass for speaker diarization.

        Args:
            x (torch.Tensor): Input tensor with shape (Batch, Seq_Len, Feature_Dim).

        Returns:
            torch.Tensor: Speaker presence predictions (Batch, Seq_Len, Num_Speakers).
        """
        # Convolutional feature extraction
        x = x.transpose(1, 2)  # Convert to (Batch, Feature_Dim, Seq_Len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv_activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv_activation(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv_activation(x)
        
        x = x.transpose(1, 2)  # Convert back to (Batch, Seq_Len, Feature_Dim)
        
        # Fully connected transformation
        x = self.fc1(x)
        x = self.bn4(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        
        # BiLSTM layer
        x, _ = self.bilstm(x)
        
        # Self-Attention
        attn_output, _ = self.attention(x, x, x)
        x = self.ln1(attn_output + x)
        
        # Final classification layer
        x = self.fc2(self.dropout(x))
        return torch.sigmoid(x)  # Sigmoid for multi-label classification
