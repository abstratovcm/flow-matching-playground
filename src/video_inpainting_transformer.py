"""
video_inpainting_transformer.py

This module defines the VideoInpaintingTransformer, a Transformer-based neural network designed
for video inpainting tasks. The model takes observed frames (before and after a missing segment)
and predicts the missing frames. It supports both teacher forcing (with a causal mask applied) and
autoregressive decoding with scheduled sampling.
"""

import math
import torch
import torch.nn as nn
import random


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encodings (batch_first mode).

    This module precomputes positional encodings which are added to the input embeddings
    to incorporate information about the positions of tokens in the sequence.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model (int): Dimension of the embeddings.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length for which to precompute positional encodings.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model) containing the positional encodings.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices.
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices.
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encodings to the input embeddings.

        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output embeddings with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VideoInpaintingTransformer(nn.Module):
    def __init__(self, input_size, d_model=256, nhead=8, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=1024, missing_len=25, dropout=0.1):
        """
        Initializes the VideoInpaintingTransformer model.

        Args:
            input_size (int): Dimensionality of the input (e.g., 2 for (x, y) coordinates).
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of layers in the encoder.
            num_decoder_layers (int): Number of layers in the decoder.
            dim_feedforward (int): Dimension of the feedforward network.
            missing_len (int): Number of frames to predict (length of the missing segment).
            dropout (float): Dropout rate.
        """
        super(VideoInpaintingTransformer, self).__init__()
        self.d_model = d_model
        self.missing_len = missing_len

        # Linear embedding of the input.
        self.embedding = nn.Linear(input_size, d_model)
        # Positional encoding module.
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Transformer module.
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        # Final output projection to input size.
        self.output_linear = nn.Linear(d_model, input_size)

    def forward(self, obs_before, obs_after, tgt=None, teacher_forcing_ratio=0.0):
        """
        Forward pass of the model.

        Args:
            obs_before (Tensor): Observed frames before the missing segment (shape: [batch, obs_before_len, input_size]).
            obs_after (Tensor): Observed frames after the missing segment (shape: [batch, obs_after_len, input_size]).
            tgt (Tensor, optional): Ground truth missing frames (shape: [batch, missing_len, input_size]). 
                                    If provided, teacher forcing is used.
            teacher_forcing_ratio (float): Ratio for scheduled sampling; applies when tgt is provided.

        Returns:
            Tensor: Predicted missing frames (shape: [batch, missing_len, input_size]).
        """
        batch_size = obs_before.size(0)
        device = obs_before.device

        # Build the encoder memory from both observed segments.
        memory_seq = torch.cat([obs_before, obs_after], dim=1)  # Shape: (batch, memory_len, input_size)
        memory = self.embedding(memory_seq) * math.sqrt(self.d_model)
        memory = self.pos_encoder(memory)  # Shape: (batch, memory_len, d_model)

        # ----- Parallel Teacher-Forced Decoding Branch with Causal Mask -----
        if tgt is not None and teacher_forcing_ratio == 1.0:
            # Embed and add positional encoding to the target sequence.
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)  # Shape: (batch, missing_len, d_model)
            # Generate a causal mask to ensure each position attends only to previous positions.
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(device)
            out = self.transformer(src=memory, tgt=tgt_emb, tgt_mask=tgt_mask)  # Shape: (batch, missing_len, d_model)
            output = self.output_linear(out)  # Shape: (batch, missing_len, input_size)
            return output

        # ----- Autoregressive Decoding with Scheduled Sampling -----
        # Initialize the decoder with a start token (zeros).
        decoder_input = torch.zeros(batch_size, 1, self.embedding.in_features, device=device)
        decoder_emb = self.embedding(decoder_input) * math.sqrt(self.d_model)
        # Add positional encoding for the first position.
        decoder_emb = decoder_emb + self.pos_encoder.pe[:, :1, :]

        outputs = []
        for t in range(self.missing_len):
            # Generate a causal mask for the current decoder input length.
            tgt_mask = self.transformer.generate_square_subsequent_mask(decoder_emb.size(1)).to(device)
            out = self.transformer(src=memory, tgt=decoder_emb, tgt_mask=tgt_mask)
            last_out = out[:, -1:, :]  # Get the output corresponding to the latest token.
            pred = self.output_linear(last_out)  # Predict the next frame.
            outputs.append(pred)

            # Decide whether to use ground truth (teacher forcing) or prediction for the next token.
            if tgt is not None and t < tgt.size(1) and random.random() < teacher_forcing_ratio:
                next_token = tgt[:, t:t+1, :]
            else:
                next_token = pred

            # Embed the next token and add the appropriate positional encoding.
            next_token_emb = self.embedding(next_token) * math.sqrt(self.d_model)
            pos = self.pos_encoder.pe[:, decoder_emb.size(1):decoder_emb.size(1) + 1, :]
            next_token_emb = next_token_emb + pos
            decoder_emb = torch.cat([decoder_emb, next_token_emb], dim=1)

        output = torch.cat(outputs, dim=1)  # Concatenate predictions along the sequence dimension.
        return output
