import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from multi_encoder_input import MultiEncoderInput
from variance_adaptor import AccentVarianceAdaptor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[..., :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.pe[:, :x.size(1), :]

class PostNet(nn.Module):
    def __init__(self, num_mels=80):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_mels, 512, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, num_mels, 5, padding=2)
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class FastSpeech2Accent(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.multi_encoder_input = MultiEncoderInput(
            content_dim=config.content_dim,
            speaker_dim=config.speaker_dim,
            accent_dim=config.accent_dim,
            hidden_dim=config.hidden_dim
        )
        
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        
        # Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, config.num_encoder_layers)
        
        self.variance_adaptor = AccentVarianceAdaptor(
            hidden_dim=config.hidden_dim,
            num_pitch_bins=config.num_pitch_bins,
            num_energy_bins=config.num_energy_bins
        )
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, config.num_decoder_layers)
        
        self.mel_dense = nn.Linear(config.hidden_dim, config.num_mels)
        self.postnet = PostNet(config.num_mels)
    
    def forward(self, content_emb, speaker_emb, accent_emb, 
                pitch_target=None, energy_target=None, duration_target=None):
        
        # Fuse embeddings
        encoder_input = self.multi_encoder_input(content_emb, speaker_emb, accent_emb)
        
        # Add positional encoding
        encoder_input = encoder_input + self.pos_encoding(encoder_input)
        
        # Encode
        encoder_output = self.encoder(encoder_input)
        
        # Variance adaptor
        adaptor_output = self.variance_adaptor(
            encoder_output,
            pitch_target=pitch_target,
            energy_target=energy_target,
            duration_target=duration_target
        )
        
        # Decode
        decoder_output = self.decoder(adaptor_output, encoder_output)
        
        # Mel projection
        mel_pred = self.mel_dense(decoder_output)
        
        # PostNet refinement
        mel_postnet = self.postnet(mel_pred.transpose(1, 2)).transpose(1, 2)
        mel_pred_refined = mel_pred + mel_postnet
        
        return mel_pred, mel_pred_refined