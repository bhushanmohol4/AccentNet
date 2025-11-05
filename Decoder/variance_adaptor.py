import torch
import torch.nn as nn

class AccentVarianceAdaptor(nn.Module):
    def __init__(self, hidden_dim=256, num_pitch_bins=256, num_energy_bins=256):
        super().__init__()
        self.pitch_embedding = nn.Embedding(num_pitch_bins, hidden_dim)
        self.energy_embedding = nn.Embedding(num_energy_bins, hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def quantize(self, values, num_bins, v_min=0, v_max=1):
        """Quantize continuous values to discrete bins."""
        if values.dim() == 3:
            values = values.squeeze(-1)
        
        values = torch.clamp(values, v_min, v_max)
        boundaries = torch.linspace(v_min, v_max, num_bins, device=values.device)
        bins = torch.bucketize(values, boundaries)
        bins = torch.clamp(bins, 0, num_bins - 1)
        return bins
    
    def forward(self, encoder_output, pitch_target=None, energy_target=None, 
                duration_target=None):
        output = encoder_output
        
        # Add pitch
        if pitch_target is not None:
            pitch_bins = self.quantize(pitch_target, 256, 50, 400)
            pitch_emb = self.pitch_embedding(pitch_bins)
            output = output + self.dropout(pitch_emb)
        
        # Add energy
        if energy_target is not None:
            energy_bins = self.quantize(energy_target, 256, 0, 1)
            energy_emb = self.energy_embedding(energy_bins)
            output = output + self.dropout(energy_emb)
        
        # Length regulation
        if duration_target is not None:
            output = self.length_regulate(output, duration_target)
        
        return output
    
    def length_regulate(self, encoder_output, duration_target):
        batch_size, seq_len, hidden_dim = encoder_output.shape
        durations = torch.round(duration_target).long().clamp(min=1)
        
        output_list = []
        for i in range(batch_size):
            frames = []
            for j in range(seq_len):
                frame = encoder_output[i, j].unsqueeze(0)
                repeated = frame.repeat(durations[i, j], 1)
                frames.append(repeated)
            
            output_list.append(torch.cat(frames, dim=0))
        
        # Pad to max length
        max_len = max([o.shape for o in output_list])
        padded = torch.zeros(batch_size, max_len, hidden_dim,
                            device=encoder_output.device, dtype=encoder_output.dtype)
        
        for i, o in enumerate(output_list):
            padded[i, :o.shape] = o
        
        return padded
