import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class AccentConversionDataset(Dataset):
    def __init__(self, metadata_file, embeddings_dir):
        """
        Args:
            metadata_file: Text file with source_id | target_id per line
            embeddings_dir: Directory with pre-extracted embeddings
        """
        self.embeddings_dir = embeddings_dir
        self.pairs = []
        
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.strip():
                    source_id, target_id, sentence_id = line.strip().split('|')
                    self.pairs.append((source_id.strip(), target_id.strip(), sentence_id.strip()))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        print(self.pairs)
        source_id, target_id,sentence_id = self.pairs[idx]
        
        # Load source embeddings
        source_path = os.path.join(self.embeddings_dir, source_id)
        content_emb = np.load(source_path + f'/{sentence_id}_content.npy')  # [seq_len, 768]
        speaker_emb = np.load(source_path + f'/{sentence_id}_speaker.npy')  #
        
        
        # Load prosody
        with open(source_path + f'/{sentence_id}_prosody.pkl', 'rb') as f:
            prosody = pickle.load(f)
        
        # Load target data
        target_path = os.path.join(self.embeddings_dir, target_id)
        accent_emb = np.load(target_path + f'/{sentence_id}_accent.npy')    # [seq_len, 256]
        mel_target = np.load(target_path + f'/{sentence_id}_mel.npy')  # [mel_len, 80]
        
        return {
            'content_emb': torch.from_numpy(content_emb).float(),
            'speaker_emb': torch.from_numpy(speaker_emb).float(),
            'accent_emb': torch.from_numpy(accent_emb).float(),
            'pitch': torch.from_numpy(prosody['pitch']).float(),
            'energy': torch.from_numpy(prosody['energy']).float(),
            'duration': torch.from_numpy(prosody['duration']).float(),
            'mel_target': torch.from_numpy(mel_target).float(),
        }
