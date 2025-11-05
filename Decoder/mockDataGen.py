#!/usr/bin/env python3
"""
Generate mock embedding files for testing the accent converter model.

This creates synthetic (non-real) embeddings with the correct shapes and format.
Useful for:
- Verifying dataset loading works
- Testing model forward pass
- Debugging without real data
- Quick integration tests

Run this ONCE to create mock data, then train on it to verify everything works.

Usage:
    python create_mock_dataset.py --output_dir data/mock_embeddings --num_sentences 5 --num_speakers 3
"""

import numpy as np
import pickle
import argparse
from pathlib import Path

class MockDatasetGenerator:
    """Generate synthetic embedding files for testing."""
    
    def __init__(self, output_dir='data/mock_embeddings', 
                 num_sentences=5, num_speakers=3, seed=42):
        """
        Args:
            output_dir: Where to save mock embeddings
            num_sentences: Number of sentences to generate (5-10 is enough for testing)
            num_speakers: Number of speakers (3-4 is enough)
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.num_sentences = num_sentences
        self.num_speakers = num_speakers
        self.seed = seed
        
        np.random.seed(seed)
    
    def generate_all(self):
        """Generate all mock embeddings."""
        
        print(f"Generating mock dataset:")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Sentences: {self.num_sentences}")
        print(f"  Speakers: {self.num_speakers}")
        print()
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        speaker_names = [f'speaker_{i:03d}' for i in range(self.num_speakers)]
        sentence_ids = [f'sentence_{i:03d}' for i in range(self.num_sentences)]
        
        # Generate embeddings for each speaker
        for speaker_name in speaker_names:
            speaker_dir = self.output_dir / speaker_name
            speaker_dir.mkdir(exist_ok=True)
            
            print(f"Creating {speaker_name}/")
            
            for sentence_id in sentence_ids:
                # ===== CONTENTFEC (same for all speakers reading same sentence) =====
                # Shape: [seq_len, 768]
                # This represents "what is said" - should be similar across speakers
                seq_len = np.random.randint(100, 200)  # Variable length
                content_emb = np.random.randn(seq_len, 768).astype(np.float32)
                
                # Add some structure: normalize and make more realistic
                content_emb = content_emb / np.linalg.norm(content_emb, axis=1, keepdims=True)
                
                # Save ContentVec
                np.save(
                    speaker_dir / f'{sentence_id}_content.npy',
                    content_emb
                )
                
                # ===== SPEAKER EMBEDDING (unique per speaker) =====
                # Shape: 
                # This represents "who is speaking"
                speaker_emb = np.random.randn(192).astype(np.float32)
                speaker_emb = speaker_emb / np.linalg.norm(speaker_emb)  # Normalize
                
                # Save Speaker
                np.save(
                    speaker_dir / f'{sentence_id}_speaker.npy',
                    speaker_emb
                )
                
                # ===== ACCENT EMBEDDING (varies by speaker) =====
                # Shape: [seq_len, 256]
                # This represents accent characteristics
                accent_emb = np.random.randn(seq_len, 256).astype(np.float32)
                accent_emb = accent_emb / np.linalg.norm(accent_emb, axis=1, keepdims=True)
                
                # Save Accent
                np.save(
                    speaker_dir / f'{sentence_id}_accent.npy',
                    accent_emb
                )
                
                # ===== PROSODY =====
                # Dictionary with pitch, energy, duration
                
                # Mel length (roughly 3x content length)
                mel_len = int(seq_len * np.random.uniform(2.5, 3.5))
                
                # Pitch: F0 values in Hz (typical range 50-400 Hz)
                pitch = np.random.uniform(100, 250, mel_len).astype(np.float32)
                
                # Energy: normalized loudness (0-1)
                energy = np.random.uniform(0.3, 1.0, mel_len).astype(np.float32)
                
                # Duration: how many mel frames per content frame
                duration = np.ones(seq_len, dtype=np.float32) * (mel_len / seq_len)
                
                prosody = {
                    'pitch': pitch,
                    'energy': energy,
                    'duration': duration
                }
                
                # Save Prosody
                with open(speaker_dir / f'{sentence_id}_prosody.pkl', 'wb') as f:
                    pickle.dump(prosody, f)
                
                # ===== MEL-SPECTROGRAM =====
                # Shape: [mel_len, 80]
                # This is the acoustic feature (ground truth target)
                mel_target = np.random.randn(mel_len, 80).astype(np.float32)
                
                # Make it more realistic (typically 0-1 range after normalization)
                mel_target = (mel_target - mel_target.mean()) / (mel_target.std() + 1e-7)
                mel_target = (mel_target + 3) / 6  # Shift to roughly 0-1 range
                mel_target = np.clip(mel_target, 0, 1)
                
                # Save Mel
                np.save(
                    speaker_dir / f'{sentence_id}_mel.npy',
                    mel_target
                )
                
                print(f"  ✓ {sentence_id}: "
                      f"content=[{seq_len}, 768], "
                      f"prosody=[{mel_len}], "
                      f"mel=[{mel_len}, 80]")
        
        print("\n✓ Mock dataset generation complete!")
        print(f"  Created {self.num_speakers} speakers × {self.num_sentences} sentences")
        print(f"  Total pairs for training: {self.num_speakers * (self.num_speakers - 1) * self.num_sentences}")
    
    def verify(self):
        """Verify all files were created correctly."""
        
        print("\nVerifying mock dataset...")
        
        speaker_dirs = sorted([d for d in self.output_dir.iterdir() if d.is_dir()])
        
        for speaker_dir in speaker_dirs:
            speaker_name = speaker_dir.name
            
            # Count files
            content_files = list(speaker_dir.glob('*_content.npy'))
            speaker_files = list(speaker_dir.glob('*_speaker.npy'))
            accent_files = list(speaker_dir.glob('*_accent.npy'))
            prosody_files = list(speaker_dir.glob('*_prosody.pkl'))
            mel_files = list(speaker_dir.glob('*_mel.npy'))
            
            if all(len(f) == self.num_sentences for f in [
                content_files, speaker_files, accent_files, prosody_files, mel_files
            ]):
                print(f"  ✓ {speaker_name}: All files present ({self.num_sentences} sentences)")
            else:
                print(f"  ✗ {speaker_name}: Missing files!")
                print(f"    Content: {len(content_files)}, Speaker: {len(speaker_files)}, "
                      f"Accent: {len(accent_files)}, Prosody: {len(prosody_files)}, "
                      f"Mel: {len(mel_files)}")
        
        print("\n✓ Verification complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate mock embedding dataset')
    parser.add_argument('--output_dir', default='data/mock_embeddings',
                       help='Output directory for mock embeddings')
    parser.add_argument('--num_sentences', type=int, default=5,
                       help='Number of sentences to generate')
    parser.add_argument('--num_speakers', type=int, default=3,
                       help='Number of speakers to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    generator = MockDatasetGenerator(
        output_dir=args.output_dir,
        num_sentences=args.num_sentences,
        num_speakers=args.num_speakers,
        seed=args.seed
    )
    
    generator.generate_all()
    generator.verify()