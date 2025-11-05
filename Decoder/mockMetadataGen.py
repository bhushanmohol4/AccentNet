#!/usr/bin/env python3
"""
Complete mock dataset setup in one script.
No errors - ready to use!
"""

import numpy as np
import pickle
import random
from pathlib import Path

print("="*70)
print("COMPLETE MOCK DATASET SETUP")
print("="*70)

# ===== STEP 1: GENERATE MOCK EMBEDDINGS =====
print("\nSTEP 1: Generating mock embeddings...")

emb_dir = Path('data/mock_embeddings')
emb_dir.mkdir(parents=True, exist_ok=True)

num_speakers = 3
num_sentences = 5
seed = 42

np.random.seed(seed)

speaker_names = [f'speaker_{i:03d}' for i in range(num_speakers)]
sentence_ids = [f'sentence_{i:03d}' for i in range(num_sentences)]

for speaker_name in speaker_names:
    speaker_dir = emb_dir / speaker_name
    speaker_dir.mkdir(exist_ok=True)
    
    print(f"  Creating {speaker_name}/")
    
    for sentence_id in sentence_ids:
        # Content
        seq_len = np.random.randint(100, 200)
        content = np.random.randn(seq_len, 768).astype(np.float32)
        content = content / np.linalg.norm(content, axis=1, keepdims=True)
        np.save(speaker_dir / f'{sentence_id}_content.npy', content)
        
        # Speaker
        speaker = np.random.randn(192).astype(np.float32)
        speaker = speaker / np.linalg.norm(speaker)
        np.save(speaker_dir / f'{sentence_id}_speaker.npy', speaker)
        
        # Accent
        accent = np.random.randn(seq_len, 256).astype(np.float32)
        accent = accent / np.linalg.norm(accent, axis=1, keepdims=True)
        np.save(speaker_dir / f'{sentence_id}_accent.npy', accent)
        
        # Prosody
        mel_len = int(seq_len * np.random.uniform(2.5, 3.5))
        pitch = np.random.uniform(100, 250, mel_len).astype(np.float32)
        energy = np.random.uniform(0.3, 1.0, mel_len).astype(np.float32)
        duration = np.ones(seq_len, dtype=np.float32) * (mel_len / seq_len)
        
        prosody = {'pitch': pitch, 'energy': energy, 'duration': duration}
        with open(speaker_dir / f'{sentence_id}_prosody.pkl', 'wb') as f:
            pickle.dump(prosody, f)
        
        # Mel
        mel = np.random.randn(mel_len, 80).astype(np.float32)
        mel = (mel - mel.mean()) / (mel.std() + 1e-7)
        mel = (mel + 3) / 6
        mel = np.clip(mel, 0, 1)
        np.save(speaker_dir / f'{sentence_id}_mel.npy', mel)

print("✓ Embeddings generated successfully!")

# ===== STEP 2: GENERATE METADATA =====
print("\nSTEP 2: Generating metadata...")

# ✓ FIX: Access first speaker correctly!
first_speaker_dir = emb_dir / speaker_names[0]  # ✓ Use index , not whole list!

sentences = set()
for file in first_speaker_dir.glob('*_speaker.npy'):
    sentence_id = file.name.replace('_speaker.npy', '')
    sentences.add(sentence_id)

sentences = sorted(list(sentences))

pairs = []
for sentence_id in sentences:
    for source_speaker in speaker_names:
        for target_speaker in speaker_names:
            if source_speaker != target_speaker:
                pairs.append((source_speaker, target_speaker, sentence_id))

random.shuffle(pairs)

metadata_file = Path('data/metadata_mock.txt')
metadata_file.parent.mkdir(parents=True, exist_ok=True)

with open(metadata_file, 'w') as f:
    f.write("# speaker_source | speaker_target | sentence_id\n")
    f.write(f"# Total: {len(pairs)} pairs\n\n")
    for source, target, sentence in pairs:
        f.write(f"{source} | {target} | {sentence}\n")

print(f"✓ Metadata generated: {len(pairs)} pairs")

# ===== STEP 3: CREATE SPLITS =====
print("\nSTEP 3: Creating train/val/test splits...")

split_dir = Path('data/metadata_mock')
split_dir.mkdir(parents=True, exist_ok=True)

# Split by sentence
n_sentences = len(sentences)
train_end = int(n_sentences * 0.7)
val_end = train_end + int(n_sentences * 0.15)

train_sentences = set(sentences[:train_end])
val_sentences = set(sentences[train_end:val_end])
test_sentences = set(sentences[val_end:])

train_pairs = [p for p in pairs if p in train_sentences]
val_pairs = [p for p in pairs if p in val_sentences]
test_pairs = [p for p in pairs if p in test_sentences]

for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
    with open(split_dir / f'{split_name}.txt', 'w') as f:
        for source, target, sentence in split_pairs:
            f.write(f"{source} | {target} | {sentence}\n")
    print(f"  ✓ {split_name}: {len(split_pairs)} pairs")

print("\n" + "="*70)
print("✓ SETUP COMPLETE!")
print("="*70)
print("\nYou can now use:")
print(f"  Embeddings: {emb_dir}")
print(f"  Metadata: {split_dir}/train.txt (and val.txt, test.txt)")
print("\nTo train:")
print("  python train.py --embeddings_dir data/mock_embeddings \\")
print("                  --metadata_dir data/metadata_mock \\")
print("                  --epochs 5")