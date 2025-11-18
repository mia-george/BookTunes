# BookTunes

An AI powered system that generates emotion aligned soundtracks for books by analyzing text on a page by page basis and selecting suitable music pieces from a predefined library.

## Team
Mia George, Aryaan Peshoton, Srija Kethireddy

## Overview

BookTunes creates an immersive reading experience by matching the emotional content of literary text with appropriate background music. The system uses modern language models to extract emotional representations from text and retrieves music tracks through similarity search in an embedding space.

### Emotion Categories

The system works with nine emotion categories mapped to both text and music:

- **Amazement** – Feeling of wonder and happiness
- **Solemnity** – Feeling of transcendence, inspiration, thrills
- **Tenderness** – Sensuality, affection, feeling of love
- **Nostalgia** – Dreamy, melancholic, sentimental feelings
- **Calmness** – Relaxation, serenity, meditativeness
- **Power** – Feeling strong, heroic, triumphant, energetic
- **Joyful activation** – Feels like dancing, bouncy feeling, animated, amused
- **Tension** – Nervous, impatient, irritated
- **Sadness** – Depressed, sorrowful

## Technical Pipeline

### 1. CNN-Based Emotion Recognition Model
- ResNet-based CNN trained to predict 9 emotion classes
- Extracts 512D embeddings from penultimate layer
- Reduced to 128D via PCA for efficient retrieval
- Embeddings encode spectral structure, rhythmic texture, harmonic content, and emotion-correlated acoustic cues

### 2. Semantic Text-to-Emotion Mapping
- Uses Sentence-BERT (MiniLM-L6-v2) to embed text
- Matches text to emotion descriptions via cosine similarity
- Enhanced with:
  - Top-K emotion sparsification
  - Temperature scaling (0.25)
  - L2 normalization
- Produces smooth and nuanced emotion distributions

### 3. FAISS-Based Retrieval
- Fast and accurate nearest neighbor search
- Song embeddings reduced from 512D to 128D using PCA
- Penalizes music tracks with low emotional variance
- Ensures retrieved audio reflects the correct emotion of text

### 4. Audio Blending Engine
- **Pitch shifting** – Harmonic alignment between tracks
- **BPM alignment** – Time stretching for tempo consistency
- **Harmonic/percussive source separation** – Isolates melody and rhythm components
- **LUFS loudness normalization** – Consistent perceived loudness across tracks
- Creates seamless transitions between music selections

## Datasets

### Music Dataset
[Emotify Dataset](https://www.projects.science.uu.nl/memotion/emotifydata/) from Utrecht University
- 400 emotion-annotated tracks
- Genres: pop, classical, rock, electronic
- Human-annotated emotion classifications across 9 categories

### Text Dataset
[Project Gutenberg](https://www.gutenberg.org/)
- Public domain books for testing
- Provides diverse literary text samples

## Requirements

- Python 3.8+
- CUDA-enabled GPU (recommended)
- PyTorch
- FAISS
- Sentence-Transformers
- librosa (for audio processing)
- scikit-learn

## Installation

```bash
git clone https://github.com/mia-george/BookTunes.git
cd BookTunes
pip install -r requirements.txt
```

## Usage

```python
aryaan: how to run the code, etc
```

## Future Improvements

- Enhance music quality of generated tracks
