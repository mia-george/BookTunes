# BookTunes

An AI powered system that generates emotion aligned soundtracks for books by analyzing text on a page by page basis and selecting suitable music pieces from a predefined library.

## Team
Mia George, Aryaan Peshoton, Srija Kethireddy

## Overview

BookTunes creates an immersive reading experience by matching the emotional content of literary text with appropriate background music. The system uses modern language models to extract emotional representations from text and retrieves music tracks through similarity search in an embedding space.

### Emotion Categories

The system works with nine emotion categories mapped to both text and music:

- **Amazement**: This text expresses wonder, awe, astonishment, breathtaking discovery, miraculous revelation, epic grandeur, magnificent spectacle, inspiring triumph, jaw-dropping beauty, cosmic scale, overwhelming majesty.

- **Solemnity**: This text expresses solemnity, reverence, dignity, sacred ceremony, spiritual depth, formal gravity, serious contemplation, religious devotion, ritual importance, profound respect, weighty significance.

- **Tenderness**: This text expresses tenderness, warmth, gentleness, intimate affection, caring love, emotional closeness, soft vulnerability, protective nurturing, delicate sensitivity, heartfelt compassion, sweet devotion.

- **Nostalgia**: This text expresses nostalgia, longing for the past, bittersweet memories, wistful reminiscence, yearning for old times, sentimental reflection, missing what was, remembering yesterday, looking back fondly, homesickness, lost innocence.

- **Calmness**: This text expresses calmness, peace, tranquility, serenity, stillness, relaxation, quietness, meditation, gentle ease, untroubled mind, soothing comfort, restful contentment, peaceful silence.

- **Power**: This text expresses power, strength, energy, boldness, heroism, triumph, victory, confidence, determination, dominance, intensity, explosive force, unstoppable drive, fierce courage, commanding presence.

- **Joyful activation**: 'This text expresses joy, happiness, excitement, playfulness, fun, celebration, cheerfulness, enthusiasm, lively energy, dancing spirit, positive vibes, exuberance, delightful pleasure, upbeat optimism.',

- **Tension**:'This text expresses tension, stress, anxiety, nervousness, unease, worry, suspense, fear, restlessness, discomfort, apprehension, dread, agitation, uncertainty, foreboding danger, tightness, pressure.
Sadness: This text expresses sadness, sorrow, grief, melancholy, depression, heartbreak, loss, despair, loneliness, pain, crying, suffering, hopelessness, emptiness, mourning, tearful anguish, emotional hurt.

## Technical Pipeline

<img width="666" height="251" alt="Screenshot 2025-11-17 at 23 45 24" src="https://github.com/user-attachments/assets/2232abc8-fc6f-42e9-a7df-bd57178dc533" />
## 1. CNN-Based Emotion Recognition Model
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

## Future Improvements

- Enhance music quality of generated tracks
- Update to our CNN model since we saw there may be overfitting
