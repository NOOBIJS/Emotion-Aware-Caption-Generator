# Emotion-Aware Caption Generator from Image

> An end-to-end deep learning system that generates emotionally-rich image captions by combining Vision Transformer (ViT) image encoding, GPT-based caption generation, and DeepFace-powered real-time emotion detection.

**BUET CSE-330 Machine Learning Lab Project**

---

## Demo

| Input Image | Base Caption | Detected Emotion | Emotion-Aware Caption |
|---|---|---|---|
| Happy person smiling | "a person standing in a park" | 😊 happy | "a **joyful** person standing in a park" |
| Person looking worried | "a man sitting on a bench" | 😨 fear | "a **fearful** man sitting on a bench" |
| Couple arguing | "two people talking" | 😠 angry | "two **angry** people talking" |

---

## Architecture

```
Input Image
     │
     ├──────────────────────────────────────┐
     │                                      │
     ▼                                      ▼
┌─────────────┐                    ┌──────────────────┐
│  ViT Encoder│                    │  DeepFace        │
│  (timm)     │                    │  Emotion Detect  │
│             │                    │                  │
│ Image → 768d│                    │ 7 emotions:      │
│  embedding  │                    │ happy/sad/angry/ │
└──────┬──────┘                    │ fear/disgust/    │
       │                           │ surprise/neutral │
       ▼                           └────────┬─────────┘
┌─────────────┐                            │
│  Linear     │                            │ emotion label
│  Projection │                            │
│  768→512    │                            │
└──────┬──────┘                            │
       │                                   │
       ▼                                   │
┌─────────────┐                            │
│  GPT Decoder│                            │
│  (custom)   │                            │
│             │                            │
│ Image emb + │                            │
│ token seq → │                            │
│ base caption│                            │
└──────┬──────┘                            │
       │                                   │
       ▼                                   ▼
┌─────────────────────────────────────────────┐
│           NLTK Post-Processing              │
│  Insert emotion adjective before first noun │
│  e.g. "a [joyful] person in a park"        │
└────────────────────┬────────────────────────┘
                     │
                     ▼
          Emotion-Aware Caption ✅
```

---

## Evaluation Results

Evaluated on held-out test set from Flickr8k dataset:

| Metric | Score |
|--------|-------|
| BLEU-1 | 33.04 |
| BLEU-2 | 16.23 |
| BLEU-3 | 9.13  |
| BLEU-4 | 5.28  |
| METEOR | 22.59 |

Full metrics: [`results/evaluation_summary.csv`](results/evaluation_summary.csv)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Image Encoder | Vision Transformer (ViT) via `timm` |
| Caption Decoder | Custom GPT (Transformer decoder) |
| Emotion Detection | DeepFace + TensorFlow |
| Text Processing | NLTK (POS tagging) |
| Backend API | FastAPI + Uvicorn |
| Frontend | React.js |
| Containerization | Docker |
| Model Hosting | HuggingFace Hub (~1.91GB) |
| Training Data | Flickr8k dataset |

---

## Project Structure

```
Emotion-Aware-Caption-Generator/
├── backend/
│   ├── app.py                    # FastAPI server — /api/caption endpoint
│   ├── Dockerfile                # Container setup
│   ├── requirements.txt          # Python dependencies
│   └── model_classes/
│       ├── caption_model.py      # ViT + GPT combined model
│       ├── vit.py                # Vision Transformer encoder
│       ├── gpt.py                # GPT decoder (custom)
│       ├── tokenizer.py          # GPT-2 tokenizer wrapper
│       ├── emotion.py            # DeepFace emotion detection + caption fusion
│       └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── App.js                # Main React component
│   │   └── App.css               # Styling
│   └── package.json
├── notebooks/
│   ├── 01_training_pipeline.ipynb   # Full training code with outputs
│   ├── 02_evaluation.ipynb          # BLEU/METEOR evaluation
│   └── 03_custom_testing.ipynb      # Custom image testing
└── results/
    ├── evaluation_summary.csv        # Quantitative results
    ├── presentation.pdf              # Project presentation slides
    └── sample_images/               # Sample input images (7 emotions)
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+
- (Optional) CUDA-capable GPU

### 1. Run the Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the API server
# The model (~1.91GB) is auto-downloaded from HuggingFace on first run
uvicorn app:app --host 0.0.0.0 --port 7860
```

API will be available at `http://localhost:7860`

**Endpoints:**
- `GET /` — Health check
- `POST /api/caption` — Upload image → returns base caption, emotion, emotion-aware caption

### 2. Run with Docker

```bash
cd backend
docker build -t emotion-caption-api .
docker run -p 7860:7860 emotion-caption-api
```

### 3. Run the Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`

---

## API Usage

```bash
curl -X POST http://localhost:7860/api/caption \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "base_caption": "a person standing in a park",
  "detected_emotion": "happy",
  "emotion_aware_caption": "a joyful person standing in a park"
}
```

---

## How It Works

1. **Image Encoding** — The input image is resized to 224×224 and passed through a pretrained ViT encoder, producing a 768-dimensional embedding.

2. **Caption Generation** — The image embedding is projected to 512d and prepended to the GPT decoder's token sequence. The GPT decoder autoregressively generates a base caption token by token until `[EOS]`.

3. **Emotion Detection** — DeepFace analyzes the image to detect the dominant facial emotion from 7 classes: happy, sad, angry, surprise, fear, disgust, neutral.

4. **Emotion Fusion** — NLTK POS tagging identifies the first noun in the base caption. The emotion adjective (e.g., "joyful" for happy) is inserted before it, producing the final emotion-aware caption.

---

## Sample Results

Sample images for each emotion class are in [`results/sample_images/`](results/sample_images/).

![Happy](results/sample_images/happy_1.jpg) | ![Sad](results/sample_images/sad_1.jpg) | ![Angry](results/sample_images/angry_1.jpg)

---

## Team

**BUET CSE L-3 T-2 — Team ML Newbies**

Submitted for: CSE-330 Machine Learning Lab

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al.
- [DeepFace](https://github.com/serengil/deepface) — Facial analysis library
- Flickr8k Dataset for training and evaluation

---

## License

MIT License — free to use for educational purposes.
