# Screen Learning Analytics Pipeline

Cost-effective, open-source pipeline for analyzing screen recordings in educational contexts. Features OCR extraction, 3-path coherence filtering, semantic screen transition detection, and automated behavior classification.

**Impact**: Reduces $1000+ commercial OCR costs to near-zero and 90+ hours of manual coding to minutes.

## Overview

Traditional approaches to analyzing screen recordings face three critical challenges:
1. **Cost**: Commercial OCR services (Google Vision API) cost $1.50 per 1,000 images (~$936 for our 67K frames)
2. **Manual labor**: Behavior coding requires 90+ hours for just 16 participants
3. **No end-to-end solution**: Existing tools only handle isolated steps

This pipeline provides a fully automated, open-source solution that democratizes screen-based learning analytics.

## Pipeline Stages

### 1. Preprocessing
- Extract frames at 1-second intervals from screen recordings
- Apply grayscale conversion, adaptive thresholding, morphological operations
- Optimize for OCR accuracy

### 2. OCR + Filtering
- **OCR**: Use Tesseract (open-source) to extract text from 67K+ frames
- **3-Path Filter**: Novel filtering approach combining spell-checking and semantic coherence
  - **Path 1**: High coherence (≥50%) + minimum spelling (≥50%)
  - **Path 2**: Moderate coherence (≥30%) + good spelling (≥65%)
  - **Path 3**: Lower coherence + excellent spelling (≥75%) — for code snippets
- **Result**: 18.7% pass rate (6,674 high-quality frames from 35,601 extractions)

### 3. Screen Transition Detection
- Use SBERT embeddings (384-dimensional semantic vectors)
- Detect screen changes via cosine similarity (threshold: 0.7)
- Reconstruct learner navigation patterns across content
- **Result**: 6,000+ high-quality screen transitions detected

### 4. Clustering + Behavior Detection
- **Clustering**: Identify learner patterns by screen changes/minute
  - Focused learners (<19 changes/min)
  - Moderate learners (19-24 changes/min)
  - High-activity learners (>24 changes/min)
- **Behavior Classification**: Random Forest multi-label classifier
  - Trained on 551 manually annotated segments
  - TF-IDF features (1,000 dimensions, 1-3 grams)
  - Predicts 15 behavior codes (Quiz, Search Examples, Search Theory, AI Examples, Website, etc.)
  - **F1 score**: 0.73

### 5. Validation
- **BLEU scores** for text coherence:
  - OCR-baseline: 0.0911
  - OCR-filtered (mixed): 0.1613 (+77%)
  - PASS-only: 0.2783 (+205%, 2× better than FAIL texts)

## Key Results

- ✅ **Cost reduction**: $936 → near-zero (Tesseract open-source)
- ✅ **Scale enablement**: 90+ hours → minutes
- ✅ **Quality improvement**: 205% BLEU score increase (PASS vs baseline)
- ✅ **Automation**: Multi-label behavior classification (F1=0.73)

## Installation

```bash
# Clone repository
git clone https://github.com/dprol/screen-learning-analytics-pipeline.git
cd screen-learning-analytics-pipeline

# Install dependencies
pip install -r requirements.txt

# Download models
python -m spacy download en_core_web_sm
```

## Usage

### Step 1: Apply Coherence Filter
```bash
python apply_coherence_filter.py \
  --input ocr-results/ \
  --output ocr-filtered/
```

### Step 2: Detect Screen Transitions
```bash
python visualize_transitions.py \
  --input ocr-filtered/ \
  --output screen-analysis/
```

### Step 3: Predict Behaviors
```bash
python predict_behaviors.py \
  --input ocr-filtered/ \
  --model models/behavior_classifier.pkl \
  --output ocr-filtered-with-behaviors/
```

### Step 4: Analyze Clusters
```bash
python analyze_cluster_performance.py \
  --input ocr-filtered-with-behaviors/ \
  --output analysis/
```

### Step 5: Calculate Validation Metrics
```bash
python calculate_bleu_coherence.py \
  --baseline ocr-results-baseline/ \
  --filtered ocr-filtered/ \
  --output validation/
```

## Core Components

| Script | Description | Stage |
|--------|-------------|-------|
| `apply_coherence_filter.py` | 3-path coherence filter (spell + semantic) | 2 |
| `improve_ocr_quality.py` | OCR preprocessing and quality improvement | 1-2 |
| `visualize_transitions.py` | Screen transition detection (SBERT) | 3 |
| `predict_behaviors.py` | Multi-label behavior classification | 4 |
| `analyze_cluster_performance.py` | Learner clustering analysis | 4 |
| `calculate_bleu_coherence.py` | BLEU validation metrics | 5 |
| `create_behavior_visualizations.py` | Behavior pattern visualizations | 4 |
| `create_coherence_filter_figure.py` | Filter architecture diagram | 2 |

## Requirements

- Python 3.8+
- Tesseract OCR 4.0+
- Dependencies: see `requirements.txt`

```
opencv-python>=4.5.0
pytesseract>=0.3.8
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
spacy>=3.0.0
nltk>=3.6.0
```

## Dataset

The pipeline was validated on:
- **16 participants** (programming education context)
- **67,297 frames** (18.7 hours of recordings)
- **35,601 text extractions** (98.2% OCR success rate)
- **6,674 PASS frames** (18.7% pass rate after filtering)
- **1,533 behavior predictions** across 5 major codes

## Limitations

1. **Behavior code coverage**: Only 5 out of 15 codes had sufficient training data
2. **OCR error correction**: Filter identifies quality but doesn't correct errors (only pass/fail)
3. **Domain specificity**: Validated on programming education; other domains may need adaptation

## Future Work

- LLM-based OCR reconstruction for error correction
- Real-time deployment for live classroom monitoring
- Expanded behavior code coverage with active learning
- Multi-language support

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

- Daniel Prol - dprol@[institution].edu
- Project: https://github.com/dprol/screen-learning-analytics-pipeline

## Acknowledgments

This work benefits three communities:
- **Researchers**: Open-source toolkit for screen analytics
- **Educators**: Real-time insights into student navigation patterns
- **The field**: Reproducible framework lowering barriers to studying digital learning at scale
