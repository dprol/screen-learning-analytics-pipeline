#!/usr/bin/env python3
"""
Train a supervised classifier to infer behavior codes from OCR text.

Uses manual annotations from .eaf files as training labels and OCR-clean text as features.
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import pickle

def parse_eaf_codes(eaf_path):
    """
    Parse ELAN .eaf file to extract behavior codes with timestamps.

    Returns:
        list of dicts with 'start_time', 'end_time', 'codes' (list)
    """
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    # Extract time slots
    time_slots = {}
    for time_slot in root.findall('.//TIME_SLOT'):
        slot_id = time_slot.get('TIME_SLOT_ID')
        time_value = time_slot.get('TIME_VALUE')
        if time_value:
            time_slots[slot_id] = int(time_value) / 1000.0

    # Extract code annotations
    code_segments = []
    for tier in root.findall('.//TIER'):
        if tier.get('TIER_ID') == 'Codes':
            for annotation in tier.findall('.//ALIGNABLE_ANNOTATION'):
                start_slot = annotation.get('TIME_SLOT_REF1')
                end_slot = annotation.get('TIME_SLOT_REF2')
                annotation_value = annotation.find('ANNOTATION_VALUE')

                if annotation_value is not None and annotation_value.text:
                    if start_slot in time_slots and end_slot in time_slots:
                        # Parse comma-separated codes
                        codes = [c.strip() for c in annotation_value.text.split(',')]

                        code_segments.append({
                            'start_time': time_slots[start_slot],
                            'end_time': time_slots[end_slot],
                            'codes': codes
                        })

    return code_segments

def aggregate_ocr_in_window(ocr_df, start_time, end_time):
    """
    Aggregate all OCR text within a time window.

    Args:
        ocr_df: DataFrame with OCR results
        start_time: Window start (seconds)
        end_time: Window end (seconds)

    Returns:
        Concatenated text from all frames in window
    """
    window_frames = ocr_df[
        (ocr_df['timestamp_seconds'] >= start_time) &
        (ocr_df['timestamp_seconds'] <= end_time)
    ]

    # Concatenate all text (prefer PASS texts)
    pass_frames = window_frames[window_frames['passes_quality_filter'] == 'Yes']

    if len(pass_frames) > 0:
        texts = pass_frames['improved_text'].tolist()
    else:
        texts = window_frames['improved_text'].tolist()

    # Join all texts with space
    combined_text = ' '.join([str(t) for t in texts if pd.notna(t)])

    return combined_text

def create_training_data(participant_ids, ocr_dir='ocr-clean', eaf_dir='ML Aayush videos project/Aayush'):
    """
    Create training dataset from multiple participants.

    Returns:
        X (texts), y (multi-label binary matrix), label_names
    """
    all_texts = []
    all_labels = []
    all_code_sets = set()

    print("Creating training dataset...")

    for participant_id in participant_ids:
        print(f"\n  Processing {participant_id}...")

        # Load EAF codes
        eaf_path = Path(eaf_dir) / f'{participant_id}.eaf'
        if not eaf_path.exists():
            print(f"    ⚠ .eaf not found, skipping")
            continue

        try:
            code_segments = parse_eaf_codes(eaf_path)
            print(f"    Found {len(code_segments)} coded segments")
        except Exception as e:
            print(f"    ✗ Error parsing .eaf: {e}")
            continue

        # Load OCR data
        ocr_path = Path(ocr_dir) / f'{participant_id}_clean.csv'
        if not ocr_path.exists():
            print(f"    ⚠ OCR clean file not found, skipping")
            continue

        ocr_df = pd.read_csv(ocr_path)
        print(f"    Loaded {len(ocr_df)} OCR frames")

        # Create training examples
        for segment in code_segments:
            # Aggregate OCR text in this time window
            text = aggregate_ocr_in_window(ocr_df, segment['start_time'], segment['end_time'])

            if text.strip():
                all_texts.append(text)
                all_labels.append(segment['codes'])
                all_code_sets.update(segment['codes'])

        print(f"    Created {len(code_segments)} training examples")

    print(f"\n✓ Total training examples: {len(all_texts)}")
    print(f"✓ Unique behavior codes: {len(all_code_sets)}")
    print(f"  Codes: {sorted(all_code_sets)}")

    # Convert to multi-label binary matrix
    label_names = sorted(all_code_sets)
    y_binary = np.zeros((len(all_labels), len(label_names)), dtype=int)

    for i, codes in enumerate(all_labels):
        for code in codes:
            if code in label_names:
                j = label_names.index(code)
                y_binary[i, j] = 1

    return all_texts, y_binary, label_names

def train_classifier(X_train, y_train, label_names):
    """
    Train a multi-label classifier for behavior codes.

    Returns:
        Trained model and vectorizer
    """
    print("\nTraining classifier...")

    # Text vectorization with TF-IDF
    print("  Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 3),  # unigrams, bigrams, trigrams
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    print(f"  Feature dimensions: {X_train_vec.shape}")

    # Train multi-label Random Forest
    print("  Training Random Forest classifier...")
    base_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    # Multi-label wrapper
    classifier = MultiOutputClassifier(base_classifier, n_jobs=-1)
    classifier.fit(X_train_vec, y_train)

    print("  ✓ Training complete")

    return classifier, vectorizer

def evaluate_classifier(classifier, vectorizer, X_test, y_test, label_names):
    """Evaluate classifier performance."""
    print("\nEvaluating classifier...")

    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)

    # Overall metrics
    print(f"\n  Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")
    print(f"  Subset Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Per-class metrics
    print("\n  Per-class Performance:")
    for i, label in enumerate(label_names):
        if y_test[:, i].sum() > 0:  # Only if label exists in test set
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test[:, i], y_pred[:, i], average='binary', zero_division=0
            )
            print(f"    {label:12s}: Precision={precision:.3f}, Recall={recall:.3f}, "
                  f"F1={f1:.3f}, Support={int(y_test[:, i].sum())}")

    return y_pred

def save_model(classifier, vectorizer, label_names, output_dir='models'):
    """Save trained model and metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save classifier
    with open(output_path / 'behavior_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    # Save vectorizer
    with open(output_path / 'behavior_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save label names
    with open(output_path / 'behavior_labels.pkl', 'wb') as f:
        pickle.dump(label_names, f)

    print(f"\n✓ Model saved to {output_path}/")

def main():
    print("=" * 80)
    print("BEHAVIOR CODE CLASSIFIER TRAINING")
    print("=" * 80)

    # List of participants with .eaf annotations
    participants = [
        'Participant9', 'Participant12', 'Participant14', 'Participant22',
        'Participant23', 'Participant24', 'Participant25', 'Participant26',
        'Participant27', 'Participant28', 'Participant29', 'Participant30',
        'Participant32', 'Participant33', 'Participant34', 'Participant35'
    ]

    # Create training data
    X, y, label_names = create_training_data(participants)

    if len(X) == 0:
        print("✗ No training data found!")
        return

    # Split train/test
    print("\nSplitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training examples: {len(X_train)}")
    print(f"  Test examples: {len(X_test)}")

    # Train classifier
    classifier, vectorizer = train_classifier(X_train, y_train, label_names)

    # Evaluate
    y_pred = evaluate_classifier(classifier, vectorizer, X_test, y_test, label_names)

    # Save model
    save_model(classifier, vectorizer, label_names)

    # Show some predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)

    for i in range(min(5, len(X_test))):
        print(f"\nExample {i+1}:")
        print(f"  Text: {X_test[i][:150]}...")

        true_codes = [label_names[j] for j in range(len(label_names)) if y_test[i, j] == 1]
        pred_codes = [label_names[j] for j in range(len(label_names)) if y_pred[i, j] == 1]

        print(f"  True labels: {', '.join(true_codes)}")
        print(f"  Predicted:   {', '.join(pred_codes)}")
        print(f"  Match: {'✓' if set(true_codes) == set(pred_codes) else '✗'}")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
