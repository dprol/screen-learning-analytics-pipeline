#!/usr/bin/env python3
"""
Apply trained behavior classifier to predict codes for all OCR-clean data.

Creates a new dataset with predicted behavior codes for each frame.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import Counter

def load_model(model_dir='models'):
    """Load trained classifier, vectorizer, and label names."""
    model_path = Path(model_dir)

    with open(model_path / 'behavior_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    with open(model_path / 'behavior_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open(model_path / 'behavior_labels.pkl', 'rb') as f:
        label_names = pickle.load(f)

    return classifier, vectorizer, label_names

def predict_behaviors_for_participant(participant_id, classifier, vectorizer, label_names,
                                     ocr_dir='ocr-filtered', output_dir='ocr-filtered-with-behaviors'):
    """
    Predict behavior codes for a single participant's OCR data.

    Returns:
        DataFrame with added behavior prediction columns
    """
    # Load OCR data from coherence-filtered corpus
    ocr_path = Path(ocr_dir) / f'{participant_id}_clean_coherence.csv'

    if not ocr_path.exists():
        print(f"  ⚠ {participant_id}: OCR file not found")
        return None

    df = pd.read_csv(ocr_path)
    print(f"  Processing {participant_id}: {len(df)} frames")

    # Only predict on PASS frames (coherence filter with 3-path OR logic)
    pass_mask = df['filter_status'] == 'PASS'
    pass_indices = df[pass_mask].index

    if len(pass_indices) == 0:
        print(f"    ⚠ No PASS frames found")
        return None

    # Get texts for prediction
    texts = df.loc[pass_indices, 'improved_text'].fillna('').tolist()

    # Vectorize
    X = vectorizer.transform(texts)

    # Predict
    y_pred = classifier.predict(X)
    y_pred_proba = np.array([
        estimator.predict_proba(X)[:, 1]
        for estimator in classifier.estimators_
    ]).T  # Shape: (n_samples, n_labels)

    # Convert predictions to behavior codes
    predicted_codes = []
    predicted_probabilities = []

    for i in range(len(y_pred)):
        # Get codes where prediction = 1
        codes = [label_names[j] for j in range(len(label_names)) if y_pred[i, j] == 1]

        # Get probabilities for those codes
        probs = {label_names[j]: y_pred_proba[i, j] for j in range(len(label_names))}

        predicted_codes.append(','.join(codes) if codes else 'None')
        predicted_probabilities.append(str(probs))

    # Add predictions to dataframe (only for PASS frames)
    df['predicted_behaviors'] = 'None'
    df['behavior_probabilities'] = '{}'

    df.loc[pass_indices, 'predicted_behaviors'] = predicted_codes
    df.loc[pass_indices, 'behavior_probabilities'] = predicted_probabilities

    # Statistics
    behavior_counts = Counter()
    for codes_str in predicted_codes:
        if codes_str != 'None':
            codes = codes_str.split(',')
            behavior_counts.update(codes)

    print(f"    ✓ Predicted behaviors for {len(pass_indices)} PASS frames")
    print(f"    Top behaviors: {dict(behavior_counts.most_common(3))}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_file = output_path / f'{participant_id}_coherence_with_behaviors.csv'
    df.to_csv(output_file, index=False)
    print(f"    Saved to {output_file}")

    return df

def generate_summary_statistics(all_dfs, label_names, output_dir='ocr-filtered-with-behaviors'):
    """Generate summary statistics across all participants."""

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall behavior distribution
    all_behaviors = Counter()
    total_frames = 0
    total_pass_frames = 0
    frames_with_predictions = 0

    for participant_id, df in all_dfs.items():
        total_frames += len(df)
        pass_frames = (df['filter_status'] == 'PASS').sum()
        total_pass_frames += pass_frames

        for codes_str in df['predicted_behaviors']:
            if codes_str != 'None':
                frames_with_predictions += 1
                codes = codes_str.split(',')
                all_behaviors.update(codes)

    print(f"\nTotal frames processed: {total_frames}")
    print(f"Total PASS frames: {total_pass_frames} ({total_pass_frames/total_frames*100:.1f}%)")
    print(f"Frames with predicted behaviors: {frames_with_predictions} ({frames_with_predictions/total_pass_frames*100:.1f}% of PASS)")

    print(f"\nBehavior code distribution:")
    for code, count in all_behaviors.most_common():
        pct = count / total_pass_frames * 100
        print(f"  {code:12s}: {count:5d} ({pct:5.1f}% of PASS frames)")

    # Save summary
    summary_df = pd.DataFrame([
        {'behavior': code, 'count': count, 'percentage': count/total_pass_frames*100}
        for code, count in all_behaviors.most_common()
    ])

    summary_path = Path(output_dir) / 'behavior_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved to {summary_path}")

    # Per-participant summary
    participant_summary = []
    for participant_id, df in all_dfs.items():
        pass_frames = (df['filter_status'] == 'PASS').sum()

        behaviors = Counter()
        for codes_str in df['predicted_behaviors']:
            if codes_str != 'None':
                codes = codes_str.split(',')
                behaviors.update(codes)

        row = {
            'participant': participant_id,
            'total_frames': len(df),
            'pass_frames': pass_frames,
            'frames_with_behaviors': sum(1 for c in df['predicted_behaviors'] if c != 'None')
        }

        # Add count for each behavior
        for label in label_names:
            row[f'count_{label}'] = behaviors.get(label, 0)

        participant_summary.append(row)

    participant_df = pd.DataFrame(participant_summary)
    participant_path = Path(output_dir) / 'participant_behavior_summary.csv'
    participant_df.to_csv(participant_path, index=False)
    print(f"✓ Per-participant summary saved to {participant_path}")

def main():
    print("=" * 80)
    print("PREDICTING BEHAVIORS FOR ALL PARTICIPANTS")
    print("=" * 80)

    # Load trained model
    print("\nLoading trained model...")
    classifier, vectorizer, label_names = load_model()
    print(f"✓ Model loaded with {len(label_names)} behavior codes:")
    print(f"  {', '.join(label_names)}")

    # List of participants
    participants = [
        'Participant9', 'Participant12', 'Participant14', 'Participant22',
        'Participant23', 'Participant24', 'Participant25', 'Participant26',
        'Participant27', 'Participant28', 'Participant29', 'Participant30',
        'Participant32', 'Participant33', 'Participant34', 'Participant35'
    ]

    print(f"\nProcessing {len(participants)} participants...")
    print()

    # Predict for each participant
    all_dfs = {}

    for participant_id in participants:
        df = predict_behaviors_for_participant(
            participant_id, classifier, vectorizer, label_names
        )
        if df is not None:
            all_dfs[participant_id] = df

    # Generate summary statistics
    if all_dfs:
        generate_summary_statistics(all_dfs, label_names)

    print("\n" + "=" * 80)
    print("✓ PREDICTION COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to: ocr-filtered-with-behaviors/")
    print(f"  - Individual CSV files with predictions")
    print(f"  - behavior_summary.csv (overall statistics)")
    print(f"  - participant_behavior_summary.csv (per-participant breakdown)")
    print(f"\nUsing 3-path coherence filter:")
    print(f"  Path 1: coherence ≥ 0.50 AND spell ≥ 50%")
    print(f"  Path 2: spell ≥ 65% AND coherence ≥ 0.30")
    print(f"  Path 3: spell ≥ 75%")

if __name__ == '__main__':
    main()
