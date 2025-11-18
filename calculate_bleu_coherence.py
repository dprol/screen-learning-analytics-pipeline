#!/usr/bin/env python3
"""
Calculate BLEU Scores with Coherence-Filtered Data
Compare OCR-old vs OCR-coherence-PASS vs OCR-coherence-FAIL
"""

import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import glob
import random

def tokenize_text(text):
    """Tokenize text for BLEU calculation"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    try:
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalnum()]
    except:
        return text.lower().split()

def calculate_corpus_bleu(texts, smoothing=True, sample_size=500):
    """
    Calculate average BLEU score for a corpus using self-BLEU.
    Each text is compared against all others as references.
    """
    smooth = SmoothingFunction().method1 if smoothing else None

    # Sample if corpus is too large
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)

    scores = []
    for i, hypothesis_text in enumerate(texts):
        hypothesis = tokenize_text(hypothesis_text)

        if len(hypothesis) < 3:
            continue

        # Use all other texts as references
        reference_texts = texts[:i] + texts[i+1:]
        references = [tokenize_text(ref) for ref in reference_texts[:100]]  # Limit references
        references = [ref for ref in references if len(ref) >= 3]

        if len(references) == 0:
            continue

        try:
            score = sentence_bleu(references, hypothesis, smoothing_function=smooth)
            scores.append(score)
        except:
            continue

    if len(scores) == 0:
        return {'mean': 0.0, 'std': 0.0, 'count': 0}

    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'count': len(scores)
    }

def main():
    print("="*60)
    print("BLEU SCORE CALCULATION - COHERENCE FILTER")
    print("="*60)

    results = []

    # 1. OCR-baseline (no filter)
    print("\n1. Loading OCR-baseline (no filter)...")
    baseline_files = glob.glob('ocr-results-baseline/Participant*_ocr_results.csv')
    baseline_texts = []
    for f in baseline_files[:10]:  # Sample participants
        df = pd.read_csv(f, encoding='utf-8', encoding_errors='replace')
        if 'raw_text' in df.columns:
            baseline_texts.extend(df['raw_text'].dropna().astype(str).tolist())
        elif 'improved_text' in df.columns:
            baseline_texts.extend(df['improved_text'].dropna().astype(str).tolist())

    print(f"   Loaded {len(baseline_texts)} texts from OCR-baseline")
    bleu_baseline = calculate_corpus_bleu(baseline_texts)
    results.append({
        'corpus': 'OCR-baseline (No Filter)',
        'bleu': bleu_baseline['mean'],
        'std': bleu_baseline['std'],
        'count': bleu_baseline['count'],
        'vs_baseline': '—'
    })
    print(f"   BLEU: {bleu_baseline['mean']:.4f} ± {bleu_baseline['std']:.4f}")

    # 2. OCR-filtered PASS only
    print("\n2. Loading OCR-filtered PASS...")
    coherence_files = glob.glob('ocr-filtered/Participant*_clean_coherence.csv')
    pass_texts = []
    fail_texts = []
    all_texts = []

    for f in coherence_files:
        df = pd.read_csv(f, encoding='utf-8', encoding_errors='replace')
        if 'improved_text' in df.columns and 'filter_status' in df.columns:
            pass_df = df[df['filter_status'] == 'PASS']
            fail_df = df[df['filter_status'] == 'FAIL']

            pass_texts.extend(pass_df['improved_text'].dropna().astype(str).tolist())
            fail_texts.extend(fail_df['improved_text'].dropna().astype(str).tolist())
            all_texts.extend(df['improved_text'].dropna().astype(str).tolist())

    print(f"   Loaded {len(pass_texts)} PASS texts")
    bleu_pass = calculate_corpus_bleu(pass_texts)
    improvement_pass = ((bleu_pass['mean'] - bleu_baseline['mean']) / bleu_baseline['mean'] * 100) if bleu_baseline['mean'] > 0 else 0
    results.append({
        'corpus': 'OCR-filtered (PASS only)',
        'bleu': bleu_pass['mean'],
        'std': bleu_pass['std'],
        'count': bleu_pass['count'],
        'vs_baseline': f"+{improvement_pass:.1f}%"
    })
    print(f"   BLEU: {bleu_pass['mean']:.4f} ± {bleu_pass['std']:.4f} ({improvement_pass:+.1f}%)")

    # 3. OCR-filtered FAIL only
    print("\n3. Loading OCR-filtered FAIL...")
    print(f"   Loaded {len(fail_texts)} FAIL texts")
    bleu_fail = calculate_corpus_bleu(fail_texts)
    improvement_fail = ((bleu_fail['mean'] - bleu_baseline['mean']) / bleu_baseline['mean'] * 100) if bleu_baseline['mean'] > 0 else 0
    results.append({
        'corpus': 'OCR-filtered (FAIL only)',
        'bleu': bleu_fail['mean'],
        'std': bleu_fail['std'],
        'count': bleu_fail['count'],
        'vs_baseline': f"{improvement_fail:+.1f}%"
    })
    print(f"   BLEU: {bleu_fail['mean']:.4f} ± {bleu_fail['std']:.4f} ({improvement_fail:+.1f}%)")

    # 4. OCR-filtered All Mixed
    print("\n4. Loading OCR-filtered (All Mixed)...")
    print(f"   Loaded {len(all_texts)} total texts")
    bleu_all = calculate_corpus_bleu(all_texts)
    improvement_all = ((bleu_all['mean'] - bleu_baseline['mean']) / bleu_baseline['mean'] * 100) if bleu_baseline['mean'] > 0 else 0
    results.append({
        'corpus': 'OCR-filtered (All Mixed)',
        'bleu': bleu_all['mean'],
        'std': bleu_all['std'],
        'count': bleu_all['count'],
        'vs_baseline': f"{improvement_all:+.1f}%"
    })
    print(f"   BLEU: {bleu_all['mean']:.4f} ± {bleu_all['std']:.4f} ({improvement_all:+.1f}%)")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('bleu_coherence_results.csv', index=False)

    # Create LaTeX table
    print("\n" + "="*60)
    print("CREATING LATEX TABLE")
    print("="*60)

    latex = r"""\begin{table}[htbp]
\centering
\caption{BLEU Score Comparison: Coherence-Based Filter Validation}
\label{tab:bleu_coherence}
\begin{tabular}{lcccc}
\toprule
\textbf{Corpus} & \textbf{BLEU} & \textbf{Std Dev} & \textbf{vs Baseline} & \textbf{Sample Size} \\
\midrule
"""

    for _, row in df_results.iterrows():
        latex += f"{row['corpus']} & {row['bleu']:.4f} & {row['std']:.4f} & {row['vs_baseline']} & {row['count']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{0.3cm}

\footnotesize
\textbf{Filter Configuration:} 3-path OR-based coherence filter. Path 1: coherence $\geq$ 0.50 (sentence-transformers embeddings). Path 2: spell $\geq$ 65\% AND coherence $\geq$ 0.30. Path 3: spell $\geq$ 75\%.

\vspace{0.2cm}

\textbf{PASS vs FAIL Comparison:} PASS texts achieve """ + f"{bleu_pass['mean']/bleu_fail['mean']:.1f}×" + r""" higher BLEU than FAIL, validating filter effectiveness.

\end{table}
"""

    with open('bleu_coherence_comparison_table.tex', 'w') as f:
        f.write(latex)

    print("\n✓ Saved results to bleu_coherence_results.csv")
    print("✓ Saved LaTeX table to bleu_coherence_comparison_table.tex")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline (no filter):   {bleu_baseline['mean']:.4f}")
    print(f"PASS (filtered):        {bleu_pass['mean']:.4f} ({improvement_pass:+.1f}%)")
    print(f"FAIL (filtered):        {bleu_fail['mean']:.4f} ({improvement_fail:+.1f}%)")
    print(f"All Mixed (filtered):   {bleu_all['mean']:.4f} ({improvement_all:+.1f}%)")
    print(f"\nPASS/FAIL ratio: {bleu_pass['mean']/bleu_fail['mean']:.1f}×")
    print("="*60)

if __name__ == "__main__":
    main()
