#!/usr/bin/env python3
"""
Apply Coherence-Based Semantic Filter to OCR Results
Uses sentence-transformers embeddings to calculate semantic coherence
Combined with spell checking for 3-path OR-based filtering
"""

import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import glob
import os
from scipy.spatial.distance import cosine

# Try to use enchant, fallback to nltk
try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    ENCHANT_AVAILABLE = False
    print("⚠ enchant not available, using NLTK...")
    import nltk
    try:
        from nltk.corpus import words
        nltk.data.find('corpora/words')
    except LookupError:
        print("Downloading NLTK words corpus...")
        nltk.download('words', quiet=True)
        from nltk.corpus import words
    ENGLISH_WORDS = set(word.lower() for word in words.words())

# Suppress warnings
os.environ['TRANSFORMERS_NO_TF'] = '1'
import warnings
warnings.filterwarnings('ignore')

class CoherenceFilter:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the coherence filter with embedding model and spell checker"""
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

        print("Loading spell checker...")
        if ENCHANT_AVAILABLE:
            self.spell_checker = enchant.Dict("en_US")
            self.use_enchant = True
            print("✓ Using enchant dictionary")
        else:
            self.use_enchant = False
            print("✓ Using NLTK words dictionary")

        print("✓ CoherenceFilter initialized")

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text):
        """Get embedding for a single text"""
        encoded_input = self.tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embedding[0].numpy()

    def semantic_coherence_score(self, text):
        """
        Calculate semantic coherence score (0-1) based on intra-text embedding similarity.

        Strategy:
        - Split text into sentences/phrases
        - Calculate embeddings for each
        - Measure average pairwise cosine similarity
        - High similarity = coherent text, Low similarity = garbage/noise

        Returns:
            float: Coherence score 0-1 (higher = more coherent)
        """
        if not text or len(text.strip()) < 10:
            return 0.0

        # Split into sentences (by punctuation)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]

        # If too few sentences, split by phrases/chunks
        if len(sentences) < 2:
            tokens = text.split()
            if len(tokens) < 5:
                return 0.0
            # Create overlapping windows of 5 tokens
            sentences = [' '.join(tokens[i:i+5]) for i in range(len(tokens)-4)]

        if len(sentences) < 2:
            return 0.0

        # Get embeddings for all sentences
        try:
            embeddings = [self.get_embedding(s) for s in sentences[:10]]  # Limit to first 10 to avoid slowdown
        except Exception as e:
            print(f"Error encoding text: {e}")
            return 0.0

        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarities.append(sim)

        if not similarities:
            return 0.0

        # Return average similarity (coherence score)
        coherence = np.mean(similarities)
        return float(coherence)

    def spell_check_percentage(self, text):
        """Calculate percentage of correctly spelled words"""
        if not text or len(text.strip()) == 0:
            return 0.0

        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)

        if len(words) == 0:
            return 0.0

        if self.use_enchant:
            correct_count = sum(1 for word in words if self.spell_checker.check(word))
        else:
            # Use NLTK words
            correct_count = sum(1 for word in words if word.lower() in ENGLISH_WORDS)

        return (correct_count / len(words)) * 100

    def apply_filter(self, text):
        """
        Apply 3-path OR-based filter with coherence + spell checking

        3-PATH FILTER:
        Path 1: coherence ≥ 0.65 (very coherent text)
        Path 2: spell ≥ 65% AND coherence ≥ 0.45 (moderate coherence + good spelling)
        Path 3: spell ≥ 80% (excellent spelling, even if low coherence for code/technical text)

        Returns:
            dict: {
                'filter_status': 'PASS' or 'FAIL',
                'coherence_score': float,
                'spell_percentage': float,
                'pass_path': which path passed (1, 2, 3, or None)
            }
        """
        coherence = self.semantic_coherence_score(text)
        spell_pct = self.spell_check_percentage(text)

        # Path 1: High coherence
        if coherence >= 0.65:
            return {
                'filter_status': 'PASS',
                'coherence_score': coherence,
                'spell_percentage': spell_pct,
                'pass_path': 1
            }

        # Path 2: Good spelling + moderate coherence
        if spell_pct >= 65 and coherence >= 0.45:
            return {
                'filter_status': 'PASS',
                'coherence_score': coherence,
                'spell_percentage': spell_pct,
                'pass_path': 2
            }

        # Path 3: Excellent spelling (for code snippets, technical terms)
        if spell_pct >= 80:
            return {
                'filter_status': 'PASS',
                'coherence_score': coherence,
                'spell_percentage': spell_pct,
                'pass_path': 3
            }

        # Failed all paths
        return {
            'filter_status': 'FAIL',
            'coherence_score': coherence,
            'spell_percentage': spell_pct,
            'pass_path': None
        }

def process_participant_file(input_csv, output_csv, coherence_filter):
    """Process a single participant's OCR file with coherence filtering"""
    print(f"\nProcessing: {input_csv}")

    # Load OCR data (handle encoding errors)
    df = pd.read_csv(input_csv, encoding='utf-8', encoding_errors='replace')

    if 'ocr_text' not in df.columns:
        print(f"  ✗ No 'ocr_text' column found, skipping")
        return

    # Apply filter to each row
    results = []
    for idx, row in df.iterrows():
        text = str(row['ocr_text']) if pd.notna(row['ocr_text']) else ""
        result = coherence_filter.apply_filter(text)
        results.append(result)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} rows...")

    # Add results to dataframe
    df['filter_status'] = [r['filter_status'] for r in results]
    df['coherence_score'] = [r['coherence_score'] for r in results]
    df['spell_percentage'] = [r['spell_percentage'] for r in results]
    df['pass_path'] = [r['pass_path'] for r in results]

    # Save to output
    df.to_csv(output_csv, index=False)

    # Statistics
    total = len(df)
    passed = (df['filter_status'] == 'PASS').sum()
    failed = (df['filter_status'] == 'FAIL').sum()
    pass_rate = (passed / total * 100) if total > 0 else 0

    path_counts = df['pass_path'].value_counts().to_dict()

    print(f"  ✓ Saved to: {output_csv}")
    print(f"  Total: {total} | PASS: {passed} ({pass_rate:.1f}%) | FAIL: {failed}")
    print(f"  Path distribution: P1={path_counts.get(1.0, 0)} | P2={path_counts.get(2.0, 0)} | P3={path_counts.get(3.0, 0)}")

def main():
    """Main function to process all participant files"""
    print("="*60)
    print("COHERENCE-BASED SEMANTIC FILTER")
    print("="*60)

    # Initialize filter
    coherence_filter = CoherenceFilter()

    # Find all OCR result files
    ocr_files = glob.glob('ocr-results-new/Participant*_ocr_results.csv')

    if not ocr_files:
        print("No OCR files found matching pattern: ocr-results-new/Participant*_ocr_results.csv")
        return

    print(f"\nFound {len(ocr_files)} participant files")

    # Process each file
    for ocr_file in sorted(ocr_files):
        participant_id = ocr_file.replace('_ocr_results.csv', '')
        output_file = f"{participant_id}_clean.csv"

        process_participant_file(ocr_file, output_file, coherence_filter)

    print("\n" + "="*60)
    print("✓ All files processed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
