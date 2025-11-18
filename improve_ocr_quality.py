#!/usr/bin/env python3
"""
OCR Quality Improvement Pipeline
Cleans and improves existing OCR results using NLP techniques
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class OCRQualityImprover:
    def __init__(self, ocr_dir='ocr-results-new', output_dir='ocr-results-improved'):
        """Initialize OCR quality improver"""
        self.ocr_dir = Path(ocr_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Common programming/technical terms
        self.tech_vocabulary = {
            'javascript', 'function', 'expression', 'invoked', 'immediately',
            'programming', 'variable', 'scope', 'lexical', 'algorithm',
            'search', 'query', 'results', 'wikipedia', 'google', 'github',
            'stackoverflow', 'documentation', 'tutorial', 'example',
            'python', 'java', 'react', 'node', 'database', 'api'
        }

        # Common OCR errors
        self.common_corrections = {
            r'\bjs\b': 'is',
            r'\bhet\b': 'the',
            r'\barr\b': 'an',
            r'\bwhieh\b': 'which',
            r'\bfrom\b': 'from',
            r'\bSecorids\b': 'Seconds',
            r'\bFunston\b': 'Function',
            r'\bFunclionalt\b': 'Functional',
            r'\bFuncionaly\b': 'Functionally',
            r'\bSprograraming\b': 'programming',
            r'\blejmediately\b': 'immediately',
            r'\bInvaked\b': 'Invoked',
            r'\bmmediaiely\b': 'immediately',
            r'\bexpresioni\b': 'expression',
            r'\bJove\b': 'JavaScript',
            r'\bgeiacrsinmesitey\b': 'geeksforgeeks',
            r'\binvclen\b': 'invoked',
            r'\bwaked\b': 'invoked',
            r'\bFinston\b': 'Function',
        }

    def clean_text(self, text):
        """Clean OCR text with multiple strategies"""
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Remove excessive whitespace and special characters
        text = re.sub(r'\s+', ' ', text)

        # 2. Remove lines with too many special characters (likely noise)
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if len(line.strip()) < 3:
                continue
            # Calculate ratio of letters to total chars
            letters = sum(c.isalpha() for c in line)
            if len(line) > 0 and letters / len(line) > 0.3:
                clean_lines.append(line)

        text = ' '.join(clean_lines)

        # 3. Apply common corrections
        for pattern, replacement in self.common_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 4. Fix common URL patterns
        text = re.sub(r'googte\.?com', 'google.com', text, flags=re.IGNORECASE)
        text = re.sub(r'wikipedia\.?og', 'wikipedia.org', text, flags=re.IGNORECASE)
        text = re.sub(r'github\.?om', 'github.com', text, flags=re.IGNORECASE)

        # 5. Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', ' ', text)
        text = re.sub(r'[_]{3,}', ' ', text)

        # 6. Clean up spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)

        return text.strip()

    def assess_quality(self, text):
        """Assess text quality (0-100 score)"""
        if not text or len(text) < 20:
            return 0

        score = 0

        # 1. Length check (20 points)
        if 20 <= len(text) <= 1000:
            score += 20
        elif len(text) > 1000:
            score += 10

        # 2. Alphabetic ratio (30 points)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        score += int(alpha_ratio * 30)

        # 3. Word spacing (20 points)
        space_ratio = text.count(' ') / len(text)
        if 0.10 <= space_ratio <= 0.30:
            score += 20
        elif 0.05 <= space_ratio < 0.10 or 0.30 < space_ratio <= 0.40:
            score += 10

        # 4. Common words (30 points)
        words = text.lower().split()
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in',
                       'on', 'at', 'to', 'for', 'of', 'with', 'and', 'or',
                       'what', 'how', 'when', 'where', 'why', 'which'}
        common_count = sum(1 for w in words if w in common_words or w in self.tech_vocabulary)
        if len(words) > 0:
            score += int((common_count / len(words)) * 30)

        return min(score, 100)

    def extract_readable_content(self, text, min_quality=40):
        """Extract most readable parts of text"""
        if not text:
            return "", 0

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        # Score each sentence
        scored_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            quality = self.assess_quality(sent)
            if quality >= min_quality:
                scored_sentences.append((sent, quality))

        if not scored_sentences:
            # Return cleaned text if no good sentences
            cleaned = self.clean_text(text)
            return cleaned[:500], self.assess_quality(cleaned)

        # Return best sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        best_sentences = [s[0] for s in scored_sentences[:3]]
        result = '. '.join(best_sentences)

        return result[:500], scored_sentences[0][1]

    def improve_participant_ocr(self, participant):
        """Improve OCR for a single participant"""
        input_file = self.ocr_dir / f"{participant}_ocr_results.csv"
        output_file = self.output_dir / f"{participant}_ocr_results_improved.csv"

        if not input_file.exists():
            print(f"  ⚠ File not found: {input_file}")
            return None

        print(f"\nProcessing {participant}...")

        try:
            # Load data
            df = pd.read_csv(input_file, encoding='latin-1', on_bad_lines='skip', engine='python')

            if 'raw_text' not in df.columns:
                print(f"  ✗ No 'raw_text' column found")
                return None

            # Process each row
            improved_texts = []
            quality_scores = []
            original_lengths = []
            improved_lengths = []

            for idx, row in df.iterrows():
                original_text = str(row['raw_text'])

                # Clean and improve
                cleaned = self.clean_text(original_text)
                readable, quality = self.extract_readable_content(cleaned, min_quality=30)

                improved_texts.append(readable)
                quality_scores.append(quality)
                original_lengths.append(len(original_text))
                improved_lengths.append(len(readable))

            # Add improved columns
            df['improved_text'] = improved_texts
            df['quality_score'] = quality_scores
            df['original_length'] = original_lengths
            df['improved_length'] = improved_lengths

            # Save
            df.to_csv(output_file, index=False, encoding='utf-8')

            # Statistics
            avg_quality_before = df.apply(lambda row: self.assess_quality(str(row['raw_text'])), axis=1).mean()
            avg_quality_after = df['quality_score'].mean()
            high_quality_count = (df['quality_score'] >= 50).sum()

            print(f"  ✓ Processed {len(df)} frames")
            print(f"  → Average quality before: {avg_quality_before:.1f}/100")
            print(f"  → Average quality after: {avg_quality_after:.1f}/100")
            print(f"  → High quality frames (≥50): {high_quality_count} ({high_quality_count/len(df)*100:.1f}%)")
            print(f"  → Saved to: {output_file}")

            return {
                'participant': participant,
                'total_frames': len(df),
                'avg_quality_before': avg_quality_before,
                'avg_quality_after': avg_quality_after,
                'high_quality_count': high_quality_count,
                'improvement': avg_quality_after - avg_quality_before
            }

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def improve_all_participants(self):
        """Improve OCR for all participants"""
        print("="*60)
        print("OCR QUALITY IMPROVEMENT PIPELINE")
        print("="*60)

        # Find all participant files
        participant_files = sorted(self.ocr_dir.glob("Participant*_ocr_results.csv"))
        participants = [f.stem.replace('_ocr_results', '') for f in participant_files]

        print(f"\nFound {len(participants)} participants")

        results = []
        for participant in participants:
            result = self.improve_participant_ocr(participant)
            if result:
                results.append(result)

        # Summary report
        if results:
            print("\n" + "="*60)
            print("SUMMARY REPORT")
            print("="*60)

            summary_df = pd.DataFrame(results)
            summary_file = self.output_dir / "improvement_summary.csv"
            summary_df.to_csv(summary_file, index=False)

            print(f"\nTotal participants processed: {len(results)}")
            print(f"Average quality improvement: {summary_df['improvement'].mean():.1f} points")
            print(f"Best improvement: {summary_df['improvement'].max():.1f} points ({summary_df.loc[summary_df['improvement'].idxmax(), 'participant']})")
            print(f"\n✓ Summary saved: {summary_file}")

            # Create markdown report
            self.create_improvement_report(summary_df)

    def create_improvement_report(self, summary_df):
        """Create markdown report of improvements"""
        report = []
        report.append("# OCR Quality Improvement Report")
        report.append("")
        report.append("## Summary Statistics")
        report.append("")
        report.append("| Participant | Frames | Quality Before | Quality After | Improvement | High Quality % |")
        report.append("|-------------|--------|----------------|---------------|-------------|----------------|")

        for _, row in summary_df.iterrows():
            high_qual_pct = (row['high_quality_count'] / row['total_frames']) * 100
            report.append(f"| {row['participant']} | {row['total_frames']} | "
                         f"{row['avg_quality_before']:.1f} | {row['avg_quality_after']:.1f} | "
                         f"+{row['improvement']:.1f} | {high_qual_pct:.1f}% |")

        report.append("")
        report.append("## Overall Statistics")
        report.append("")
        report.append(f"- **Total Participants**: {len(summary_df)}")
        report.append(f"- **Average Quality Before**: {summary_df['avg_quality_before'].mean():.1f}/100")
        report.append(f"- **Average Quality After**: {summary_df['avg_quality_after'].mean():.1f}/100")
        report.append(f"- **Average Improvement**: +{summary_df['improvement'].mean():.1f} points")
        report.append(f"- **Total Frames Processed**: {summary_df['total_frames'].sum():,}")
        report.append("")
        report.append("## Improvement Techniques Applied")
        report.append("")
        report.append("1. **Text Cleaning**: Removed excessive whitespace and special characters")
        report.append("2. **Noise Filtering**: Removed lines with <30% alphabetic characters")
        report.append("3. **Spelling Correction**: Fixed common OCR errors (e.g., 'Funston' → 'Function')")
        report.append("4. **URL Normalization**: Corrected common URL misspellings")
        report.append("5. **Sentence Extraction**: Selected highest quality sentences per frame")
        report.append("6. **Quality Scoring**: Each frame scored 0-100 for readability")
        report.append("")

        report_file = self.output_dir / "improvement_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))

        print(f"✓ Improvement report saved: {report_file}")


def main():
    """Main execution"""
    improver = OCRQualityImprover(
        ocr_dir='ocr-results-new',
        output_dir='ocr-results-improved'
    )

    improver.improve_all_participants()

    print("\n" + "="*60)
    print("OCR IMPROVEMENT COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review improved files in ocr-results-improved/")
    print("  2. Re-run similarity analysis with improved OCR")
    print("  3. Compare quality scores before/after")


if __name__ == "__main__":
    main()
