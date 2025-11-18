#!/usr/bin/env python3
"""
Conceptual Transitions Visualizer
Creates visualizations showing how participants transition between semantic screens
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TransitionVisualizer:
    def __init__(self, similarity_dir='similarity-results', output_dir='transition-visualizations'):
        """Initialize the visualizer"""
        self.similarity_dir = Path(similarity_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 10

    def load_similarities(self, participant):
        """Load similarity data for a participant"""
        filepath = self.similarity_dir / f"{participant}_similarities.csv"
        if not filepath.exists():
            print(f"  ! File not found: {filepath}")
            return None

        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df):,} comparisons for {participant}")
        return df

    def load_transitions(self, participant, threshold=0.7):
        """Load significant transitions for a participant"""
        filepath = self.similarity_dir / f"{participant}_transitions_threshold_{threshold}.csv"
        if not filepath.exists():
            print(f"  ! File not found: {filepath}")
            return None

        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df):,} significant transitions for {participant}")
        return df

    def plot_similarity_heatmap(self, participant, max_frames=500):
        """
        Create a heatmap showing similarity between consecutive frames
        """
        print(f"\nCreating similarity heatmap for {participant}...")

        df = self.load_similarities(participant)
        if df is None:
            return

        # Filter to consecutive frames only (gap = 1)
        consecutive = df[df['frame_gap'] == 1].copy()

        # Limit to first N frames for visibility
        consecutive = consecutive[consecutive['frame1'] < max_frames]

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot as line chart
        ax.plot(consecutive['frame1'], consecutive['similarity'],
                linewidth=1.5, alpha=0.7, color='steelblue')
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.7)')

        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'{participant}: Frame-to-Frame Similarity (Consecutive Frames)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save
        output_file = self.output_dir / f"{participant}_consecutive_similarity.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def plot_transition_distribution(self, participant):
        """
        Plot distribution of similarity scores showing transition types
        """
        print(f"\nCreating transition distribution for {participant}...")

        df = self.load_similarities(participant)
        if df is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Histogram of similarities
        axes[0].hist(df['similarity'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Threshold (0.7)')
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{participant}: Similarity Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Similarity by frame gap
        gap_stats = df.groupby('frame_gap')['similarity'].agg(['mean', 'std', 'count'])
        gap_stats = gap_stats[gap_stats['count'] > 10]  # Filter low-count gaps

        axes[1].errorbar(gap_stats.index, gap_stats['mean'],
                        yerr=gap_stats['std'],
                        marker='o', linestyle='-', capsize=3, alpha=0.7)
        axes[1].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.7)')
        axes[1].set_xlabel('Frame Gap')
        axes[1].set_ylabel('Mean Similarity')
        axes[1].set_title(f'{participant}: Similarity vs Frame Gap')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Save
        output_file = self.output_dir / f"{participant}_distribution.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def plot_transition_timeline(self, participant, threshold=0.7):
        """
        Plot timeline showing when significant transitions occur
        """
        print(f"\nCreating transition timeline for {participant}...")

        transitions = self.load_transitions(participant, threshold)
        if transitions is None:
            return

        # Filter to consecutive or near-consecutive transitions (gap <= 5)
        near_consecutive = transitions[transitions['frame_gap'] <= 5].copy()

        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot scatter of transitions
        scatter = ax.scatter(near_consecutive['timestamp1'],
                           near_consecutive['similarity'],
                           c=near_consecutive['frame_gap'],
                           cmap='viridis',
                           alpha=0.6,
                           s=30)

        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.3, label='Threshold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Similarity')
        ax.set_title(f'{participant}: Significant Transitions Timeline (Frame Gap ≤ 5)')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frame Gap')

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save
        output_file = self.output_dir / f"{participant}_timeline.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def plot_transition_flow(self, participant, window_size=100):
        """
        Plot flow of semantic changes showing intensity of transitions
        """
        print(f"\nCreating transition flow for {participant}...")

        df = self.load_similarities(participant)
        if df is None:
            return

        # Get consecutive transitions only
        consecutive = df[df['frame_gap'] == 1].copy()
        consecutive = consecutive.sort_values('frame1')

        # Calculate rolling statistics
        consecutive['similarity_rolling_mean'] = consecutive['similarity'].rolling(window=window_size, center=True).mean()
        consecutive['similarity_rolling_std'] = consecutive['similarity'].rolling(window=window_size, center=True).std()
        consecutive['distance'] = 1 - consecutive['similarity']
        consecutive['distance_rolling_mean'] = consecutive['distance'].rolling(window=window_size, center=True).mean()

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # Top plot: Similarity with rolling mean
        axes[0].plot(consecutive['frame1'], consecutive['similarity'],
                    alpha=0.3, color='lightblue', linewidth=0.5, label='Raw Similarity')
        axes[0].plot(consecutive['frame1'], consecutive['similarity_rolling_mean'],
                    color='steelblue', linewidth=2, label=f'Rolling Mean (window={window_size})')
        axes[0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.7)')
        axes[0].set_ylabel('Similarity')
        axes[0].set_title(f'{participant}: Semantic Flow Analysis')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Bottom plot: Distance (1 - similarity) to show transition intensity
        axes[1].fill_between(consecutive['frame1'], 0, consecutive['distance'],
                            alpha=0.3, color='coral', label='Semantic Distance')
        axes[1].plot(consecutive['frame1'], consecutive['distance_rolling_mean'],
                    color='darkred', linewidth=2, label=f'Rolling Mean Distance')
        axes[1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='High Change (distance > 0.3)')
        axes[1].set_xlabel('Frame Number')
        axes[1].set_ylabel('Semantic Distance (1 - Similarity)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # Save
        output_file = self.output_dir / f"{participant}_flow.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def compare_participants(self, participants):
        """
        Compare transition patterns between participants
        """
        print(f"\nComparing {len(participants)} participants...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        colors = ['steelblue', 'coral', 'green', 'purple', 'orange']

        for idx, participant in enumerate(participants):
            df = self.load_similarities(participant)
            if df is None:
                continue

            color = colors[idx % len(colors)]

            # 1. Similarity distribution
            axes[0, 0].hist(df['similarity'], bins=50, alpha=0.5,
                          label=participant, color=color, edgecolor='none')

            # 2. Consecutive frame similarity
            consecutive = df[df['frame_gap'] == 1].sort_values('frame1')
            if len(consecutive) > 0:
                axes[0, 1].plot(consecutive['frame1'][:500],
                              consecutive['similarity'][:500],
                              alpha=0.7, linewidth=1.5, label=participant, color=color)

            # 3. Mean similarity by gap
            gap_stats = df.groupby('frame_gap')['similarity'].mean()
            gap_stats = gap_stats[gap_stats.index <= 20]
            axes[1, 0].plot(gap_stats.index, gap_stats.values,
                          marker='o', alpha=0.7, linewidth=2, label=participant, color=color)

            # 4. Transition rate (% below threshold)
            transition_rate = (df['similarity'] < 0.7).mean() * 100
            axes[1, 1].bar(idx, transition_rate, alpha=0.7, color=color, label=participant)

        # Formatting
        axes[0, 0].set_xlabel('Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Similarity Distribution Comparison')
        axes[0, 0].axvline(x=0.7, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel('Frame Number')
        axes[0, 1].set_ylabel('Similarity')
        axes[0, 1].set_title('Consecutive Frame Similarity (First 500 frames)')
        axes[0, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_xlabel('Frame Gap')
        axes[1, 0].set_ylabel('Mean Similarity')
        axes[1, 0].set_title('Similarity Decay by Frame Gap')
        axes[1, 0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_xlabel('Participant')
        axes[1, 1].set_ylabel('Transition Rate (%)')
        axes[1, 1].set_title('Percentage of Significant Transitions (similarity < 0.7)')
        axes[1, 1].set_xticks(range(len(participants)))
        axes[1, 1].set_xticklabels(participants, rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Save
        output_file = self.output_dir / "participants_comparison.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file}")

    def generate_all_visualizations(self, participants):
        """Generate all visualizations for given participants"""
        print("="*60)
        print("TRANSITION VISUALIZATION GENERATOR")
        print("="*60)

        for participant in participants:
            print(f"\n{'='*60}")
            print(f"Processing: {participant}")
            print(f"{'='*60}")

            self.plot_similarity_heatmap(participant, max_frames=500)
            self.plot_transition_distribution(participant)
            self.plot_transition_timeline(participant)
            self.plot_transition_flow(participant, window_size=100)

        # Comparison plot
        if len(participants) > 1:
            self.compare_participants(participants)

        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ All visualizations saved to: {self.output_dir}")


def main():
    """Main execution"""

    # Initialize visualizer
    visualizer = TransitionVisualizer(
        similarity_dir='similarity-results',
        output_dir='transition-visualizations'
    )

    # Generate visualizations for P12 and P25
    participants = ['Participant12', 'Participant25']

    visualizer.generate_all_visualizations(participants)


if __name__ == "__main__":
    main()
