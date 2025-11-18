#!/usr/bin/env python3
"""
Create Behavior Pattern Visualizations with CORRECT data
Uses updated coherence filter statistics: 6,674 PASS frames, 1,533 predictions
"""

import matplotlib.pyplot as plt
import numpy as np

# CORRECT DATA from behavior_summary.csv (after PATH 1 fix)
behaviors = {
    'Quiz': 574,
    'SEg': 435,
    'STh': 243,
    'AIEg': 41,
    'Website': 36,
    'AILoc': 135,
    'SLoc': 135,
    'Practical': 33,
    'IDE': 16,
    'AITh': 1
}

total_predictions = sum(behaviors.values())  # 1,649 (some frames have multiple codes)
total_pass_frames = 6674  # From coherence filter
unique_predictions = 1533  # Frames with at least one prediction

# Calculate percentages (of total predictions for top behaviors)
top_5_behaviors = {
    'Quiz': 574,
    'SEg': 435,
    'STh': 243,
    'AIEg': 41,
    'Website': 36
}

# For display: percentage of unique predictions
percentages = {k: (v / unique_predictions * 100) for k, v in top_5_behaviors.items()}

def create_simple_bar_chart():
    """Create simple bar chart showing top 5 behavior codes"""
    fig, ax = plt.subplots(figsize=(14, 10))

    labels = ['Quiz', 'SEg\n(Search Examples)', 'STh\n(Search Theory)',
              'AIEg\n(AI Examples)', 'Website']
    values = [percentages['Quiz'], percentages['SEg'], percentages['STh'],
              percentages['AIEg'], percentages['Website']]
    counts = [top_5_behaviors['Quiz'], top_5_behaviors['SEg'],
              top_5_behaviors['STh'], top_5_behaviors['AIEg'],
              top_5_behaviors['Website']]

    colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']

    x = np.arange(len(labels))
    width = 0.7

    bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add percentage labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=16, fontweight='bold')

    ax.set_ylabel('Percentage of Predictions (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Behavior Codes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('behavior_patterns_simple.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: behavior_patterns_simple.png")
    plt.close()

def create_dual_visualization():
    """Create side-by-side: bar chart + pie chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # LEFT: Bar chart
    labels = ['Quiz', 'SEg\n(Search Examples)', 'STh\n(Search Theory)',
              'AIEg\n(AI Examples)', 'Website']
    values = [percentages['Quiz'], percentages['SEg'], percentages['STh'],
              percentages['AIEg'], percentages['Website']]

    colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']

    x = np.arange(len(labels))
    width = 0.7

    bars = ax1.bar(x, values, width, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add percentage labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=13, fontweight='bold')

    ax1.set_title('Dominant Behavior Patterns\n(from 1,533 predictions across 6,674 PASS frames)',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage of Predictions (%)', fontsize=12)
    ax1.set_xlabel('Behavior Code', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, 45)
    ax1.grid(axis='y', alpha=0.3)

    # RIGHT: Pie chart showing AI vs Traditional
    # Traditional resources: Quiz + SEg + STh + Website = 574 + 435 + 243 + 36 = 1,288
    # AI resources: AIEg + AILoc + AITh = 41 + 135 + 1 = 177
    traditional = behaviors['Quiz'] + behaviors['SEg'] + behaviors['STh'] + behaviors['Website']
    ai_assisted = behaviors['AIEg'] + behaviors['AILoc'] + behaviors['AITh']
    other = behaviors['Practical'] + behaviors['SLoc'] + behaviors['IDE']

    pie_data = [traditional, ai_assisted, other]
    pie_labels = ['Traditional Resources\n(Quiz, SEg, STh, Website)',
                  'AI-Assisted\n(AI Examples)', 'Other']
    pie_colors = ['#27ae60', '#e74c3c', '#bdc3c7']

    wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, colors=pie_colors,
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'},
                                         wedgeprops={'edgecolor': 'black', 'linewidth': 2})

    # Make percentage text white for visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)

    ax2.set_title('Traditional vs AI-Assisted Learning Patterns',
                  fontsize=12, fontweight='bold')

    # Add note at bottom
    fig.text(0.5, 0.02,
             'Modern learners blend traditional resources (69.0%) with AI assistance (10.2%)',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('behavior_patterns_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: behavior_patterns_visualization.png")
    plt.close()

def print_statistics():
    """Print updated statistics"""
    print("\n" + "="*80)
    print("BEHAVIOR PREDICTION STATISTICS (UPDATED)")
    print("="*80)
    print(f"\nTotal PASS frames (coherence filter): {total_pass_frames:,}")
    print(f"Frames with predictions: {unique_predictions:,} ({unique_predictions/total_pass_frames*100:.1f}%)")
    print(f"Total prediction instances: {total_predictions:,}")
    print("\nTop 5 Behaviors:")
    print("-"*80)
    for behavior, count in top_5_behaviors.items():
        pct = count / unique_predictions * 100
        print(f"  {behavior:12s}: {count:4d} ({pct:5.1f}% of predictions)")
    print("\nCategories:")
    print("-"*80)
    traditional = behaviors['Quiz'] + behaviors['SEg'] + behaviors['STh'] + behaviors['Website']
    ai_assisted = behaviors['AIEg'] + behaviors['AILoc'] + behaviors['AITh']
    print(f"  Traditional: {traditional:4d} ({traditional/total_predictions*100:.1f}%)")
    print(f"  AI-Assisted: {ai_assisted:4d} ({ai_assisted/total_predictions*100:.1f}%)")
    print("="*80)

if __name__ == '__main__':
    print("Creating behavior pattern visualizations with CORRECT data...")
    print("Based on 3-path coherence filter:")
    print("  - PATH 1: coherence ≥ 0.50 AND spell ≥ 50%")
    print("  - PATH 2: spell ≥ 65% AND coherence ≥ 0.30")
    print("  - PATH 3: spell ≥ 75%")

    create_simple_bar_chart()
    create_dual_visualization()
    print_statistics()

    print("\n✓ All visualizations updated successfully!")
