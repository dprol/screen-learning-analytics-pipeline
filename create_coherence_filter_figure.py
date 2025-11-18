#!/usr/bin/env python3
"""
Create visualization of the 3-Path Coherence Filter
Shows 4 examples with coherence scores instead of semantic scoring
Professional styling without bright colors
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_coherence_filter_figure(output_path='coherence_filter_explanation.png', include_title=True):
    """
    Create a figure with 4 examples of the coherence filter

    Parameters:
    - output_path: str, path to save the output image
    - include_title: bool, whether to include the main title (False for LaTeX figures)
    """
    # Create 2x2 grid layout - LARGE format
    fig = plt.figure(figsize=(16, 12))

    # Main title (optional for LaTeX)
    if include_title:
        fig.suptitle('Three-Path Coherence Filter Examples',
                     fontsize=18, fontweight='bold', y=0.97)

    # Create 4 subplots in 2x2 grid with very tight horizontal spacing
    positions = [
        [0.02, 0.51, 0.48, 0.45],   # Top left: PATH 1
        [0.505, 0.51, 0.48, 0.45],  # Top right: PATH 2
        [0.02, 0.03, 0.48, 0.45],   # Bottom left: PATH 3
        [0.505, 0.03, 0.48, 0.45],  # Bottom right: FAIL
    ]

    axes = []
    for pos in positions:
        ax = fig.add_axes(pos)
        ax.axis('off')
        axes.append(ax)

    # No longer needed - titles integrated in content boxes

    # ============================================================================
    # Example 1: PATH 1 - High Coherence
    # ============================================================================
    example1 = """PATH 1: Coh ≥ 0.50 AND Spell ≥ 50%

✓ PASS EXAMPLE

Text:
"What is currying in JavaScript.
 Currying transforms functions with
 multiple arguments into a sequence
 of functions."

Coherence Analysis:
  Sentence 1: "What is currying..."
  Sentence 2: "Currying transforms..."

  Embedding similarity: 0.78
  → High semantic relatedness

Spell Check: 100%
Coherence Score: 0.78

✓ Spell: 100% ≥ 50% → PASS ✓
✓ Coherence: 0.78 ≥ 0.50 → PASS ✓

→ PASS via PATH 1"""

    axes[0].text(0.05, 0.98, example1, va='top', fontsize=11, family='monospace',
                   color='black', transform=axes[0].transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='#666', linewidth=2))

    # ============================================================================
    # Example 2: PATH 2 - Moderate Coherence + Good Spelling
    # ============================================================================
    example2 = """PATH 2: Spell ≥ 65% AND Coh ≥ 0.30

✓ PASS EXAMPLE

Text:
"Function currying example code.
 This pattern is useful for partial
 application scenarios."

Spell Check: 67%
  Valid: Function, currying, example,
         code, This, pattern, is,
         useful, for, partial,
         application, scenarios
  = 12/18 = 67%

Coherence Score: 0.52
  Moderate semantic coherence

✓ Spell: 67% ≥ 65% → PASS ✓
✓ Coherence: 0.52 ≥ 0.30 → PASS ✓

→ PASS via PATH 2"""

    axes[1].text(0.05, 0.98, example2, va='top', fontsize=11, family='monospace',
                   color='black', transform=axes[1].transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='#666', linewidth=2))

    # ============================================================================
    # Example 3: PATH 3 - Excellent Spelling (Code/Technical)
    # ============================================================================
    example3 = """PATH 3: Spell ≥ 75%

✓ PASS EXAMPLE

Text:
"const add = (a) => (b) => a + b;
 This creates a curried function
 for addition operations."

Spell Check: 85%
  Valid: const, add, a, b, This,
         creates, a, function,
         for, addition, operations
  = 17/20 = 85%

Coherence Score: 0.22
  Low coherence (code + text mix)

✓ 85% ≥ 75% → PASS via PATH 3

Note: Excellent spelling alone
      suffices for technical content"""

    axes[2].text(0.05, 0.98, example3, va='top', fontsize=11, family='monospace',
                   color='black', transform=axes[2].transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='#666', linewidth=2))

    # ============================================================================
    # Example 4: FAIL - OCR Noise
    # ============================================================================
    example4 = """FAIL: All Paths

✗ FAIL EXAMPLE

Text:
"teer i aaa oh teme xzf quantum
 potato sideways"

Spell Check:
  Valid: i, oh, quantum, potato,
         sideways
  Invalid: teer, aaa, teme, xzf
  = 5/9 = 56%

Coherence Score: 0.12
  Random fragments, no semantic
  relationship between words

Check all paths:
  Path 1: 0.12 < 0.50 → FAIL ✗
  Path 2: 56% < 65% → FAIL ✗
  Path 3: 56% < 75% → FAIL ✗

→ OVERALL: REJECT ✗

Garbled OCR output:
  • Low spelling accuracy
  • No coherent meaning
  • Fragmented semantics"""

    axes[3].text(0.05, 0.98, example4, va='top', fontsize=11, family='monospace',
                   color='black', transform=axes[3].transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='#666', linewidth=2))

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Coherence filter figure saved to {output_path}")

    return output_path


if __name__ == '__main__':
    # Generate version without title for LaTeX (main version)
    create_coherence_filter_figure('coherence_filter_explanation.png', include_title=False)
    # Also generate version with title for reference
    create_coherence_filter_figure('coherence_filter_explanation_with_title.png', include_title=True)
    # Generate LaTeX-optimized version
    create_coherence_filter_figure('coherence_filter_explanation_latex.png', include_title=False)
