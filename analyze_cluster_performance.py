import pandas as pd
import numpy as np

# Data from updated Table 1
data = {
    'Participant': [9, 12, 14, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35],
    'SBERT_Changes': [1317, 1493, 1761, 1685, 1969, 1614, 2508, 1934, 1366, 1389, 1337, 1033, 1177, 1292, 1134, 1273],
    'Duration_mins': [64.53, 61.83, 71.65, 65.37, 80.55, 75.08, 71.55, 68.40, 60.82, 75.72, 66.23, 73.87, 72.90, 69.22, 72.72, 70.92],
    'Curry_Debug': ['I', 'C', 'I', 'I', 'I', 'I', 'C', 'C', 'I', 'C', 'CL', 'I', 'I', 'I', 'I', 'I'],
    'IIFE_Debug': ['I', 'C', 'I', 'CL', 'I', 'I', 'C', 'I', 'I', 'I', 'I', 'C', 'I', 'C', 'I', 'C']
}

df = pd.DataFrame(data)

# Calculate changes per minute
df['Changes_Per_Min'] = df['SBERT_Changes'] / df['Duration_mins']

# Assign clusters based on changes/min (from paper)
# High-Activity: ~27.57 changes/min
# Moderate: ~22.20 changes/min
# Focused: ~17.27 changes/min
def assign_cluster(cpm):
    if cpm >= 24:
        return 'High-Activity'
    elif cpm >= 19:
        return 'Moderate'
    else:
        return 'Focused'

df['Cluster'] = df['Changes_Per_Min'].apply(assign_cluster)

# Create numeric scores for debugging performance
def debug_score(val):
    if val == 'C':
        return 2
    elif val == 'CL':
        return 1
    else:  # 'I'
        return 0

df['Curry_Score'] = df['Curry_Debug'].apply(debug_score)
df['IIFE_Score'] = df['IIFE_Debug'].apply(debug_score)
df['Total_Score'] = df['Curry_Score'] + df['IIFE_Score']

# Sort by changes per minute
df = df.sort_values('Changes_Per_Min', ascending=False)

print("="*90)
print("CORRELATION ANALYSIS: Behavioral Clusters vs Debugging Performance")
print("="*90)
print()

print(df[['Participant', 'Changes_Per_Min', 'Cluster', 'Curry_Debug', 'IIFE_Debug', 'Total_Score']].to_string(index=False))
print()

# Group statistics by cluster
print("\n" + "="*90)
print("CLUSTER STATISTICS")
print("="*90)

for cluster in ['High-Activity', 'Moderate', 'Focused']:
    cluster_df = df[df['Cluster'] == cluster]
    print(f"\n{cluster} (n={len(cluster_df)}):")
    print(f"  Changes/min range: {cluster_df['Changes_Per_Min'].min():.2f} - {cluster_df['Changes_Per_Min'].max():.2f}")
    print(f"  Average total score: {cluster_df['Total_Score'].mean():.2f} / 4.00")
    print(f"  Curry correct: {(cluster_df['Curry_Debug'] == 'C').sum()} / {len(cluster_df)}")
    print(f"  IIFE correct: {(cluster_df['IIFE_Debug'] == 'C').sum()} / {len(cluster_df)}")
    print(f"  Both correct: {((cluster_df['Curry_Debug'] == 'C') & (cluster_df['IIFE_Debug'] == 'C')).sum()} / {len(cluster_df)}")
    print(f"  Both incorrect: {((cluster_df['Curry_Debug'] == 'I') & (cluster_df['IIFE_Debug'] == 'I')).sum()} / {len(cluster_df)}")
    print(f"  Participants: {cluster_df['Participant'].tolist()}")

# Correlation analysis
print("\n" + "="*90)
print("CORRELATION COEFFICIENTS")
print("="*90)
print(f"Changes/min vs Total Score: r = {df['Changes_Per_Min'].corr(df['Total_Score']):.3f}")
print(f"Changes/min vs Curry Score: r = {df['Changes_Per_Min'].corr(df['Curry_Score']):.3f}")
print(f"Changes/min vs IIFE Score: r = {df['Changes_Per_Min'].corr(df['IIFE_Score']):.3f}")

# Key findings
print("\n" + "="*90)
print("KEY FINDINGS")
print("="*90)

# Find participants who did well
high_performers = df[df['Total_Score'] >= 3]
print(f"\nHigh Performers (score ≥ 3): {len(high_performers)} participants")
for _, row in high_performers.iterrows():
    print(f"  Participant{row['Participant']}: {row['Cluster']:15s} - {row['Changes_Per_Min']:.2f} changes/min - Score: {row['Total_Score']}")

# Find participants who did poorly
low_performers = df[df['Total_Score'] == 0]
print(f"\nLow Performers (score = 0): {len(low_performers)} participants")
for _, row in low_performers.iterrows():
    print(f"  Participant{row['Participant']}: {row['Cluster']:15s} - {row['Changes_Per_Min']:.2f} changes/min")

# Success rate by cluster
print("\n" + "="*90)
print("SUCCESS RATES BY CLUSTER")
print("="*90)
for cluster in ['High-Activity', 'Moderate', 'Focused']:
    cluster_df = df[df['Cluster'] == cluster]
    success_rate = (cluster_df['Total_Score'] >= 2).sum() / len(cluster_df) * 100
    print(f"{cluster:15s}: {success_rate:.1f}% scored ≥ 2/4")
