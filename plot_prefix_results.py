import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.font_manager import FontProperties

# Read the data
df = pd.read_csv('prefix_strategy_results.csv')

# Set the style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'DejaVu Sans', 'Apple Color Emoji', 'Segoe UI Emoji'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (12, 6)
})

# Create the plot
fig, ax = plt.subplots()

# Plot each strategy with different markers and colors
strategies = df['strategy'].unique()
colors = sns.color_palette("husl", len(strategies))
markers = ['o', 's', '^', 'D']

for strategy, color, marker in zip(strategies, colors, markers):
    strategy_data = df[df['strategy'] == strategy]
    ax.plot(strategy_data['prefix'], strategy_data['agreement_rate'],
            marker=marker, color=color, label=strategy, linewidth=2, markersize=8)

# Customize the plot
ax.set_xlabel('Prefix Type')
ax.set_ylabel('Agreement Rate')
ax.set_title('Impact of Different Prefixes on Judge Model Agreement Rates')
ax.grid(True, linestyle='--', alpha=0.7)

# Set y-axis limits
ax.set_ylim(0.3, 1.0)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Ensure proper emoji display in labels
for label in ax.get_xticklabels():
    label.set_fontproperties(FontProperties(family=['Arial Unicode MS', 'DejaVu Sans', 'Apple Color Emoji', 'Segoe UI Emoji']))

# Add a horizontal line for the baseline
baseline = df[df['prefix'] == 'baseline']['agreement_rate'].iloc[0]
ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# Adjust layout and legend
plt.tight_layout()
ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot
plt.savefig('prefix_impact_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a second plot focusing on the difference from baseline
fig, ax = plt.subplots()

# Calculate difference from baseline
baseline_rates = df[df['prefix'] == 'baseline'].set_index('strategy')['agreement_rate']
df['diff_from_baseline'] = df.apply(
    lambda row: row['agreement_rate'] - baseline_rates[row['strategy']], axis=1
)

# Plot the differences
for strategy, color, marker in zip(strategies, colors, markers):
    strategy_data = df[df['strategy'] == strategy]
    ax.plot(strategy_data['prefix'], strategy_data['diff_from_baseline'],
            marker=marker, color=color, label=strategy, linewidth=2, markersize=8)

# Customize the plot
ax.set_xlabel('Prefix Type')
ax.set_ylabel('Difference from Baseline Agreement Rate')
ax.set_title('Impact of Prefixes Relative to Baseline')
ax.grid(True, linestyle='--', alpha=0.7)

# Set y-axis limits for the difference plot
ax.set_ylim(-0.4, 0.1)  # Adjusted to show the negative differences clearly

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Ensure proper emoji display in labels
for label in ax.get_xticklabels():
    label.set_fontproperties(FontProperties(family=['Arial Unicode MS', 'DejaVu Sans', 'Apple Color Emoji', 'Segoe UI Emoji']))

# Add a horizontal line at y=0
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Adjust layout and legend
plt.tight_layout()
ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot
plt.savefig('prefix_impact_difference_plot.png', dpi=300, bbox_inches='tight')
plt.close() 