#!/usr/bin/env python3
"""Plot training metrics from TensorBoard logs."""

import os
import numpy as np
import matplotlib.pyplot as plt

# Option 1: Use tbparse (easiest)
try:
    from tbparse import SummaryReader
    USE_TBPARSE = True
except ImportError:
    USE_TBPARSE = False
    print("tbparse not found. Install with: pip install tbparse")
    print("Falling back to tensorboard.backend...")

# Option 2: Use tensorboard directly
if not USE_TBPARSE:
    from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data_tbparse(log_dir):
    """Load TensorBoard data using tbparse."""
    reader = SummaryReader(log_dir)
    df = reader.scalars
    return df


def load_tensorboard_data_native(log_dir):
    """Load TensorBoard data using native tensorboard."""
    # Find event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if 'events.out.tfevents' in f:
                event_files.append(os.path.join(root, f))
    
    if not event_files:
        raise FileNotFoundError(f"No event files found in {log_dir}")
    
    data = {}
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            if tag not in data:
                data[tag] = {'steps': [], 'values': []}
            data[tag]['steps'].extend(steps)
            data[tag]['values'].extend(values)
    
    return data


def plot_training_curves(log_dir, output_path=None):
    """Plot training curves from TensorBoard logs."""
    
    print(f"Loading TensorBoard data from: {log_dir}")
    
    if USE_TBPARSE:
        df = load_tensorboard_data_tbparse(log_dir)
        
        # Get unique tags
        tags = df['tag'].unique()
        print(f"Found metrics: {list(tags)}")
        
        # Key metrics to plot
        metrics_to_plot = [
            ('rollout/ep_rew_mean', 'Episode Reward (Mean)', 'blue'),
            ('rollout/ep_len_mean', 'Episode Length (Mean)', 'green'),
            ('train/value_loss', 'Value Loss', 'red'),
            ('train/policy_gradient_loss', 'Policy Loss', 'purple'),
            ('train/entropy_loss', 'Entropy Loss', 'orange'),
            ('train/approx_kl', 'Approx KL', 'brown'),
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle('DRL Patch Funnel - Training Progress', fontsize=14, fontweight='bold')
        
        for idx, (tag, title, color) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            tag_data = df[df['tag'] == tag]
            if len(tag_data) > 0:
                steps = tag_data['step'].values
                values = tag_data['value'].values
                
                # Sort by step
                sort_idx = np.argsort(steps)
                steps = steps[sort_idx]
                values = values[sort_idx]
                
                ax.plot(steps, values, color=color, linewidth=1.5, alpha=0.8)
                
                # Smooth line (rolling average)
                if len(values) > 10:
                    window = min(50, len(values) // 10)
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    ax.plot(steps[window-1:], smoothed, color=color, linewidth=2.5, label='Smoothed')
                
                # Find max/min
                if 'rew' in tag.lower():
                    max_idx = np.argmax(values)
                    max_val = values[max_idx]
                    max_step = steps[max_idx]
                    ax.axhline(y=max_val, color='darkgreen', linestyle='--', alpha=0.5)
                    ax.scatter([max_step], [max_val], color='green', s=100, zorder=5, marker='*')
                    ax.text(max_step, max_val, f'  Max: {max_val:.1f}', fontsize=9, color='darkgreen')
                
                ax.set_xlabel('Timesteps')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for\n