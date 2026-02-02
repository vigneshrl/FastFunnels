import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Hierarchical DRL Patch Funnel Architecture', 
            fontsize=16, ha='center', fontweight='bold')
    
    # Observation Space Box
    obs_box = FancyBboxPatch((0.5, 5), 4, 3.5, boxstyle="round,pad=0.05",
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(obs_box)
    ax.text(2.5, 8.2, 'Observation Space (69D)', fontsize=11, ha='center', fontweight='bold')
    
    obs_items = [
        'Patch State: 5D (x,y,θ,v,ω)',
        'Shape: 2D (a,b)',
        'LIDAR: 6D (aggregated)',
        'Heading: 2D (sin,cos)',
        'Goal: 4D (direction)',
        'Agent Pos: 4D (rel x,y)',
        'Agent Vel: 4D (v,ω)',
        'Inside Flags: 2D',
        'Domain: 8D',
        'Neurocontroller: 32D'
    ]
    for i, item in enumerate(obs_items):
        ax.text(0.7, 7.9 - i*0.32, item, fontsize=8, va='top')
    
    # PPO Policy Box
    ppo_box = FancyBboxPatch((5.5, 5), 3, 3.5, boxstyle="round,pad=0.05",
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(ppo_box)
    ax.text(7, 8.2, 'PPO Policy', fontsize=11, ha='center', fontweight='bold')
    
    ppo_items = [
        'Network: [256,256]',
        'Learning Rate: 3e-4',
        'Batch Size: 256',
        'N Epochs: 5',
        'Gamma: 0.98',
        'GAE Lambda: 0.95',
        'Clip Range: 0.2',
        'Entropy: 0.01',
        'SDE: Enabled'
    ]
    for i, item in enumerate(ppo_items):
        ax.text(5.7, 7.9 - i*0.35, item, fontsize=8, va='top')
    
    # Action Space Box
    act_box = FancyBboxPatch((9.5, 5), 4, 3.5, boxstyle="round,pad=0.05",
                              facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(act_box)
    ax.text(11.5, 8.2, 'Action Space (8D)', fontsize=11, ha='center', fontweight='bold')
    
    act_items = [
        'Patch Control (4D):',
        '  - velocity cmd',
        '  - steering cmd', 
        '  - semi-axis a',
        '  - semi-axis b',
        '',
        'Agent Control (4D):',
        '  - Agent 1: v, δ',
        '  - Agent 2: v, δ'
    ]
    for i, item in enumerate(act_items):
        ax.text(9.7, 7.9 - i*0.35, item, fontsize=8, va='top')
    
    # Arrows
    ax.annotate('', xy=(5.4, 6.75), xytext=(4.6, 6.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(9.4, 6.75), xytext=(8.6, 6.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Reward Box
    rew_box = FancyBboxPatch((3, 0.5), 8, 3.5, boxstyle="round,pad=0.05",
                              facecolor='lightsalmon', edgecolor='black', linewidth=2)
    ax.add_patch(rew_box)
    ax.text(7, 3.7, 'Reward Function', fontsize=11, ha='center', fontweight='bold')
    
    rew_items = [
        ('Progress', '+200 × Δprogress × 100', 'Primary objective'),
        ('Containment', '+2.0 × (agents inside / total)', 'Keep agents in patch'),
        ('Movement', '+5.0 if v>1.0, else -1.0', 'Encourage forward motion'),
        ('Outside Penalty', '-0.2 × outside_distance', 'Discourage agents leaving'),
        ('Inter-Agent', '-0.1 per collision', 'Avoid agent-agent collision'),
    ]
    
    ax.text(3.3, 3.3, 'Component', fontsize=9, fontweight='bold')
    ax.text(6, 3.3, 'Formula', fontsize=9, fontweight='bold')
    ax.text(9, 3.3, 'Purpose', fontsize=9, fontweight='bold')
    
    for i, (comp, formula, purpose) in enumerate(rew_items):
        y = 3.0 - i*0.45
        ax.text(3.3, y, comp, fontsize=8)
        ax.text(6, y, formula, fontsize=8)
        ax.text(9, y, purpose, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('model_architecture.pdf', bbox_inches='tight')
    plt.show()
    print("Saved: model_architecture.png and model_architecture.pdf")

if __name__ == "__main__":
    create_architecture_diagram()