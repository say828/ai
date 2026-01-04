import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create comprehensive architecture diagram"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # Three subplots:
    # (a) Overall architecture
    # (b) Phase 1 (Task 0)
    # (c) Phase 2 (Task 1+)
    
    # (a) Overall Architecture
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('(a) Overall Architecture', fontsize=14, fontweight='bold')
    
    # Input layer
    input_box = FancyBboxPatch((0.5, 4), 1.5, 2, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax1.add_patch(input_box)
    ax1.text(1.25, 5, 'Input\n(784)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # W_in (shared encoder)
    win_box = FancyBboxPatch((3, 4), 2, 2,
                             boxstyle="round,pad=0.1",
                             edgecolor='red', facecolor='lightyellow', linewidth=3)
    ax1.add_patch(win_box)
    ax1.text(4, 5.5, r'$\mathbf{W_{in}}$', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(4, 4.5, '(shared)', ha='center', va='center', fontsize=9, style='italic')
    
    # Hidden state
    hidden_box = FancyBboxPatch((6, 4), 1.5, 2,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax1.add_patch(hidden_box)
    ax1.text(6.75, 5, 'Hidden\n(256)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output heads
    for i, y_pos in enumerate([7.5, 5, 2.5]):
        out_box = FancyBboxPatch((8.5, y_pos-0.4), 1, 0.8,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='blue', facecolor='lightcoral', linewidth=2)
        ax1.add_patch(out_box)
        ax1.text(9, y_pos, f'Task {i}' if i < 3 else '...', ha='center', va='center', fontsize=9)
    
    # Arrows
    ax1.arrow(2, 5, 0.8, 0, head_width=0.3, head_length=0.15, fc='black', ec='black')
    ax1.arrow(5, 5, 0.8, 0, head_width=0.3, head_length=0.15, fc='black', ec='black')
    ax1.arrow(7.5, 5, 0.8, 0, head_width=0.3, head_length=0.15, fc='black', ec='black')
    
    # Coherence feedback
    ax1.annotate('', xy=(6.75, 3.5), xytext=(6.75, 4),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax1.text(6.75, 3.2, 'Coherence\nCheck', ha='center', va='top', fontsize=9, color='green', fontweight='bold')
    
    # (b) Phase 1: Task 0
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('(b) Phase 1: First Task (Learn Both)', fontsize=14, fontweight='bold')
    
    # Similar structure but highlight learning
    input_box2 = FancyBboxPatch((1, 4), 1.5, 2, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='lightblue', linewidth=2)
    ax2.add_patch(input_box2)
    ax2.text(1.75, 5, 'Input', ha='center', va='center', fontsize=10)
    
    # W_in learning
    win_box2 = FancyBboxPatch((3.5, 4), 2, 2, boxstyle="round,pad=0.1",
                              edgecolor='red', facecolor='#ffcccc', linewidth=3)
    ax2.add_patch(win_box2)
    ax2.text(4.5, 5.5, r'$\mathbf{W_{in}}$', ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(4.5, 4.5, 'LEARNING', ha='center', va='center', fontsize=9, color='red', fontweight='bold')
    
    hidden_box2 = FancyBboxPatch((6.5, 4), 1.5, 2, boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax2.add_patch(hidden_box2)
    ax2.text(7.25, 5, 'Hidden', ha='center', va='center', fontsize=10)
    
    # W_out[0] learning
    out_box2 = FancyBboxPatch((8.5, 4), 1.2, 2, boxstyle="round,pad=0.1",
                              edgecolor='blue', facecolor='#ccccff', linewidth=3)
    ax2.add_patch(out_box2)
    ax2.text(9.1, 5.5, r'$\mathbf{W_{out}^{[0]}}$', ha='center', va='center', fontsize=11, fontweight='bold')
    ax2.text(9.1, 4.5, 'LEARNING', ha='center', va='center', fontsize=9, color='blue', fontweight='bold')
    
    # Bidirectional arrows for learning
    ax2.annotate('', xy=(3.4, 5), xytext=(2.5, 5),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax2.annotate('', xy=(8.4, 5), xytext=(8, 5),
                arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
    
    # (c) Phase 2: Task 1+
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('(c) Phase 2: Subsequent Tasks (Freeze W_in)', fontsize=14, fontweight='bold')
    
    input_box3 = FancyBboxPatch((1, 4), 1.5, 2, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='lightblue', linewidth=2)
    ax3.add_patch(input_box3)
    ax3.text(1.75, 5, 'Input', ha='center', va='center', fontsize=10)
    
    # W_in FROZEN
    win_box3 = FancyBboxPatch((3.5, 4), 2, 2, boxstyle="round,pad=0.1",
                              edgecolor='red', facecolor='#ffeeee', linewidth=3, linestyle='--')
    ax3.add_patch(win_box3)
    ax3.text(4.5, 5.5, r'$\mathbf{W_{in}}$', ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.text(4.5, 4.5, 'FROZEN', ha='center', va='center', fontsize=9, color='darkred', fontweight='bold')
    
    # Add lock icon effect (using text symbol)
    ax3.text(4.5, 6.3, '[LOCKED]', ha='center', va='center', fontsize=8, color='darkred')
    
    hidden_box3 = FancyBboxPatch((6.5, 4), 1.5, 2, boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax3.add_patch(hidden_box3)
    ax3.text(7.25, 5, 'Hidden', ha='center', va='center', fontsize=10)
    
    # W_out[t] learning
    out_box3 = FancyBboxPatch((8.5, 4), 1.2, 2, boxstyle="round,pad=0.1",
                              edgecolor='blue', facecolor='#ccccff', linewidth=3)
    ax3.add_patch(out_box3)
    ax3.text(9.1, 5.5, r'$\mathbf{W_{out}^{[t]}}$', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(9.1, 4.5, 'LEARNING', ha='center', va='center', fontsize=9, color='blue', fontweight='bold')
    
    # One-way arrows
    ax3.arrow(2.5, 5, 0.8, 0, head_width=0.3, head_length=0.15, fc='black', ec='black')
    ax3.arrow(5.5, 5, 0.8, 0, head_width=0.3, head_length=0.15, fc='black', ec='black')
    ax3.annotate('', xy=(8.4, 5), xytext=(8, 5),
                arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
    
    # (d) Coherence Computation
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('(d) Coherence Computation', fontsize=14, fontweight='bold')
    
    # 4 components
    components = [
        ('Predictability', 2, 7, 'Low entropy\nin transitions'),
        ('Stability', 6, 7, 'Low variance\nin states'),
        ('Complexity', 2, 3, 'Optimal variance\n(~0.5)'),
        ('Circularity', 6, 3, 'Autocorrelation\n(lag=1)')
    ]
    
    for name, x, y, desc in components:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                             boxstyle="round,pad=0.1",
                             edgecolor='purple', facecolor='lavender', linewidth=2)
        ax4.add_patch(box)
        ax4.text(x, y+0.2, name, ha='center', va='center', fontsize=10, fontweight='bold')
        ax4.text(x, y-0.3, desc, ha='center', va='center', fontsize=7, style='italic')
    
    # Combine to coherence
    ax4.text(4, 5, 'Coherence = Weighted Sum', ha='center', va='center', 
             fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontweight='bold')
    
    # Acceptance criterion
    ax4.text(4, 1.5, 'Accept update if:', ha='center', va='center', fontsize=10)
    ax4.text(4, 0.8, r'$\Phi_{new} \geq 0.95 \times \Phi_{old}$', ha='center', va='center', 
             fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_a_continual_learning/paper/figures/architecture_diagram.png', 
                dpi=300, bbox_inches='tight')
    print("Saved: architecture_diagram.png")
    
    # Also create simplified version for paper
    create_simplified_diagram()

def create_simplified_diagram():
    """Simpler version for paper"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Input
    input_box = FancyBboxPatch((0.5, 1.5), 1, 1,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1, 2, 'Input\nx', ha='center', va='center', fontsize=11)
    
    # W_in
    win_box = FancyBboxPatch((2.5, 1.2), 1.5, 1.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='red', facecolor='#ffe6e6', linewidth=3)
    ax.add_patch(win_box)
    ax.text(3.25, 2.3, r'$\mathbf{W_{in}}$', ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(3.25, 1.7, '(frozen after\ntask 0)', ha='center', va='center', fontsize=8, style='italic')
    
    # Hidden
    hidden_box = FancyBboxPatch((5, 1.5), 1, 1,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(hidden_box)
    ax.text(5.5, 2, 'h', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # W_out heads
    for i, (y, label) in enumerate([(3, '0'), (2, '1'), (1, 't')]):
        out_box = FancyBboxPatch((7.5, y-0.3), 1.5, 0.6,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='blue', facecolor='lightcoral', linewidth=2)
        ax.add_patch(out_box)
        ax.text(8.25, y, f'$W_{{out}}^{{[{label}]}}$', ha='center', va='center', fontsize=10)
    
    # Outputs
    for i, y in enumerate([3, 2, 1]):
        out = FancyBboxPatch((10, y-0.3), 0.8, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor='lightyellow', linewidth=2)
        ax.add_patch(out)
        ax.text(10.4, y, f'$y^{{[{i if i < 2 else "t"}]}}$', ha='center', va='center', fontsize=10)
    
    # Arrows
    ax.arrow(1.5, 2, 0.8, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(4, 2, 0.8, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 2, 1.3, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(9, 2, 0.8, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Coherence annotation
    ax.annotate('Coherence\ncheck', xy=(5.5, 1.2), xytext=(5.5, 0.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green'),
                fontsize=9, color='green', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_a_continual_learning/paper/figures/architecture_simple.png',
                dpi=300, bbox_inches='tight')
    print("Saved: architecture_simple.png")

if __name__ == '__main__':
    import os
    os.makedirs('/Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_a_continual_learning/paper/figures', exist_ok=True)
    create_architecture_diagram()
    print("All diagrams created successfully!")
