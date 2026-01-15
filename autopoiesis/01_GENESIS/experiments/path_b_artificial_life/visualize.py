"""
GENESIS Path B: Visualization Tools
====================================

Visualization for artificial life experiments:
- Real-time grid world animation
- Population dynamics plots
- Behavioral repertoire visualization
- Niche specialization heatmaps

Author: GENESIS Project
Date: 2026-01-04
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional, Callable
import os


# Color schemes
AGENT_COLORS = {
    'autopoietic': '#2196F3',  # Blue
    'rl': '#FF9800',           # Orange  
    'neat': '#9C27B0',         # Purple
    'random': '#607D8B'        # Gray
}

CELL_COLORS = {
    'empty': '#F5F5F5',
    'resource': '#4CAF50',
    'obstacle': '#424242',
    'predator': '#F44336',
    'agent': '#2196F3'
}


class GridWorldVisualizer:
    """
    Real-time visualization of grid world
    """
    
    def __init__(
        self,
        grid_size: int = 100,
        figsize: Tuple[int, int] = (10, 10),
        show_trails: bool = True,
        trail_length: int = 20
    ):
        self.grid_size = grid_size
        self.figsize = figsize
        self.show_trails = show_trails
        self.trail_length = trail_length
        
        self.fig = None
        self.ax = None
        self.im = None
        self.agent_scatter = None
        self.trail_lines = {}
    
    def init_plot(self):
        """Initialize matplotlib figure"""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.set_title('Grid World', fontsize=14)
        
        # Initialize empty image
        empty_grid = np.ones((self.grid_size, self.grid_size, 3))
        self.im = self.ax.imshow(empty_grid, origin='lower', extent=[0, self.grid_size, 0, self.grid_size])
        
        return self.fig, self.ax
    
    def render_frame(
        self,
        grid: np.ndarray,
        agent_positions: Dict[str, Tuple[int, int]],
        agent_types: Dict[str, str],
        agent_trails: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        step: int = 0,
        stats: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Render a single frame
        
        Args:
            grid: Grid world state
            agent_positions: agent_id -> (x, y)
            agent_types: agent_id -> type
            agent_trails: agent_id -> list of recent positions
            step: Current step number
            stats: Optional statistics to display
            
        Returns:
            RGB array of frame
        """
        # Convert grid to RGB
        rgb = np.ones((self.grid_size, self.grid_size, 3))
        
        # Color cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = grid[i, j]
                if cell == 1:  # Resource
                    rgb[i, j] = [0.3, 0.8, 0.3]  # Green
                elif cell == 2:  # Obstacle
                    rgb[i, j] = [0.26, 0.26, 0.26]  # Dark gray
                elif cell == 3:  # Predator
                    rgb[i, j] = [0.96, 0.26, 0.21]  # Red
        
        # Draw agent trails
        if self.show_trails and agent_trails:
            for agent_id, trail in agent_trails.items():
                agent_type = agent_types.get(agent_id, 'autopoietic')
                color = self._hex_to_rgb(AGENT_COLORS.get(agent_type, '#2196F3'))
                
                for idx, (x, y) in enumerate(trail[-self.trail_length:]):
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        alpha = 0.3 * (idx + 1) / self.trail_length
                        rgb[x, y] = rgb[x, y] * (1 - alpha) + np.array(color) * alpha
        
        # Draw agents
        for agent_id, (x, y) in agent_positions.items():
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                agent_type = agent_types.get(agent_id, 'autopoietic')
                color = self._hex_to_rgb(AGENT_COLORS.get(agent_type, '#2196F3'))
                rgb[x, y] = color
                
                # Draw larger marker
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            rgb[nx, ny] = color
        
        return rgb
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to RGB tuple (0-1)"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    
    def save_frame(self, rgb: np.ndarray, filepath: str, step: int = 0, stats: Optional[Dict] = None):
        """Save a single frame as image"""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(rgb, origin='lower')
        ax.set_title(f'Step {step}', fontsize=14)
        
        if stats:
            stats_text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=100)
        plt.close(fig)
    
    def create_animation(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 10,
        stats_history: Optional[List[Dict]] = None
    ):
        """
        Create animation from frames
        
        Args:
            frames: List of RGB frames
            output_path: Output file path (mp4 or gif)
            fps: Frames per second
            stats_history: Optional statistics per frame
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(frames[0], origin='lower')
        title = ax.set_title('Step 0', fontsize=14)
        
        if stats_history:
            stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                                verticalalignment='top', 
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def update(frame_idx):
            im.set_array(frames[frame_idx])
            title.set_text(f'Step {frame_idx * 10}')
            
            if stats_history and frame_idx < len(stats_history):
                text = '\n'.join([f'{k}: {v}' for k, v in stats_history[frame_idx].items()])
                stats_text.set_text(text)
            
            return [im, title]
        
        anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=True)
        
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        else:
            anim.save(output_path, writer='ffmpeg', fps=fps)
        
        plt.close(fig)
        print(f"Animation saved to {output_path}")


class PopulationPlotter:
    """
    Plot population dynamics over time
    """
    
    @staticmethod
    def plot_population_curves(
        population_history: Dict[str, List[Tuple[int, int]]],
        output_path: Optional[str] = None,
        title: str = "Population Dynamics"
    ):
        """
        Plot population over time by agent type
        
        Args:
            population_history: type -> [(time, count)]
            output_path: Optional file path to save
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for agent_type, history in population_history.items():
            times = [h[0] for h in history]
            counts = [h[1] for h in history]
            color = AGENT_COLORS.get(agent_type, '#000000')
            ax.plot(times, counts, label=agent_type, color=color, linewidth=2)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Population', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Population plot saved to {output_path}")
        
        plt.close(fig)
        return fig
    
    @staticmethod
    def plot_survival_curves(
        survival_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
        output_path: Optional[str] = None,
        title: str = "Survival Curves"
    ):
        """
        Plot survival curves by agent type
        
        Args:
            survival_curves: type -> (times, survival_fractions)
            output_path: Optional file path to save
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for agent_type, (times, survival) in survival_curves.items():
            color = AGENT_COLORS.get(agent_type, '#000000')
            ax.plot(times, survival, label=agent_type, color=color, linewidth=2)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Survival Fraction', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Survival plot saved to {output_path}")
        
        plt.close(fig)
        return fig
    
    @staticmethod
    def plot_diversity_over_time(
        diversity_history: List[Dict],
        output_path: Optional[str] = None,
        title: str = "Diversity Metrics Over Time"
    ):
        """
        Plot diversity metrics over time
        
        Args:
            diversity_history: List of diversity metric dicts
            output_path: Optional file path to save
            title: Plot title
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        times = range(len(diversity_history))
        
        # Behavioral entropy
        be = [d.get('behavioral_entropy', 0) for d in diversity_history]
        axes[0].plot(times, be, 'b-', linewidth=2)
        axes[0].set_title('Behavioral Entropy', fontsize=12)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Entropy')
        axes[0].grid(True, alpha=0.3)
        
        # Pairwise diversity
        pd = [d.get('pairwise_diversity', 0) for d in diversity_history]
        axes[1].plot(times, pd, 'g-', linewidth=2)
        axes[1].set_title('Pairwise Diversity', fontsize=12)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Mean Distance')
        axes[1].grid(True, alpha=0.3)
        
        # Action diversity
        ad = [d.get('action_diversity', 0) for d in diversity_history]
        axes[2].plot(times, ad, 'r-', linewidth=2)
        axes[2].set_title('Action Diversity', fontsize=12)
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Action Entropy')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Diversity plot saved to {output_path}")
        
        plt.close(fig)
        return fig


class BehaviorVisualizer:
    """
    Visualize behavioral patterns
    """
    
    @staticmethod
    def plot_behavioral_embedding(
        embeddings: np.ndarray,
        agent_types: List[str],
        clusters: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        title: str = "Behavioral Embedding (t-SNE)"
    ):
        """
        Plot 2D behavioral embedding
        
        Args:
            embeddings: (n_agents, 2) array
            agent_types: List of agent types
            clusters: Optional cluster labels
            output_path: Optional file path to save
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot by agent type
        unique_types = list(set(agent_types))
        for agent_type in unique_types:
            mask = np.array([t == agent_type for t in agent_types])
            color = AGENT_COLORS.get(agent_type, '#000000')
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                      c=color, label=agent_type, s=50, alpha=0.7)
        
        # Add cluster boundaries if available
        if clusters is not None:
            # Draw cluster centers
            unique_clusters = list(set(clusters))
            for c in unique_clusters:
                if c == -1:  # Noise in DBSCAN
                    continue
                mask = clusters == c
                center = np.mean(embeddings[mask], axis=0)
                ax.scatter(center[0], center[1], marker='x', s=200, 
                          c='black', linewidths=3)
        
        ax.set_xlabel('Embedding Dimension 1', fontsize=12)
        ax.set_ylabel('Embedding Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Embedding plot saved to {output_path}")
        
        plt.close(fig)
        return fig
    
    @staticmethod
    def plot_action_distribution(
        action_counts: Dict[str, np.ndarray],
        output_path: Optional[str] = None,
        title: str = "Action Distribution by Agent Type"
    ):
        """
        Plot action distribution comparison
        
        Args:
            action_counts: type -> action counts array
            output_path: Optional file path to save
            title: Plot title
        """
        action_names = ['Stay', 'Up', 'Down', 'Left', 'Right', 'Consume']
        n_actions = len(action_names)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_actions)
        width = 0.2
        
        for i, (agent_type, counts) in enumerate(action_counts.items()):
            # Normalize
            if np.sum(counts) > 0:
                counts = counts / np.sum(counts)
            
            color = AGENT_COLORS.get(agent_type, '#000000')
            ax.bar(x + i * width, counts[:n_actions], width, label=agent_type, color=color, alpha=0.8)
        
        ax.set_xlabel('Action', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x + width * (len(action_counts) - 1) / 2)
        ax.set_xticklabels(action_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Action distribution plot saved to {output_path}")
        
        plt.close(fig)
        return fig


class NicheVisualizer:
    """
    Visualize spatial niche specialization
    """
    
    @staticmethod
    def plot_niche_heatmap(
        position_histories: Dict[str, List[Tuple[int, int]]],
        agent_types: Dict[str, str],
        grid_size: int = 100,
        output_path: Optional[str] = None,
        title: str = "Spatial Niche Heatmap"
    ):
        """
        Plot heatmap of agent positions by type
        
        Args:
            position_histories: agent_id -> position history
            agent_types: agent_id -> type
            grid_size: Grid dimension
            output_path: Optional file path to save
            title: Plot title
        """
        unique_types = list(set(agent_types.values()))
        n_types = len(unique_types)
        
        fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5))
        if n_types == 1:
            axes = [axes]
        
        for ax, agent_type in zip(axes, unique_types):
            heatmap = np.zeros((grid_size, grid_size))
            
            for agent_id, positions in position_histories.items():
                if agent_types.get(agent_id) == agent_type:
                    for x, y in positions:
                        if 0 <= x < grid_size and 0 <= y < grid_size:
                            heatmap[x, y] += 1
            
            # Log scale for better visualization
            heatmap = np.log1p(heatmap)
            
            im = ax.imshow(heatmap, origin='lower', cmap='hot')
            ax.set_title(f'{agent_type}', fontsize=12)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Log visits')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Niche heatmap saved to {output_path}")
        
        plt.close(fig)
        return fig
    
    @staticmethod
    def plot_specialization_comparison(
        specializations: Dict[str, Dict[str, float]],
        output_path: Optional[str] = None,
        title: str = "Niche Specialization Comparison"
    ):
        """
        Compare niche specialization across agent types
        
        Args:
            specializations: type -> {agent_id: specialization_score}
            output_path: Optional file path to save
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = []
        labels = []
        colors = []
        
        for agent_type, specs in specializations.items():
            values = list(specs.values())
            if len(values) > 0:
                data.append(values)
                labels.append(agent_type)
                colors.append(AGENT_COLORS.get(agent_type, '#000000'))
        
        if len(data) > 0:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax.set_xlabel('Agent Type', fontsize=12)
        ax.set_ylabel('Niche Specialization', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Specialization plot saved to {output_path}")
        
        plt.close(fig)
        return fig


class ComprehensiveReport:
    """
    Generate comprehensive visualization report
    """
    
    @staticmethod
    def generate_report(
        results: Dict,
        output_dir: str,
        prefix: str = "experiment"
    ):
        """
        Generate all visualizations from experiment results
        
        Args:
            results: Experiment results dictionary
            output_dir: Output directory for plots
            prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Population dynamics
        if 'population_history' in results:
            PopulationPlotter.plot_population_curves(
                results['population_history'],
                os.path.join(output_dir, f'{prefix}_population.png')
            )
        
        # 2. Survival curves
        if 'survival_curves' in results:
            PopulationPlotter.plot_survival_curves(
                results['survival_curves'],
                os.path.join(output_dir, f'{prefix}_survival.png')
            )
        
        # 3. Diversity metrics
        if 'diversity_history' in results:
            PopulationPlotter.plot_diversity_over_time(
                results['diversity_history'],
                os.path.join(output_dir, f'{prefix}_diversity.png')
            )
        
        # 4. Behavioral embedding
        if 'embeddings' in results and 'agent_types' in results:
            BehaviorVisualizer.plot_behavioral_embedding(
                results['embeddings'],
                results['agent_types'],
                results.get('clusters'),
                os.path.join(output_dir, f'{prefix}_embedding.png')
            )
        
        # 5. Action distribution
        if 'action_counts' in results:
            BehaviorVisualizer.plot_action_distribution(
                results['action_counts'],
                os.path.join(output_dir, f'{prefix}_actions.png')
            )
        
        # 6. Niche heatmaps
        if 'position_histories' in results and 'agent_types_dict' in results:
            NicheVisualizer.plot_niche_heatmap(
                results['position_histories'],
                results['agent_types_dict'],
                output_path=os.path.join(output_dir, f'{prefix}_niches.png')
            )
        
        print(f"\nReport generated in {output_dir}")


# =====================
# Testing
# =====================

if __name__ == "__main__":
    print("=" * 70)
    print("Visualization Tools Test")
    print("=" * 70)
    
    # Create test data
    np.random.seed(42)
    
    # Test population plot
    print("\n1. Testing Population Plot")
    pop_history = {
        'autopoietic': [(i, 10 + np.random.randint(-2, 3)) for i in range(100)],
        'rl': [(i, 8 + np.random.randint(-2, 3)) for i in range(100)],
        'neat': [(i, 6 + np.random.randint(-2, 3)) for i in range(100)],
        'random': [(i, max(0, 5 - i//20 + np.random.randint(-1, 2))) for i in range(100)]
    }
    
    PopulationPlotter.plot_population_curves(
        pop_history,
        '/Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_b_artificial_life/results/test_population.png'
    )
    
    # Test behavioral embedding
    print("\n2. Testing Behavioral Embedding")
    embeddings = np.random.randn(40, 2)
    embeddings[:10] += [2, 2]  # Autopoietic cluster
    embeddings[10:20] += [-2, 2]  # RL cluster
    embeddings[20:30] += [0, -2]  # NEAT cluster
    
    agent_types = ['autopoietic'] * 10 + ['rl'] * 10 + ['neat'] * 10 + ['random'] * 10
    
    BehaviorVisualizer.plot_behavioral_embedding(
        embeddings,
        agent_types,
        output_path='/Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_b_artificial_life/results/test_embedding.png'
    )
    
    # Test action distribution
    print("\n3. Testing Action Distribution")
    action_counts = {
        'autopoietic': np.array([10, 20, 20, 15, 15, 30]),
        'rl': np.array([5, 25, 25, 20, 20, 15]),
        'neat': np.array([15, 18, 18, 18, 18, 25]),
        'random': np.array([17, 17, 17, 17, 17, 15])
    }
    
    BehaviorVisualizer.plot_action_distribution(
        action_counts,
        output_path='/Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_b_artificial_life/results/test_actions.png'
    )
    
    print("\n" + "=" * 70)
    print("Visualization test complete!")
    print("Check results/ directory for output files.")
    print("=" * 70)
