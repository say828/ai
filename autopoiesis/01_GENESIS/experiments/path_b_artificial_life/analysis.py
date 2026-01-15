"""
GENESIS Path B: Analysis Tools
==============================

Tools for analyzing artificial life experiments:
- Behavioral clustering (t-SNE on trajectories)
- Diversity metrics (behavioral entropy)
- Emergence detection (complexity measures)
- Survival statistics

Author: GENESIS Project
Date: 2026-01-04
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class BehavioralAnalyzer:
    """
    Analyzes behavioral patterns across agent populations
    """
    
    def __init__(self):
        self.trajectory_features = {}  # agent_id -> features
        self.agent_types = {}  # agent_id -> type
        self.embeddings = None
        self.clusters = None
    
    def add_agent(self, agent_id: str, features: np.ndarray, agent_type: str):
        """Add agent trajectory features"""
        self.trajectory_features[agent_id] = features
        self.agent_types[agent_id] = agent_type
    
    def compute_embeddings(self, method: str = "tsne", n_components: int = 2) -> np.ndarray:
        """
        Compute 2D embeddings of behavioral features
        
        Args:
            method: "tsne" or "pca"
            n_components: Number of dimensions
            
        Returns:
            (n_agents, n_components) embeddings
        """
        if len(self.trajectory_features) < 5:
            print("Warning: Need at least 5 agents for embedding")
            return np.zeros((len(self.trajectory_features), n_components))
        
        features = np.array(list(self.trajectory_features.values()))
        
        # Handle NaN and Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize features
        means = np.mean(features, axis=0, keepdims=True)
        stds = np.std(features, axis=0, keepdims=True) + 1e-8
        features_norm = (features - means) / stds
        
        if method == "tsne":
            # Adjust perplexity based on sample size
            # perplexity must be less than n_samples
            perplexity = min(30, max(1, len(features) - 2))
            perplexity = max(2, perplexity)  # Minimum 2

            # Fall back to PCA if too few samples
            if len(features) < 6:
                pca = PCA(n_components=n_components)
                self.embeddings = pca.fit_transform(features_norm)
            else:
                tsne = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    random_state=42,
                    max_iter=500  # Must be >= 250 in newer sklearn
                )
                self.embeddings = tsne.fit_transform(features_norm)
        else:  # PCA
            pca = PCA(n_components=n_components)
            self.embeddings = pca.fit_transform(features_norm)
        
        return self.embeddings
    
    def cluster_behaviors(self, n_clusters: int = 5, method: str = "kmeans") -> np.ndarray:
        """
        Cluster agents by behavior
        
        Args:
            n_clusters: Number of clusters (for kmeans)
            method: "kmeans" or "dbscan"
            
        Returns:
            Cluster labels
        """
        if len(self.trajectory_features) < 3:
            return np.zeros(len(self.trajectory_features), dtype=int)
        
        features = np.array(list(self.trajectory_features.values()))
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize
        means = np.mean(features, axis=0, keepdims=True)
        stds = np.std(features, axis=0, keepdims=True) + 1e-8
        features_norm = (features - means) / stds
        
        if method == "kmeans":
            n_clusters = min(n_clusters, len(features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = kmeans.fit_predict(features_norm)
        else:  # DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            self.clusters = dbscan.fit_predict(features_norm)
        
        return self.clusters
    
    def get_cluster_stats(self) -> Dict:
        """Get statistics per cluster"""
        if self.clusters is None:
            self.cluster_behaviors()
        
        agent_ids = list(self.trajectory_features.keys())
        
        stats = defaultdict(lambda: {'count': 0, 'types': defaultdict(int)})
        
        for i, (agent_id, cluster) in enumerate(zip(agent_ids, self.clusters)):
            agent_type = self.agent_types.get(agent_id, "unknown")
            stats[cluster]['count'] += 1
            stats[cluster]['types'][agent_type] += 1
        
        return dict(stats)


class DiversityMetrics:
    """
    Compute diversity metrics for populations
    """
    
    @staticmethod
    def behavioral_entropy(features: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute entropy of behavioral feature distribution
        
        Higher entropy = more diverse behaviors
        
        Args:
            features: (n_agents, n_features) array
            n_bins: Number of bins for discretization
            
        Returns:
            Normalized entropy (0-1)
        """
        if len(features) < 2:
            return 0.0
        
        total_entropy = 0.0
        n_features = features.shape[1]
        
        for i in range(n_features):
            # Discretize feature
            hist, _ = np.histogram(features[:, i], bins=n_bins)
            hist = hist / np.sum(hist)  # Normalize
            
            # Compute entropy (ignoring zeros)
            hist = hist[hist > 0]
            if len(hist) > 0:
                total_entropy += entropy(hist)
        
        # Normalize by maximum possible entropy
        max_entropy = n_features * np.log(n_bins)
        normalized_entropy = total_entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    @staticmethod
    def pairwise_diversity(features: np.ndarray) -> float:
        """
        Average pairwise distance between agents
        
        Args:
            features: (n_agents, n_features) array
            
        Returns:
            Mean pairwise distance
        """
        if len(features) < 2:
            return 0.0
        
        # Normalize features
        features = np.nan_to_num(features, nan=0.0)
        means = np.mean(features, axis=0, keepdims=True)
        stds = np.std(features, axis=0, keepdims=True) + 1e-8
        features_norm = (features - means) / stds
        
        # Compute pairwise distances
        distances = pdist(features_norm, metric='euclidean')
        
        return float(np.mean(distances))
    
    @staticmethod
    def action_diversity(action_histories: List[List[int]], n_actions: int = 6) -> float:
        """
        Measure diversity in action distributions
        
        Args:
            action_histories: List of action sequences per agent
            n_actions: Number of possible actions
            
        Returns:
            Normalized action entropy
        """
        if len(action_histories) == 0:
            return 0.0
        
        # Aggregate action counts
        all_actions = []
        for history in action_histories:
            all_actions.extend(history)
        
        if len(all_actions) == 0:
            return 0.0
        
        # Compute distribution
        counts = np.bincount(all_actions, minlength=n_actions)
        probs = counts / np.sum(counts)
        
        # Entropy
        probs = probs[probs > 0]
        action_entropy = entropy(probs)
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(n_actions)
        
        return float(action_entropy / max_entropy) if max_entropy > 0 else 0.0


class EmergenceDetector:
    """
    Detect emergent behaviors and patterns
    """
    
    @staticmethod
    def complexity_measure(states: np.ndarray) -> float:
        """
        Compute statistical complexity of state sequence
        
        Based on Shannon entropy of state transitions.
        
        Args:
            states: Time series of internal states
            
        Returns:
            Complexity score (0-1)
        """
        if len(states) < 10:
            return 0.0
        
        # Discretize states
        n_bins = min(10, len(states) // 5)
        if n_bins < 2:
            return 0.0
        
        # Flatten if multi-dimensional
        if len(states.shape) > 1:
            states = np.mean(states, axis=1)
        
        # Bin the states
        bins = np.linspace(np.min(states), np.max(states), n_bins + 1)
        digitized = np.digitize(states, bins) - 1
        digitized = np.clip(digitized, 0, n_bins - 1)
        
        # Compute transition matrix
        transitions = np.zeros((n_bins, n_bins))
        for i in range(len(digitized) - 1):
            transitions[digitized[i], digitized[i + 1]] += 1
        
        # Normalize rows
        row_sums = np.sum(transitions, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trans_probs = transitions / row_sums
        
        # Compute entropy of transition probabilities
        total_entropy = 0.0
        for i in range(n_bins):
            probs = trans_probs[i]
            probs = probs[probs > 0]
            if len(probs) > 0:
                total_entropy += entropy(probs)
        
        # Normalize
        max_entropy = n_bins * np.log(n_bins)
        complexity = total_entropy / max_entropy if max_entropy > 0 else 0
        
        return float(complexity)
    
    @staticmethod
    def detect_niches(positions: Dict[str, List[Tuple[int, int]]], 
                      grid_size: int = 100,
                      n_regions: int = 4) -> Dict:
        """
        Detect spatial niche specialization
        
        Args:
            positions: agent_id -> list of (x, y) positions
            grid_size: Size of grid
            n_regions: Number of regions to divide grid into
            
        Returns:
            Niche statistics
        """
        region_size = grid_size // n_regions
        
        # Count visits per region per agent
        agent_region_counts = {}
        
        for agent_id, pos_list in positions.items():
            region_counts = np.zeros(n_regions * n_regions)
            
            for x, y in pos_list:
                rx = min(x // region_size, n_regions - 1)
                ry = min(y // region_size, n_regions - 1)
                region_idx = rx * n_regions + ry
                region_counts[region_idx] += 1
            
            # Normalize
            if np.sum(region_counts) > 0:
                region_counts /= np.sum(region_counts)
            
            agent_region_counts[agent_id] = region_counts
        
        # Compute niche specialization (concentration)
        specializations = {}
        for agent_id, counts in agent_region_counts.items():
            # High specialization = low entropy (concentrated in few regions)
            counts = counts[counts > 0]
            if len(counts) > 0:
                spec = 1.0 - (entropy(counts) / np.log(n_regions * n_regions))
            else:
                spec = 0.0
            specializations[agent_id] = spec
        
        return {
            'region_counts': agent_region_counts,
            'specializations': specializations,
            'mean_specialization': np.mean(list(specializations.values())) if specializations else 0.0
        }
    
    @staticmethod
    def detect_collective_patterns(positions: Dict[str, Tuple[int, int]], 
                                   threshold: float = 10.0) -> Dict:
        """
        Detect collective behavior patterns (clustering, flocking)
        
        Args:
            positions: Current positions of all agents
            threshold: Distance threshold for "near" agents
            
        Returns:
            Pattern statistics
        """
        if len(positions) < 2:
            return {'clustering_coefficient': 0.0, 'n_groups': 0}
        
        pos_array = np.array(list(positions.values()))
        
        # Compute pairwise distances
        if len(pos_array) > 1:
            dist_matrix = squareform(pdist(pos_array))
        else:
            return {'clustering_coefficient': 0.0, 'n_groups': 1}
        
        # Count nearby neighbors
        n_near = np.sum(dist_matrix < threshold, axis=1) - 1  # Exclude self
        
        # Clustering coefficient (average fraction of population nearby)
        clustering = np.mean(n_near) / max(1, len(positions) - 1)
        
        # Detect groups using DBSCAN
        if len(pos_array) >= 3:
            dbscan = DBSCAN(eps=threshold, min_samples=2)
            labels = dbscan.fit_predict(pos_array)
            n_groups = len(set(labels)) - (1 if -1 in labels else 0)
        else:
            n_groups = 1
        
        return {
            'clustering_coefficient': float(clustering),
            'n_groups': n_groups,
            'avg_neighbors': float(np.mean(n_near))
        }


class SurvivalAnalyzer:
    """
    Analyze survival statistics
    """
    
    def __init__(self):
        self.lifespans = defaultdict(list)  # agent_type -> lifespans
        self.death_causes = defaultdict(lambda: defaultdict(int))  # type -> cause -> count
    
    def record_death(self, agent_type: str, lifespan: int, cause: str = "unknown"):
        """Record agent death"""
        self.lifespans[agent_type].append(lifespan)
        self.death_causes[agent_type][cause] += 1
    
    def record_lifespan(self, agent_type: str, lifespan: int):
        """Record lifespan (for living agents too)"""
        self.lifespans[agent_type].append(lifespan)
    
    def get_survival_stats(self) -> Dict:
        """Get survival statistics by agent type"""
        stats = {}
        
        for agent_type, spans in self.lifespans.items():
            if len(spans) == 0:
                continue
            
            spans = np.array(spans)
            stats[agent_type] = {
                'mean_lifespan': float(np.mean(spans)),
                'std_lifespan': float(np.std(spans)),
                'median_lifespan': float(np.median(spans)),
                'max_lifespan': int(np.max(spans)),
                'min_lifespan': int(np.min(spans)),
                'count': len(spans),
                'death_causes': dict(self.death_causes[agent_type])
            }
        
        return stats
    
    def survival_curve(self, agent_type: str, max_time: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute survival curve (Kaplan-Meier style)
        
        Args:
            agent_type: Type of agent
            max_time: Maximum time to consider
            
        Returns:
            (times, survival_fractions)
        """
        if agent_type not in self.lifespans or len(self.lifespans[agent_type]) == 0:
            return np.array([0, max_time]), np.array([1.0, 1.0])
        
        spans = np.array(self.lifespans[agent_type])
        n_total = len(spans)
        
        # Create survival curve
        times = np.arange(0, max_time + 1, 100)
        survival = np.zeros(len(times))
        
        for i, t in enumerate(times):
            survival[i] = np.sum(spans >= t) / n_total
        
        return times, survival


class PopulationDynamicsAnalyzer:
    """
    Analyze population dynamics over time
    """
    
    def __init__(self):
        self.population_history = defaultdict(list)  # type -> [(time, count)]
        self.birth_events = []  # (time, type)
        self.death_events = []  # (time, type)
    
    def record_population(self, time: int, counts: Dict[str, int]):
        """Record population counts at a time step"""
        for agent_type, count in counts.items():
            self.population_history[agent_type].append((time, count))
    
    def record_birth(self, time: int, agent_type: str):
        """Record birth event"""
        self.birth_events.append((time, agent_type))
    
    def record_death(self, time: int, agent_type: str):
        """Record death event"""
        self.death_events.append((time, agent_type))
    
    def get_population_curves(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get population curves for each type"""
        curves = {}
        
        for agent_type, history in self.population_history.items():
            if len(history) == 0:
                continue
            
            times = np.array([h[0] for h in history])
            counts = np.array([h[1] for h in history])
            curves[agent_type] = (times, counts)
        
        return curves
    
    def get_birth_death_rates(self, window: int = 100) -> Dict:
        """Compute birth/death rates over time"""
        if len(self.birth_events) == 0 and len(self.death_events) == 0:
            return {}
        
        max_time = max(
            max([e[0] for e in self.birth_events]) if self.birth_events else 0,
            max([e[0] for e in self.death_events]) if self.death_events else 0
        )
        
        rates = {}
        for agent_type in set([e[1] for e in self.birth_events + self.death_events]):
            births = [e[0] for e in self.birth_events if e[1] == agent_type]
            deaths = [e[0] for e in self.death_events if e[1] == agent_type]
            
            # Compute rates in windows
            n_windows = max(1, max_time // window)
            birth_rate = np.zeros(n_windows)
            death_rate = np.zeros(n_windows)
            
            for b in births:
                idx = min(b // window, n_windows - 1)
                birth_rate[idx] += 1
            
            for d in deaths:
                idx = min(d // window, n_windows - 1)
                death_rate[idx] += 1
            
            rates[agent_type] = {
                'birth_rate': birth_rate / window,
                'death_rate': death_rate / window,
                'net_growth': (birth_rate - death_rate) / window
            }
        
        return rates


# =====================
# Comprehensive Analysis
# =====================

def run_comprehensive_analysis(
    agents: List,
    positions: Dict[str, Tuple[int, int]],
    position_histories: Dict[str, List[Tuple[int, int]]],
    grid_size: int = 100
) -> Dict:
    """
    Run all analysis tools on a population
    
    Args:
        agents: List of agent objects
        positions: Current positions
        position_histories: Historical positions
        grid_size: Grid size
        
    Returns:
        Comprehensive analysis results
    """
    results = {}
    
    # 1. Behavioral Analysis
    behavioral = BehavioralAnalyzer()
    for agent in agents:
        if hasattr(agent, 'get_trajectory_features'):
            features = agent.get_trajectory_features()
            agent_type = getattr(agent, 'agent_type', 'unknown')
            behavioral.add_agent(agent.id, features, agent_type)
    
    if len(behavioral.trajectory_features) >= 5:
        behavioral.compute_embeddings()
        behavioral.cluster_behaviors()
        results['behavioral'] = {
            'n_clusters': len(set(behavioral.clusters)),
            'cluster_stats': behavioral.get_cluster_stats()
        }
    else:
        results['behavioral'] = {'n_clusters': 0, 'cluster_stats': {}}
    
    # 2. Diversity Metrics
    if len(behavioral.trajectory_features) > 0:
        features = np.array(list(behavioral.trajectory_features.values()))
        results['diversity'] = {
            'behavioral_entropy': DiversityMetrics.behavioral_entropy(features),
            'pairwise_diversity': DiversityMetrics.pairwise_diversity(features)
        }
        
        # Action diversity
        action_histories = []
        for agent in agents:
            if hasattr(agent, 'action_history'):
                action_histories.append(list(agent.action_history))
        
        results['diversity']['action_diversity'] = DiversityMetrics.action_diversity(action_histories)
    else:
        results['diversity'] = {
            'behavioral_entropy': 0.0,
            'pairwise_diversity': 0.0,
            'action_diversity': 0.0
        }
    
    # 3. Emergence Detection
    results['emergence'] = {}
    
    # Niche specialization
    if len(position_histories) > 0:
        niche_stats = EmergenceDetector.detect_niches(position_histories, grid_size)
        results['emergence']['niche_specialization'] = niche_stats['mean_specialization']
    else:
        results['emergence']['niche_specialization'] = 0.0
    
    # Collective patterns
    if len(positions) > 0:
        collective = EmergenceDetector.detect_collective_patterns(positions)
        results['emergence']['clustering_coefficient'] = collective['clustering_coefficient']
        results['emergence']['n_groups'] = collective['n_groups']
    else:
        results['emergence']['clustering_coefficient'] = 0.0
        results['emergence']['n_groups'] = 0
    
    # Complexity
    complexities = []
    for agent in agents:
        if hasattr(agent, 'dynamics') and hasattr(agent.dynamics, 'state_history'):
            states = np.array(list(agent.dynamics.state_history))
            if len(states) > 10:
                c = EmergenceDetector.complexity_measure(states)
                complexities.append(c)
    
    results['emergence']['mean_complexity'] = float(np.mean(complexities)) if complexities else 0.0
    
    return results


# =====================
# Testing
# =====================

if __name__ == "__main__":
    print("=" * 70)
    print("Analysis Tools Test")
    print("=" * 70)
    
    # Create fake data
    np.random.seed(42)
    
    n_agents = 30
    features = np.random.randn(n_agents, 20)
    
    # Test behavioral analyzer
    print("\n1. Behavioral Analyzer")
    behavioral = BehavioralAnalyzer()
    for i in range(n_agents):
        agent_type = ["autopoietic", "rl", "neat", "random"][i % 4]
        behavioral.add_agent(f"agent_{i}", features[i], agent_type)
    
    embeddings = behavioral.compute_embeddings()
    print(f"   Embeddings shape: {embeddings.shape}")
    
    clusters = behavioral.cluster_behaviors(n_clusters=4)
    print(f"   Clusters: {np.bincount(clusters)}")
    
    # Test diversity metrics
    print("\n2. Diversity Metrics")
    print(f"   Behavioral entropy: {DiversityMetrics.behavioral_entropy(features):.3f}")
    print(f"   Pairwise diversity: {DiversityMetrics.pairwise_diversity(features):.3f}")
    
    # Test emergence detector
    print("\n3. Emergence Detection")
    
    # Fake positions
    positions = {f"agent_{i}": (np.random.randint(0, 100), np.random.randint(0, 100)) 
                 for i in range(n_agents)}
    
    collective = EmergenceDetector.detect_collective_patterns(positions)
    print(f"   Clustering coefficient: {collective['clustering_coefficient']:.3f}")
    print(f"   Number of groups: {collective['n_groups']}")
    
    # Test survival analyzer
    print("\n4. Survival Analyzer")
    survival = SurvivalAnalyzer()
    for i in range(50):
        agent_type = ["autopoietic", "rl", "neat", "random"][i % 4]
        lifespan = np.random.randint(100, 3000)
        survival.record_lifespan(agent_type, lifespan)
    
    stats = survival.get_survival_stats()
    for agent_type, s in stats.items():
        print(f"   {agent_type}: mean={s['mean_lifespan']:.0f}, max={s['max_lifespan']}")
    
    print("\n" + "=" * 70)
    print("Analysis tools test complete!")
    print("=" * 70)
