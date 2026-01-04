"""
GENESIS Phase 4C Integration: Emergent Communication Manager

Integrates emergent communication with Phase 4B open-ended learning system.
"""

import numpy as np
from typing import Dict, List, Optional

# Phase 4B
from phase4b_integration import Phase4B_OpenEndedManager, create_phase4b_system
from full_environment import FullALifeEnvironment

# Phase 4C
from emergent_communication import (
    CommunicatingAgent,
    CommunicationManager,
    MessageAnalyzer
)


class Phase4C_CommunicationManager(Phase4B_OpenEndedManager):
    """
    Phase 4C: Adds emergent communication to Phase 4B

    Agents can:
    - Send messages encoding internal state
    - Receive and process messages from others
    - Develop communication protocols through evolution
    - Coordinate behavior through language
    """

    def __init__(self,
                 env: FullALifeEnvironment,
                 # Inherit all Phase 4A/4B parameters
                 **kwargs):
        """
        Initialize Phase 4C system

        Args:
            env: Environment
            **kwargs: All Phase 4A/4B parameters
        """
        # Extract Phase 4C specific parameters
        self.phase4c_enabled = kwargs.pop('phase4c_enabled', True)
        message_dim = kwargs.pop('message_dim', 8)
        influence_dim = kwargs.pop('influence_dim', 32)
        local_radius = kwargs.pop('local_radius', 5.0)

        # Initialize Phase 4B
        super().__init__(env=env, **kwargs)

        if not self.phase4c_enabled:
            self.comm_manager = None
            self.communicating_agents = {}
            self.message_analyzer = None
            return

        # Communication manager
        self.comm_manager = CommunicationManager(local_radius=local_radius)

        # Message analyzer
        self.message_analyzer = MessageAnalyzer()

        # Wrap agents with communication capabilities
        self.communicating_agents = {}
        self._wrap_agents_with_communication(message_dim, influence_dim)

        # Statistics
        self.total_coordination_events = 0
        self.avg_communication_benefit = 0.0

    def _wrap_agents_with_communication(self, message_dim: int, influence_dim: int):
        """Wrap agents with communication capabilities"""
        for i, agent in enumerate(self.agents):
            if agent.id not in self.communicating_agents:
                comm_agent = CommunicatingAgent(
                    agent=agent,
                    agent_id=agent.id,
                    message_dim=message_dim,
                    influence_dim=influence_dim
                )
                self.communicating_agents[agent.id] = comm_agent

    def step(self) -> Dict:
        """Execute one step with Phase 4C communication"""
        if not self.phase4c_enabled:
            return super().step()

        # Phase 4B step (includes 4A, 3, 2, 1)
        stats = super().step()

        # Ensure new agents are wrapped
        self._wrap_agents_with_communication(8, 32)

        # Communication step
        if self.comm_manager:
            comm_agents = [
                self.communicating_agents[agent.id]
                for agent in self.agents
                if agent.id in self.communicating_agents
            ]

            self.comm_manager.step(comm_agents)

            # Analyze messages
            for msg in self.comm_manager.broadcast_messages + [m for m, _ in self.comm_manager.local_messages]:
                context = {
                    'step': self.current_step,
                    'sender_coherence': 0.5  # Would get from actual agent
                }
                self.message_analyzer.analyze_message(msg, context)

        # Add Phase 4C statistics
        if self.phase4c_enabled:
            stats['phase4c'] = {}

            if self.comm_manager:
                comm_stats = self.comm_manager.get_statistics()
                stats['phase4c']['communication'] = comm_stats

                # Per-agent stats
                if self.communicating_agents:
                    agent_stats = [
                        ca.get_statistics()
                        for ca in self.communicating_agents.values()
                    ]

                    stats['phase4c']['avg_messages_sent'] = np.mean([s['messages_sent'] for s in agent_stats])
                    stats['phase4c']['avg_messages_received'] = np.mean([s['messages_received'] for s in agent_stats])

            if self.message_analyzer:
                analysis_stats = self.message_analyzer.get_statistics()
                stats['phase4c']['protocol_analysis'] = analysis_stats

        return stats

    def get_communication_statistics(self) -> Dict:
        """Get detailed communication statistics"""
        if not self.phase4c_enabled:
            return {}

        stats = {
            'total_agents': len(self.agents),
            'communicating_agents': len(self.communicating_agents)
        }

        if self.comm_manager:
            stats['manager'] = self.comm_manager.get_statistics()

        if self.message_analyzer:
            stats['protocol'] = self.message_analyzer.get_statistics()

        # Per-agent breakdown
        if self.communicating_agents:
            agent_stats = [ca.get_statistics() for ca in self.communicating_agents.values()]

            stats['per_agent'] = {
                'avg_messages_sent': np.mean([s['messages_sent'] for s in agent_stats]),
                'max_messages_sent': max([s['messages_sent'] for s in agent_stats]),
                'avg_messages_received': np.mean([s['messages_received'] for s in agent_stats]),
                'communication_rate': sum(1 for s in agent_stats if s['messages_sent'] > 0) / len(agent_stats)
            }

        return stats


def create_phase4c_system(
        env_size: int = 50,
        initial_population: int = 300,
        phase1_enabled: bool = True,
        phase2_enabled: bool = True,
        phase3_enabled: bool = True,
        phase4_enabled: bool = True,
        phase4b_enabled: bool = True,
        phase4c_enabled: bool = True,
        use_novelty_search: bool = True,
        use_map_elites: bool = True,
        use_poet: bool = False,
        message_dim: int = 8,
        local_radius: float = 5.0) -> Phase4C_CommunicationManager:
    """
    Create complete Phase 4C system

    Args:
        env_size: Environment size
        initial_population: Initial population
        phase1_enabled: Enable Phase 1 (Infinite Learning)
        phase2_enabled: Enable Phase 2 (Multi-layer Memory)
        phase3_enabled: Enable Phase 3 (Universal Knowledge)
        phase4_enabled: Enable Phase 4A (Advanced Intelligence)
        phase4b_enabled: Enable Phase 4B (Open-Ended Learning)
        phase4c_enabled: Enable Phase 4C (Emergent Communication)
        use_novelty_search: Use novelty search
        use_map_elites: Use MAP-Elites
        use_poet: Use POET
        message_dim: Dimension of communication messages
        local_radius: Radius for local communication

    Returns:
        Phase4C_CommunicationManager
    """
    # Create environment
    env = FullALifeEnvironment(size=env_size)

    # Create manager
    manager = Phase4C_CommunicationManager(
        env=env,
        initial_pop=initial_population,
        max_population=500,
        phase2_enabled=phase2_enabled,
        phase3_enabled=phase3_enabled,
        phase4_enabled=phase4_enabled,
        use_advanced_teacher=phase4_enabled,
        use_learned_memory=phase4_enabled,
        use_knowledge_guidance=phase4_enabled,
        phase4b_enabled=phase4b_enabled,
        use_novelty_search=use_novelty_search,
        use_map_elites=use_map_elites,
        use_poet=use_poet,
        phase4c_enabled=phase4c_enabled,
        message_dim=message_dim,
        influence_dim=32,
        local_radius=local_radius
    )

    return manager
