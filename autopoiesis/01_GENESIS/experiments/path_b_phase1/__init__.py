"""
GENESIS Path B Phase 1: Full Artificial Life System

Complete implementation of autopoietic agents in a complex environment.
"""

from .full_environment import FullALifeEnvironment, ResourceConfig
from .full_agent import FullAutopoieticAgent
from .full_population import FullPopulationManager, PhylogenyNode
from .baselines import RandomAgent, FixedPolicyAgent, RLAgent, BaselinePopulationManager

__all__ = [
    'FullALifeEnvironment',
    'ResourceConfig',
    'FullAutopoieticAgent',
    'FullPopulationManager',
    'PhylogenyNode',
    'RandomAgent',
    'FixedPolicyAgent',
    'RLAgent',
    'BaselinePopulationManager'
]
