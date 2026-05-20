"""
Relationship module — state machine for relationship phases.

Components:
- Stage: 6 relationship phases (courting, sweet, passionate, stable, cold, repairing)
- Event: Observable events that trigger transitions
- RelationshipStateMachine: HMM + OU process hybrid
- OUProcess: Ornstein-Uhlenbeck mean-reverting process
"""

from .ou_process import OUProcess
from .stages import RelationshipState, Stage
from .state_machine import Event, RelationshipStateMachine, TransitionConfig

__all__ = [
    "Stage",
    "RelationshipState",
    "Event",
    "RelationshipStateMachine",
    "TransitionConfig",
    "OUProcess",
]
