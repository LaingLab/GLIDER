"""
Agent Actions

Defines action types and their validation/execution states.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class ActionType(Enum):
    """Types of actions the agent can perform."""

    # Experiment actions
    CREATE_NODE = auto()
    DELETE_NODE = auto()
    CONNECT_NODES = auto()
    DISCONNECT_NODES = auto()
    SET_NODE_PROPERTY = auto()
    CLEAR_FLOW = auto()
    VALIDATE_FLOW = auto()

    # Hardware actions
    ADD_BOARD = auto()
    REMOVE_BOARD = auto()
    ADD_DEVICE = auto()
    REMOVE_DEVICE = auto()
    CONFIGURE_DEVICE = auto()
    SCAN_PORTS = auto()
    TEST_DEVICE = auto()

    # Knowledge actions (safe - no confirmation needed)
    EXPLAIN = auto()
    SUGGEST = auto()
    GET_STATE = auto()
    GET_DOCUMENTATION = auto()


class ActionStatus(Enum):
    """Status of an agent action."""

    PENDING = auto()  # Awaiting confirmation
    CONFIRMED = auto()  # User confirmed
    REJECTED = auto()  # User rejected
    EXECUTING = auto()  # Currently running
    COMPLETED = auto()  # Successfully completed
    FAILED = auto()  # Execution failed


# Actions that don't require confirmation
SAFE_ACTIONS = {
    ActionType.EXPLAIN,
    ActionType.SUGGEST,
    ActionType.GET_STATE,
    ActionType.GET_DOCUMENTATION,
    ActionType.VALIDATE_FLOW,
    ActionType.SCAN_PORTS,
}

# Actions that always require confirmation
DANGEROUS_ACTIONS = {
    ActionType.CLEAR_FLOW,
    ActionType.REMOVE_BOARD,
}


@dataclass
class AgentAction:
    """Represents an action the agent wants to perform."""

    action_type: ActionType
    tool_name: str
    parameters: dict[str, Any]
    description: str  # Human-readable description

    # State
    status: ActionStatus = ActionStatus.PENDING
    result: Any | None = None
    error: str | None = None

    # Metadata
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def requires_confirmation(self) -> bool:
        """Check if this action requires user confirmation."""
        if self.action_type in SAFE_ACTIONS:
            return False
        return True

    @property
    def is_dangerous(self) -> bool:
        """Check if this is a dangerous action."""
        return self.action_type in DANGEROUS_ACTIONS

    @property
    def is_pending(self) -> bool:
        """Check if action is awaiting confirmation."""
        return self.status == ActionStatus.PENDING

    @property
    def is_complete(self) -> bool:
        """Check if action has finished (success or failure)."""
        return self.status in {ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.REJECTED}

    def confirm(self) -> None:
        """Mark action as confirmed."""
        self.status = ActionStatus.CONFIRMED

    def reject(self) -> None:
        """Mark action as rejected."""
        self.status = ActionStatus.REJECTED

    def start_execution(self) -> None:
        """Mark action as executing."""
        self.status = ActionStatus.EXECUTING

    def complete(self, result: Any) -> None:
        """Mark action as completed with result."""
        self.status = ActionStatus.COMPLETED
        self.result = result

    def fail(self, error: str) -> None:
        """Mark action as failed with error."""
        self.status = ActionStatus.FAILED
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "id": self.id,
            "type": self.action_type.name,
            "tool": self.tool_name,
            "description": self.description,
            "parameters": self.parameters,
            "status": self.status.name,
            "requires_confirmation": self.requires_confirmation,
            "is_dangerous": self.is_dangerous,
        }


@dataclass
class ActionBatch:
    """A batch of related actions from a single agent response."""

    actions: list[AgentAction] = field(default_factory=list)
    message: str = ""  # The agent's explanation

    def add_action(self, action: AgentAction) -> None:
        """Add an action to the batch."""
        self.actions.append(action)

    @property
    def pending_actions(self) -> list[AgentAction]:
        """Get actions still pending confirmation."""
        return [a for a in self.actions if a.is_pending]

    @property
    def confirmed_actions(self) -> list[AgentAction]:
        """Get confirmed actions."""
        return [a for a in self.actions if a.status == ActionStatus.CONFIRMED]

    @property
    def all_confirmed(self) -> bool:
        """Check if all actions are confirmed."""
        return all(a.status == ActionStatus.CONFIRMED for a in self.actions)

    @property
    def has_dangerous_actions(self) -> bool:
        """Check if batch contains dangerous actions."""
        return any(a.is_dangerous for a in self.actions)

    def confirm_all(self) -> None:
        """Confirm all pending actions."""
        for action in self.actions:
            if action.is_pending:
                action.confirm()

    def reject_all(self) -> None:
        """Reject all pending actions."""
        for action in self.actions:
            if action.is_pending:
                action.reject()
