"""
Agent class for traffic participants
"""

class Agent:
    """Base class for all traffic participants"""
    def __init__(self, agent_id):
        """
        Initialize an agent

        Args:
            agent_id: Unique identifier for the agent
        """
        self.id = agent_id
        self.agent_type = None
        self.agent_class = None
        
        # Behavior information
        self.cross_type = None
        self.signal_violation = None
        self.retrograde_type = None
        self.cardinal_direction = None

    @classmethod
    def from_dict(cls, agent_dict):
        """
        Create an Agent instance from a dictionary

        Args:
            agent_dict: Dictionary containing agent information

        Returns:
            An instance of Agent initialized with the provided data
        """
        agent = cls(agent_dict["id"])
        agent.agent_type = agent_dict.get("agent_type")
        agent.agent_class = agent_dict.get("agent_class")
        agent.cross_type = agent_dict.get("cross_type", None)
        agent.signal_violation = agent_dict.get("signal_violation", None)
        agent.retrograde_type = agent_dict.get("retrograde_type", None)
        agent.cardinal_direction = agent_dict.get("cardinal_direction", None)
        return agent

    @classmethod
    def from_tuple(cls, agent_id, agent_info):
        agent = cls(agent_id)
        agent.agent_type = agent_info[0]
        agent.agent_class = agent_info[1]
        agent.cross_type = agent_info[2]
        agent.signal_violation = agent_info[3]
        agent.retrograde_type = agent_info[4]
        agent.cardinal_direction = agent_info[5]
        return agent

    def to_dict(self):
        """Convert agent object to dictionary for JSON serialization without including None values"""
        agent_dict = {
            "id": self.id,
            "agent_type": self.agent_type,
            "agent_class": self.agent_class,
        }
        
        if self.cross_type is not None:
            agent_dict["cross_type"] = self.cross_type.tolist()
        if self.signal_violation is not None:
            agent_dict["signal_violation"] = self.signal_violation.tolist()
        if self.retrograde_type is not None:
            agent_dict["retrograde_type"] = self.retrograde_type
        if self.cardinal_direction is not None:
            agent_dict["cardinal_direction"] = self.cardinal_direction

        return agent_dict
        