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
        agent.cross_type = agent_dict.get("cross_type")
        agent.signal_violation = agent_dict.get("signal_violation")
        agent.retrograde_type = agent_dict.get("retrograde_type")
        agent.cardinal_direction = agent_dict.get("cardinal_direction")
        return agent

    def set(self, agent_info):
        self.agent_type = agent_info[0]
        self.agent_class = agent_info[1]
        
        # Behavior information
        self.cross_type = agent_info[2]
        self.signal_violation = agent_info[3]
        self.retrograde_type = agent_info[4]
        self.cardinal_direction = agent_info[5]

    def to_dict(self):
        """Convert agent object to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "agent_class": self.agent_class,
            "cross_type": self.cross_type,
            "signal_violation": self.signal_violation,
            "retrograde_type": self.retrograde_type,
            "cardinal_direction": self.cardinal_direction
        }
        
        # Trajectory data
        # self.frames = []
        # self.dimensions = None  # length, width
        
        # Interaction information
        # self.anomaly_frames = []  # Frames where anomalies occur
        # self.critical_interactions = {}  # Dictionary to store critical interactions with other agents
        
    # def add_state(self, frame_id, timestamp, x, y, vx, vy, ax, ay, heading):
    #     """
    #     Add a state to the agent's trajectory

    #     Args:
    #         frame_id: Frame ID
    #         timestamp: Timestamp
    #         x, y: Position
    #         vx, vy: Velocity
    #         ax, ay: Acceleration
    #         heading: Heading angle
    #     """
    #     self.frames.append(frame_id)
    #     self.timestamps.append(timestamp)
    #     self.positions.append((x, y))
    #     self.velocities.append((vx, vy))
    #     self.accelerations.append((ax, ay))
    #     self.headings.append(heading)
        
    # def set_dimensions(self, length, width):
    #     """Set agent dimensions"""
    #     self.dimensions = (length, width)
        
    # def add_anomaly(self, frame_id):
    #     """Add an anomaly frame"""
    #     if frame_id not in self.anomaly_frames:
    #         self.anomaly_frames.append(frame_id)
            
    # def add_critical_interaction(self, other_id, ssm_type, critical_frames, distances):
    #     """
    #     Add a critical interaction with another agent

    #     Args:
    #         other_id: ID of the other agent
    #         ssm_type: Type of safety surrogate measure
    #         critical_frames: List of critical frames
    #         distances: List of distances at critical frames
    #     """
    #     if other_id not in self.critical_interactions:
    #         self.critical_interactions[other_id] = []
    #     self.critical_interactions[other_id].append({
    #         'ssm_type': ssm_type,
    #         'critical_frames': critical_frames,
    #         'distances': distances
    #     })
        
    # def get_state_at_frame(self, frame_id):
    #     """Get agent state at a specific frame"""
    #     if frame_id in self.frames:
    #         idx = self.frames.index(frame_id)
    #         return {
    #             'position': self.positions[idx],
    #             'velocity': self.velocities[idx],
    #             'acceleration': self.accelerations[idx],
    #             'heading': self.headings[idx]
    #         }
    #     return None
        
    # def get_trajectory_segment(self, start_frame, end_frame):
    #     """Get trajectory segment between start and end frames"""
    #     start_idx = self.frames.index(start_frame) if start_frame in self.frames else 0
    #     end_idx = self.frames.index(end_frame) if end_frame in self.frames else len(self.frames)
        
    #     return {
    #         'frames': self.frames[start_idx:end_idx],
    #         'timestamps': self.timestamps[start_idx:end_idx],
    #         'positions': self.positions[start_idx:end_idx],
    #         'velocities': self.velocities[start_idx:end_idx],
    #         'accelerations': self.accelerations[start_idx:end_idx],
    #         'headings': self.headings[start_idx:end_idx]
    #     }
