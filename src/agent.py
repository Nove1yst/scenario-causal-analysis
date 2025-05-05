"""
Agent classes for traffic participants
"""
import numpy as np
from enum import Enum

class AgentType(Enum):
    """Agent type enumeration"""
    UNKNOWN = "unknown"
    PEDESTRIAN = "ped"
    MOTOR_VEHICLE = "mv"
    NON_MOTOR_VEHICLE = "nmv"

class MotorVehicleClass(Enum):
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"

class NonMotorVehicleClass(Enum):
    BICYCLE = "bicycle"
    TRICYCLE = "tricycle"
    MOTORCYCLE = "motorcycle"

class PedestrianClass(Enum):
    PEDESTRIAN = "pedestrian"

# TODO: determine the cross types
class CrossType(Enum):
    """Crossing behavior type enumeration"""
    NORMAL = "Normal"
    AGGRESSIVE = "Aggressive crossing"
    VIOLATION = "Violation crossing"
    ZIGZAG = "Zigzag crossing"

class SignalViolation(Enum):
    """Signal violation type enumeration"""
    NONE = "No violation of traffic lights"
    RED_LIGHT = "Red light running"
    STOP_SIGN = "Stop sign violation"

class RetrogradeType(Enum):
    """Retrograde behavior type enumeration"""
    NORMAL = "normal"
    FRONT = "front_retrograde"
    REAR = "rear_retrograde"
    FULL = "full_retrograde"
    UNKNOWN = "unknown"

class Agent:
    """Base class for all traffic participants"""
    def __init__(self, agent_id, fragment_id, agent_info):
        """
        Initialize an agent

        Args:
            agent_id: Unique identifier for the agent
            fragment_id: ID of the scenario fragment
            agent_type: Type of the agent
        """
        self.id = agent_id
        self.fragment_id = fragment_id
        self.agent_type = agent_info[0]
        self.agent_class = agent_info[1]
        
        # Behavior information
        self.cross_type = agent_info[2]
        self.signal_violations = agent_info[3]
        self.retrograde_type = agent_info[4]
        self.cardinal_direction = agent_info[5]
        
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

# class Pedestrian(Agent):
#     """Class for pedestrian agents"""
#     def __init__(self, agent_id, fragment_id):
#         super().__init__(agent_id, fragment_id, AgentType.PEDESTRIAN)
        
        
#     # def add_crossing_behavior(self, speed, waiting_time):
#     #     """Add crossing behavior information"""
#     #     self.crossing_speed.append(speed)
#     #     self.waiting_time.append(waiting_time)

# class MotorVehicle(Agent):
#     """Class for motor vehicle agents"""
#     def __init__(self, agent_id, fragment_id):
#         super().__init__(agent_id, fragment_id, AgentType.MOTOR_VEHICLE)
#         self.vehicle_class = None  # e.g., car, bus, truck


# class NonMotorVehicle(Agent):
#     """Class for non-motor vehicle agents (e.g., bicycles)"""
#     def __init__(self, agent_id, fragment_id):
#         super().__init__(agent_id, fragment_id, AgentType.NON_MOTOR_VEHICLE)
