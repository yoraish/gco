# General imports.
import torch
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass
from abc import ABC

# Project imports.
from gco.world.robot import Robot
from gco.world.obstacles import Obstacle
from gco.world.objects import MovableObject
from gco.world.simulator import MuJoCoSimulator
from gco.utils.transform_utils import Transform, Translation2
from gco.utils.viz_utils import visualize_world_trajectory
from gco.config import Config as cfg
from gco.observations.observation_generator import ObservationGenerator

class World:
    simulator_gui = None
    def __init__(self, 
                 size: Tuple[float, float], 
                 resolution: float = 0.05, 
                 dt=0.003,
                 robots_with_pose_init: Dict[Robot, Transform] = None,
                 obstacles_with_pose_init: Dict[Obstacle, Transform] = None,
                 objects_with_pose_init: Dict[MovableObject, Transform] = None,
                 seed: int = 42,
                 launch_viewer: bool = True):
        self.size = size
        self.sdf_resolution = resolution
        # Observation generator.
        self.observation_generator = ObservationGenerator()
        # Dictionary to store name: robot objects.
        self.robots_d = {robot.name: robot for robot in robots_with_pose_init.keys()} if robots_with_pose_init is not None else {}
        # Dictionary to store name: obstacle objects.
        self.obstacles_d = {obstacle.name: obstacle for obstacle in obstacles_with_pose_init.keys()} if obstacles_with_pose_init is not None else {}
        # Dictionary to store name: movable objects.
        self.objects_d = {object.name: object for object in objects_with_pose_init.keys()} if objects_with_pose_init is not None else {}
        # The initial poses.
        self.robot_poses_init_d = {robot.name: robots_with_pose_init[robot] for robot in robots_with_pose_init.keys()} if robots_with_pose_init is not None else {}
        self.obstacle_poses_init_d = {obstacle.name: obstacles_with_pose_init[obstacle] for obstacle in obstacles_with_pose_init.keys()} if obstacles_with_pose_init is not None else {}
        self.object_poses_init_d = {object.name: objects_with_pose_init[object] for object in objects_with_pose_init.keys()} if objects_with_pose_init is not None else {}
        # Signed distance field.
        self.sdf: torch.Tensor = torch.zeros(int(size[0] / resolution), int(size[1] / resolution))
        
        self.simulator = None
        self._persistent_simulator_finalized = False
        self.launch_viewer = launch_viewer
        self.simulator = MuJoCoSimulator(world_size=size, timestep=dt, seed=seed, save_world_states=False)

        self.finalize_world()
    
    def finalize_world(self):
        """Initialize persistent simulator mode with GUI."""
        import platform
        
        if self.simulator is None or self._persistent_simulator_finalized:
            return
        
        # Set initial world state in simulator (with model regeneration)
        self.simulator.set_world_state(
            robots=self.robots_d,
            robot_poses=self.robot_poses_init_d,
            obstacles=self.obstacles_d,
            obstacle_poses=self.obstacle_poses_init_d,
            objects=self.objects_d,
            object_poses=self.object_poses_init_d,
            regenerate_model=True  # Always regenerate for initial setup
        )
        
        # Launch persistent viewer only if requested.
        if self.launch_viewer:
            try:
                from mujoco import viewer
                self.simulator_gui = viewer.launch_passive(
                    self.simulator.model, 
                    self.simulator.data
                )
                self.simulator_gui.cam.distance = 3.5
                # self.simulator_gui.cam.lookat = [0.0, 2.0, 0.0]
                # # Set the direction of the camera to be against the x axis.
                # # Reverse camera direction by changing azimuth by 180 degrees
                # self.simulator_gui.cam.azimuth = -90.0

            except Exception as e:
                print(f"Warning: Could not launch MuJoCo viewer: {e}")
                print("Running without visualization.")
                self.simulator_gui = None
        else:
            self.simulator_gui = None
        
        self._persistent_simulator_finalized = True
    
    def _cleanup_persistent_simulator(self):
        """Clean up the persistent simulator."""
        if self.simulator_gui is not None:
            try:
                self.simulator_gui.close()
            except:
                pass  # Ignore errors during cleanup
            self.simulator_gui = None
        
        # Reset the finalized flag so we can initialize again if needed
        self._persistent_simulator_finalized = False


    ####################
    # Simulation.
    ####################
    def apply_trajectories(self, traj_d: Dict[str, torch.Tensor], visualize: bool = False, real_time: bool = True) -> List[Dict[str, Dict[str, Transform]]]:
        """
        Forward simulate the trajectories and set the new poses at the robot and object poses in the world object.
        :param traj_d: a torch tensor per robot name. The tensor should be of shape (H, DoF=2)
        :param visualize: Whether to visualize the simulation in real-time (physics only)
        :param real_time: Whether to run visualization in real-time or as fast as possible (physics only)
        :return: a list of observed world states. Each state is a (1) dictionary of robot_name:  robot pose,
         and (2) dictionary of object_name: object pose.
        """
        # Apply trajectories using persistent simulator mode.
        # Only visualize if we have a GUI
        should_visualize = visualize and self.simulator_gui is not None
        
        # Execute trajectories without regenerating the model
        result = self.simulator.simulate_trajectories_persistent(
            traj_d, 
            visualize=should_visualize, 
            real_time=real_time,
            gui=self.simulator_gui
        )
        
        world_states = result['world_states']
        
        # Update poses in the world according to the current simulator state.
        
        return world_states

    def apply_trajectories_without_update(self, traj_d: Dict[str, torch.Tensor]):
        """
        Forward simulate the trajectories without setting the new poses in the world.
        :param traj_d: a torch tensor per robot name. The tensor should be of shape (H, DoF=2)
        :return: nothing
        """
        pass

    def cleanup_persistent_simulator(self):
        """Clean up the persistent simulator. Call this when done with the world."""
        self._cleanup_persistent_simulator()
    
    def __del__(self):
        """Destructor to ensure cleanup of persistent simulator."""
        self.cleanup_persistent_simulator()

    ####################
    # Getters.
    ####################
    def get_robot(self, name: str) -> Robot:
        return self.robots_d[name]

    def get_obstacle(self, name: str) -> Obstacle:
        return self.obstacles_d[name]

    def get_all_obstacles(self) -> Dict[str, Obstacle]:
        return self.obstacles_d

    def get_all_objects(self) -> Dict[str, Obstacle]:
        return self.objects_d

    def get_all_robots(self) -> Dict[str, Robot]:
        return self.robots_d

    def get_robot_pose(self, name: str) -> Transform:
        return self.simulator.get_robot_pose(name)

    def get_object(self, name: str) -> MovableObject:
        return self.objects_d[name]

    def get_object_pose(self, name: str) -> Transform:
        return self.simulator.get_object_pose(name)

    def get_all_robot_poses(self) -> Dict[str, Transform]:
        return {name: self.simulator.get_robot_pose(name) for name in self.robots_d.keys()}

    def get_all_object_poses(self) -> Dict[str, Transform]:
        return {name: self.simulator.get_object_pose(name) for name in self.objects_d.keys()}
    
    def get_obstacle_pose(self, name: str) -> Transform:
        return self.obstacle_poses_d[name]

    def get_all_obstacle_poses(self) -> Dict[str, Transform]:
        return self.obstacle_poses_init_d

    # def reset(self):
    #     """Reset the world to the initial state."""
    #     self.robot_poses_d = {}
    #     self.obstacle_poses_d = {}
    #     self.object_poses_d = {}
    #     self.robots_d = {}
    #     self.obstacles_d = {}
    #     self.objects_d = {}
    #     self.sdf = torch.zeros(int(self.size[0] / self.sdf_resolution), int(self.size[1] / self.sdf_resolution))

    def get_state(self) -> Dict[str, Dict[str, Transform]]: 
        """Get the current state of the world."""
        return {'robots': self.get_all_robot_poses(), 'objects': self.get_all_object_poses()}

    ####################
    # Collision checking.
    ####################
    def is_state_valid(self, robot_q_d: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Check if a robot is in collision with obstacles. Only passed robots are checked for robot-robot collisions.
        :param robot_q_d: a dictionary of robot name: joint positions (B, DoF)
        :return: a boolean tensor of shape (B,) where each element is True if the robot is in collision with an obstacle, False otherwise.
        """
        # Check for intersection between robots first.
        robot_positions = torch.stack([self.get_robot_pose(robot_name).get_t() for robot_name in robot_q_d.keys()])
        robot_radii = torch.stack([self.get_robot(robot_name).radius for robot_name in robot_q_d.keys()])
        robot_collisions = self._check_robot_robot_collisions(robot_positions, robot_radii)

        # Check for intersection between robots and obstacles.
        obstacle_positions = torch.stack([self.get_obstacle_pose(obstacle_name).get_t() for obstacle_name in self.obstacles_d.keys()])
        obstacle_radii = torch.stack([self.get_obstacle(obstacle_name).radius for obstacle_name in self.obstacles_d.keys()])
        obstacle_collisions = self._check_robot_obstacle_collisions(robot_positions, robot_radii, obstacle_positions, obstacle_radii)

        # Return the sum of the collisions.
        return robot_collisions | obstacle_collisions

    def _check_robot_robot_collisions(self, robot_positions: torch.Tensor, robot_radii: torch.Tensor) -> torch.Tensor:
        """
        Check for intersection between robots.
        :param robot_positions: a tensor of shape (B, 2) representing the positions of the robots.
        :param robot_radii: a tensor of shape (B,) representing the radii of the robots.
        :return: a boolean tensor of shape (B,) where each element is True if the robot is in collision with an obstacle, False otherwise.
        """
        # Compute pairwise distances between all robots
        n_robots = robot_positions.shape[0]
        
        # Create indices for all pairs
        idx_i, idx_j = torch.triu_indices(n_robots, n_robots, offset=1, device=cfg.device)
        
        # Calculate distances between all pairs
        dists = torch.norm(robot_positions[idx_i] - robot_positions[idx_j], dim=1)
        
        # Check which pairs are in collision
        collision_mask = dists <= (robot_radii[idx_i] + robot_radii[idx_j])
        
        # Initialize collision tensor
        robot_collisions = torch.zeros(n_robots, device=robot_positions.device)
        
        # Set collisions for both robots in each colliding pair
        robot_collisions.index_add_(0, idx_i[collision_mask], torch.ones_like(robot_collisions[idx_i[collision_mask]], device=cfg.device))
        robot_collisions.index_add_(0, idx_j[collision_mask], torch.ones_like(robot_collisions[idx_j[collision_mask]], device=cfg.device))
        
        return robot_collisions.bool()

    def _check_robot_obstacle_collisions(self, robot_positions: torch.Tensor, robot_radii: torch.Tensor, obstacle_positions: torch.Tensor, obstacle_radii: torch.Tensor) -> torch.Tensor:
        """
        Check for intersection between robots and obstacles.
        :param robot_positions: a tensor of shape (B, 2) representing the positions of the robots.
        :param robot_radii: a tensor of shape (B,) representing the radii of the robots.
        :param obstacle_positions: a tensor of shape (B, 2) representing the positions of the obstacles.
        :param obstacle_radii: a tensor of shape (B,) representing the radii of the obstacles.
        :return: a boolean tensor of shape (B,) where each element is True if the robot is in collision with an obstacle, False otherwise.
        """
        # Check for intersection between robots and obstacles.
        robot_obstacle_collisions = torch.zeros(robot_positions.shape[0], device=robot_positions.device)
        # for i in range(robot_positions.shape[0]):
        #     for j in range(obstacle_positions.shape[0]):
        #         if torch.norm(robot_positions[i] - obstacle_positions[j]) <= robot_radii[i] + obstacle_radii[j]:
        #             robot_obstacle_collisions[i] = 1

        # Check out of bounds.
        out_of_bounds = (robot_positions[:, 0] < -self.size[0] / 2) | (robot_positions[:, 0] > self.size[0] / 2) | (robot_positions[:, 1] < -self.size[1] / 2) | (robot_positions[:, 1] > self.size[1] / 2)
        robot_obstacle_collisions[out_of_bounds] = 1

        return robot_obstacle_collisions

    ####################
    # Distance fields.
    ####################
    def update_sdf(self):
        """
        Update the signed distance field of all obstacles in the world.
        :return: nothing
        """
        print("\033[91m", "update_sdf not implemented", "\033[0m")
        return None

    ####################
    # Observation generation.
    ####################
    def get_object_centric_observation(self, name: str, size: Tuple, visualize: bool = False):

        """
        Get the object centric observation of a specific object. This places the object at the origin with zero rotation
        and synthesizes (a) robot poses, (b) distance field of all obstacles, (c) distance field of object (to be used when traveling around it).
        :param name: the name of the object.
        :param size: the size of the observation (H, W).
        :return: a torch tensor of shape (size[0], size[1], 2) representing the object centric observation.
        """
        return self.observation_generator.get_object_centric_observation(self.objects_d[name], size, visualize)

    ####################
    # Visualization.
    ####################
    def visualize(self, world_states: List[Dict[str, Transform]], save_path: str):
        """
        Visualize the world states.
        :param world_states: a list of world states.
        :param save_path: the path to save the visualization.
        :return: nothing
        """
        # Call a visualization function with the world states, the obstacles, the objects for the obstacles and objects,the sdf, and the save path.
        visualize_world_trajectory(self,
                                   world_states,
                                   save_path)




