"""
MuJoCo Simulator for GCO World
"""
import numpy as np
import mujoco
from mujoco import viewer
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import torch
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull
import yaml
import os
from datetime import datetime

# GCO imports
from gco.world.robot import Robot
from gco.world.obstacles import Obstacle
from gco.world.objects import MovableObject, ObjectRectangle, ObjectCircle, ObjectT
from gco.utils.transform_utils import Transform, Translation2, Transform2, mujoco_quat_to_z_theta, z_theta_to_mujoco_theta
from gco.config import Config as cfg

# Try to import trimesh for convex decomposition
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Polygon objects will not be convexified.")
    print("Install with: pip install 'trimesh[vhacd]'")
    raise ImportError("trimesh not available. Polygon objects will not be convexified.")


class MuJoCoSimulator:
    """MuJoCo-based physics simulator for GCO World"""
    
    def __init__(self, 
                 world_size: Tuple[float, float],
                 timestep: float = 0.003,
                 gravity: Tuple[float, float, float] = (0, 0, -9.81),
                 seed: int = 42,
                 save_world_states: bool = False,
                 save_frequency: int = 50):
        """
        Initialize the simulator
        
        Args:
            world_size: Size of the world (width, height)
            timestep: Simulation timestep
            gravity: Gravity vector
            seed: Random seed for deterministic simulation (future use)
            save_world_states: Whether to save all world states to YAML file during simulation
            save_frequency: How often to save world states (every N steps, default: 1)
        """
        self.world_size = world_size
        self.timestep = timestep
        self.gravity = gravity
        self.seed = seed
        self.save_world_states = save_world_states
        self.save_frequency = save_frequency
        
        # These will be populated when the world is set
        self.robots = {}
        self.obstacles = {}
        self.objects = {}
        
        # World state saving variables
        self.world_states_data = None
        self.static_data_saved = False
        self.simulation_counter = 0
        self.global_timestep_counter = 0
        
        # MuJoCo model and data (initialized when needed)
        self.model = None
        self.data = None
        self.mjcf = None
        
        # Body ID mappings for efficient access
        self.robot_body_ids = {}
        self.object_body_ids = {}

        # Some constants for robots.
        self.robot_height = cfg.robot_height
        self.object_height = cfg.object_height
        
        # Track convex decomposition results
        self.convex_mesh_names = {}
        
    def set_robot_pose_init(self, name: str, pose: Transform):
        self.robot_poses_init[name] = pose
        self.set_world_state(robots=self.robots, robot_poses=self.robot_poses_init)
        
    def set_world_state(self, 
                       robots: Dict[str, Robot],
                       robot_poses: Dict[str, Transform],
                       obstacles: Dict[str, Obstacle] = None,
                       obstacle_poses: Dict[str, Transform] = None,
                       objects: Dict[str, MovableObject] = None,
                       object_poses: Dict[str, Transform] = None,
                       regenerate_model: bool = True):
        """
        Set the world state and optionally regenerate MuJoCo model
        
        Args:
            robots: Dictionary mapping robot names to Robot objects
            robot_poses: Dictionary mapping robot names to their poses
            obstacles: Dictionary mapping obstacle names to Obstacle objects
            obstacle_poses: Dictionary mapping obstacle names to their poses
            objects: Dictionary mapping object names to MovableObject objects
            object_poses: Dictionary mapping object names to their poses
            regenerate_model: Whether to regenerate the MuJoCo model (default: True)
        """
        self.robots = robots
        self.robot_poses = robot_poses
        self.obstacles = obstacles or {}
        self.obstacle_poses = obstacle_poses or {}
        self.objects = objects or {}
        self.object_poses = object_poses or {}
        
        # Clear convex mesh cache when world state changes
        self.convex_mesh_names = {}
        
        if regenerate_model:
            # Regenerate MuJoCo model
            self._generate_mjcf(robot_poses, object_poses)
            # print("\n===== MJCF GENERATED =====\n", self.mjcf, "\n========================\n")
            self.model = mujoco.MjModel.from_xml_string(self.mjcf)
            self.data = mujoco.MjData(self.model)
            
            # Update body ID mappings
            self._update_body_id_mappings()
        else:
            # Just update poses in existing model.
            self._update_poses_only(robot_poses, object_poses)
    
    def _update_poses_only(self, robot_poses: Dict[str, Transform], object_poses: Dict[str, Transform] = None):
        """
        Update poses in the existing MuJoCo model without regenerating it.
        This is used for persistent simulator mode to avoid invalidating the GUI.
        """
        if self.model is None or self.data is None:
            return
        
        # Update robot poses
        for robot_name, pose in robot_poses.items():
            if robot_name in self.robot_body_ids:
                body_id = self.robot_body_ids[robot_name]
                pos_2d = pose.get_t().cpu().numpy()
                # Set position (x, y, z)
                self.data.xpos[body_id] = [pos_2d[0], pos_2d[1], self.robot_height / 2.0 + 0.001]
                # Set orientation (quaternion)
                theta = pose.get_theta().cpu().numpy()
                quat = z_theta_to_mujoco_theta(theta)
                self.data.xquat[body_id] = quat
                
                # Also update mocap body if it exists
                mocap_name = f"mocap_{robot_name}"
                if hasattr(self, 'mocap_body_ids') and mocap_name in self.mocap_body_ids:
                    mocap_id = self.mocap_body_ids[mocap_name]
                    self.data.xpos[mocap_id] = [pos_2d[0], pos_2d[1], self.robot_height / 2.0 + 0.001]
                    self.data.xquat[mocap_id] = quat
        
        # Update object poses
        if object_poses:
            for object_name, pose in object_poses.items():
                if object_name in self.object_body_ids:
                    body_id = self.object_body_ids[object_name]
                    pos_2d = pose.get_t().cpu().numpy()
                    # Set position (x, y, z)
                    self.data.xpos[body_id] = [pos_2d[0], pos_2d[1], self.object_height / 2.0 + 0.001]
                    # Set orientation (quaternion)
                    theta = pose.get_theta().cpu().numpy()
                    quat = z_theta_to_mujoco_theta(theta)
                    self.data.xquat[body_id] = quat
    
    def _generate_mjcf(self, robot_poses: Dict[str, torch.Tensor], object_poses: Dict[str, Transform] = None) -> str:
        """Generate MJCF string with robots and objects positioned at given poses"""
        
        # Start MJCF
        mjcf = f"""
<mujoco>
  <option gravity="0 0 {self.gravity[2]}" timestep="{self.timestep}"/>
  
  <visual>
    <global azimuth="120" elevation="-20"/>
    <quality shadowsize="4096"/>
  </visual>
  
  <asset>
    <!-- Black skybox -->
    <texture name="skybox_black" type="skybox"
             builtin="flat"
             rgb1="0 0 0" rgb2="0 0 0"
             width="32" height="32"/>
    
    <material name="floor_mat" reflectance="0.2" shininess="0.3" specular="0.1"/>
"""
        # Add mesh assets for polygons and T-shapes
        for object_name, obj in self.objects.items():
            # Polygon mesh generation with convex decomposition
            if hasattr(obj, 'vertices') or isinstance(obj, ObjectT):
                vertices_2d = obj.vertices
                thickness = self.object_height

                # Vertices are in the object frame (X=forward, Y=left) and in meters.
                # We need to convert them to MuJoCo frame to match the observation generator.
                # Apply the same transformation as in observation_generator.py:
                # 1. Swap X and Y coordinates (X and Y swapped)
                # 2. Flip both X and Y axes
                vertices_2d_flipped = torch.stack([vertices_2d[:, 1], vertices_2d[:, 0]], dim=1)
                vertices_2d_flipped[:, 1] = -vertices_2d_flipped[:, 1]  # Y flipped
                vertices_2d_flipped[:, 0] = -vertices_2d_flipped[:, 0]  # X flipped
                vertices_2d = vertices_2d_flipped.cpu().numpy()
                # Get convex mesh names (this will handle the decomposition)
                # print(f"Creating convex meshes for {object_name} with |vertices|: {len(vertices_2d)}")
                mesh_names = self._create_convex_meshes_from_polygon(object_name, vertices_2d, thickness)
                
                # Generate MJCF for each convex piece
                if len(mesh_names) == 1 and mesh_names[0] == f"mesh_{object_name}":
                    # Single mesh (either already convex or fallback)
                    # Create the mesh using the original simple method
                    centroid = np.mean(vertices_2d, axis=0)
                    triangles = []
                    n_vertices = len(vertices_2d)
                    for i in range(n_vertices):
                        j = (i + 1) % n_vertices
                        triangles.append([i, j, n_vertices])
                    
                    vertices_2d = np.vstack([vertices_2d, centroid])
                    mesh_vertices = []
                    for z in [0.0, thickness]:
                        for v in vertices_2d:
                            mesh_vertices.append([v[0], v[1], z])
                    
                    mesh_faces = []
                    n = len(vertices_2d)
                    for t in triangles:
                        mesh_faces.append([t[0], t[1], t[2]])
                    for t in triangles:
                        mesh_faces.append([t[0]+n, t[1]+n, t[2]+n])
                    original_n = n_vertices
                    for i in range(original_n):
                        j = (i+1)%original_n
                        mesh_faces.append([i, j, j+n])
                        mesh_faces.append([i, j+n, i+n])
                    
                    mesh_vertex_str = ' '.join(f"{v[0]} {v[1]} {v[2]}" for v in mesh_vertices)
                    mesh_face_str = ' '.join(f"{f[0]} {f[1]} {f[2]}" for f in mesh_faces)
                    mjcf += f"""
    <mesh name="mesh_{object_name}" vertex="{mesh_vertex_str}" face="{mesh_face_str}"/>
"""
                else:
                    # Multiple convex pieces from VHACD (including single VHACD piece)
                    # The mesh data is already stored in self.convex_mesh_names
                    # We'll add the mesh definitions in a separate loop after this
                    pass
        
        # Add convex mesh definitions for VHACD-decomposed objects
        epsilon = 1e-5  # Small value to avoid coplanarity
        for object_name, mesh_names in self.convex_mesh_names.items():
            if len(mesh_names) >= 1 and any("_vhacd_" in name for name in mesh_names):  # VHACD-decomposed meshes (including single piece)
                obj = self.objects[object_name]
                if hasattr(obj, 'vertices') or isinstance(obj, ObjectT):
                    vertices_2d = obj.vertices

                    # Vertices are in the object frame and in meters.
                    # We need to convert them to the world frame.
                    vertices_2d_flipped = torch.stack([vertices_2d[:, 1], vertices_2d[:, 0]], dim=1)
                    vertices_2d_flipped[:, 1] = -vertices_2d_flipped[:, 1]
                    # vertices_2d_flipped[:, 0] = -vertices_2d_flipped[:, 0]
                    vertices_2d = vertices_2d_flipped.cpu().numpy()
                    thickness = self.object_height
                    
                    # Recreate the original mesh for decomposition
                    centroid = np.mean(vertices_2d, axis=0)
                    triangles = []
                    n_vertices = len(vertices_2d)
                    for i in range(n_vertices):
                        j = (i + 1) % n_vertices
                        triangles.append([i, j, n_vertices])
                    
                    vertices_2d_with_centroid = np.vstack([vertices_2d, centroid])
                    mesh_vertices = []
                    for z in [0.0 + epsilon, thickness + epsilon]:
                        for v in vertices_2d_with_centroid:
                            mesh_vertices.append([v[0], v[1], z])
                    
                    mesh_faces = []
                    n = len(vertices_2d_with_centroid)
                    for t in triangles:
                        mesh_faces.append([t[0], t[1], t[2]])
                    for t in triangles:
                        mesh_faces.append([t[0]+n, t[1]+n, t[2]+n])
                    original_n = n_vertices
                    for i in range(original_n):
                        j = (i+1)%original_n
                        mesh_faces.append([i, j, j+n])
                        mesh_faces.append([i, j+n, i+n])
                    
                    # Create trimesh and decompose
                    mesh_vertices = np.array(mesh_vertices)
                    mesh_faces = np.array(mesh_faces)
                    concave_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
                    
                    if not concave_mesh.is_convex:
                        parts = trimesh.decomposition.convex_decomposition(
                            concave_mesh,
                            maxConvexHulls=8,
                            maxNumVerticesPerCH=64,
                            minimumVolumePercentErrorAllowed=0.1
                        )
                        
                        # Add each convex piece to MJCF, and collect valid mesh names
                        valid_mesh_names = []
                        for i, part in enumerate(parts):
                            mesh_name = f"mesh_{object_name}_vhacd_{i}"
                            # Add epsilon to z-coordinates to avoid coplanarity
                            verts = part['vertices'].copy()
                            verts[:,2] += epsilon
                            faces = part['faces']
                            # Filter out degenerate pieces (zero or near-zero volume)
                            try:
                                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                                if mesh.volume < 1e-10:
                                    continue  # Skip degenerate piece
                            except Exception:
                                continue  # Skip if mesh creation fails
                            v_str = " ".join(f"{x:.9g}" for x in verts.flatten())
                            f_str = " ".join(str(i) for i in faces.flatten())
                            mjcf += f"""
    <mesh name="{mesh_name}" vertex="{v_str}" face="{f_str}"/>
"""
                            valid_mesh_names.append(mesh_name)
                        # Update mesh_names to only valid ones
                        self.convex_mesh_names[object_name] = valid_mesh_names
        
        mjcf += "  </asset>\n  \n  <worldbody>\n"
        # Floor geom (must be f-string for size)
        mjcf += f"    <!-- Floor -->\n    <geom type=\"plane\" size=\"{self.world_size[0]/2} {self.world_size[1]/2} 0.1\" rgba=\"0 0 0 1\" material=\"floor_mat\"/>\n"
        mjcf += "    \n    <!-- Lighting -->\n    <light name=\"key_light\" pos=\"2 2 3\" dir=\"0 0 -1\" directional=\"true\" \n           castshadow=\"true\" cutoff=\"45\" exponent=\"1\" \n           ambient=\"0.2 0.2 0.2\" diffuse=\"0.8 0.8 0.8\" specular=\"0.1 0.1 0.1\"/>\n    <light name=\"fill_light\" pos=\"-2 1 2\" dir=\"0 0 -1\" directional=\"true\" \n           castshadow=\"false\" cutoff=\"60\" exponent=\"1\" \n           ambient=\"0.1 0.1 0.1\" diffuse=\"0.4 0.4 0.4\" specular=\"0.05 0.05 0.05\"/>\n"        # Add robots (cylinders with mocap control) at given poses
        for i, (robot_name, robot) in enumerate(self.robots.items()):
            if robot_name in robot_poses:
                # Use given pose (this could be a torch tensor or a Transform object).
                if isinstance(robot_poses[robot_name], torch.Tensor):
                    x, y = robot_poses[robot_name].cpu().numpy()[:2]
                elif isinstance(robot_poses[robot_name], list):
                    x, y = robot_poses[robot_name][:2]
                else:
                    x, y = robot_poses[robot_name].get_t().cpu().numpy()
            else:
                # Fall back to world pose
                pose = self.robot_poses[robot_name]
                x, y = pose.get_t().cpu().numpy()
            
            z = self.robot_height / 2.0  # Slightly elevated to avoid ground friction
            mjcf += f"""
    <!-- Robot {robot_name} -->
    <body name="robot_{robot_name}" pos="{x} {y} {z}">
    <freejoint/>
    <geom type="cylinder" size="{robot.radius} {self.robot_height/2}" mass="2.2" rgba="0.2 0.2 0.8 1"/>
    <site name="site_robot_{robot_name}" pos="0 0 0" size="0.001" rgba="1 0 0 1"/>
    </body>

    <body name="mocap_{robot_name}" mocap="true" pos="{x} {y} {z}">
    <site name="site_mocap_{robot_name}" pos="0 0 0" size="0.001" rgba="0 1 0 1"/>
    </body>
"""
        
        # Add obstacles (fixed)
        for i, (obstacle_name, obstacle) in enumerate(self.obstacles.items()):
            pose = self.obstacle_poses[obstacle_name]
            x, y = pose.get_t().cpu().numpy()
            z = self.object_height / 2.0 + 0.001  # Slightly elevated to avoid ground friction
            theta = pose.get_theta().cpu().numpy().item()
            theta = theta * 180 / np.pi
            if obstacle.type == "circle":
                mjcf += f"""
    <!-- Obstacle {obstacle_name} -->
    <body name="obstacle_{obstacle_name}" pos="{x} {y} {z}">
      <geom type="cylinder" size="{obstacle.radius} {self.robot_height/2}" rgba="0.5 0.5 0.5 1"/>
    </body>
"""
            elif obstacle.type == "rectangle":
                mjcf += f"""
    <!-- Obstacle {obstacle_name} (Rectangle) -->
    <body name="obstacle_{obstacle_name}" pos="{x} {y} {z}" euler="{0} {0} {theta}">
        <geom type="box" size="{obstacle.height/2} {obstacle.width/2} {self.robot_height/2}" rgba="0.5 0.5 0.5 1"/>
    </body>
"""
            elif obstacle.type == "square":
                mjcf += f"""
    <!-- Obstacle {obstacle_name} (Square) -->
    <body name="obstacle_{obstacle_name}" pos="{x} {y} {z}" euler="{0} {0} {theta}">
        <geom type="box" size="{obstacle.width/2} {obstacle.width/2} {self.robot_height/2}" rgba="0.5 0.5 0.5 1"/>
    </body>
"""
            else:
                raise ValueError(f"Obstacle {obstacle_name} type not supported")
                
        # Add movable objects
        for i, (object_name, obj) in enumerate(self.objects.items()):
            if object_poses is not None and object_name in object_poses:
                # Use given pose.
                if isinstance(object_poses[object_name], torch.Tensor):
                    x, y = object_poses[object_name].cpu().numpy()[:2]
                    theta = z_theta_to_mujoco_theta(object_poses[object_name][2])
                else:
                    x, y = object_poses[object_name].get_t().cpu().numpy()
                    theta = z_theta_to_mujoco_theta(object_poses[object_name].get_theta().cpu().numpy().item())
            else:
                # Fall back to world pose
                pose = self.object_poses[object_name]
                x, y = pose.get_t().cpu().numpy()
                theta = z_theta_to_mujoco_theta(pose.get_theta().cpu().numpy().item())
            
            theta = theta * 180 / np.pi
            z = self.object_height / 2.0 + 0.001  # Slightly elevated to avoid ground friction
            
            # Handle different object types
            if isinstance(obj, ObjectRectangle):  # Rectangle
                mjcf += f"""
    <!-- Object {object_name} (Rectangle) -->
    <body name="object_{object_name}" pos="{x} {y} {z}" euler="{0} {0} {theta}">
      <freejoint/>
      <geom type="box" size="{obj.width/2} {obj.height/2} {self.object_height/2}" mass="1.0" 
            rgba="0.8 0.3 0.3 1"/>
    </body> 
"""
            elif isinstance(obj, ObjectCircle):  # Circle
                mjcf += f"""
    <!-- Object {object_name} (Circle) -->
    <body name="object_{object_name}" pos="{x} {y} {z}" euler="{0} {0} {theta}">
      <freejoint/>
      <geom type="cylinder" size="{obj.radius} {self.object_height/2}" mass="1.0" 
            rgba="0.8 0.3 0.3 1"/>
    </body>
"""
#             elif isinstance(obj, ObjectT):
#                 mjcf += f"""
#     <!-- Object {object_name} (T-shape) -->
#     <body name="object_{object_name}" pos="{x} {y} {z}" euler="{0} {0} {theta}">
#       <freejoint/>
#       <geom type="mesh" mesh="mesh_{object_name}" mass="0.2" 
#             rgba="0.8 0.3 0.3 1"/>
#     </body>
# """
            elif hasattr(obj, 'vertices') or isinstance(obj, ObjectT):  # Polygon. Notice that the center z is at the bottom.
                # Check if this object has convex decomposition
                if object_name in self.convex_mesh_names and len(self.convex_mesh_names[object_name]) > 1:
                    # Multiple convex pieces
                    mjcf += f"""
    <!-- Object {object_name} (Polygon - Convex Decomposed) -->
    <body name="object_{object_name}" pos="{x} {y} {0.001}" euler="{0} {0} {theta}">
      <freejoint/>"""
                    # Add a geom for each convex piece
                    for mesh_name in self.convex_mesh_names[object_name]:
                        mjcf += f"""
      <geom type="mesh" mesh="{mesh_name}" mass="{0.2/len(self.convex_mesh_names[object_name])}" 
            rgba="0.8 0.3 0.3 1"/>"""
                    mjcf += f"""
    </body>
"""
                elif object_name in self.convex_mesh_names and len(self.convex_mesh_names[object_name]) == 1 and "_vhacd_" in self.convex_mesh_names[object_name][0]:
                    # Single VHACD piece
                    mesh_name = self.convex_mesh_names[object_name][0]
                    mjcf += f"""
    <!-- Object {object_name} (Polygon - Single VHACD Piece) -->
    <body name="object_{object_name}" pos="{x} {y} {0.001}" euler="{0} {0} {theta}">
      <freejoint/>
      <geom type="mesh" mesh="{mesh_name}" mass="0.2" 
            rgba="0.8 0.3 0.3 1"/>
    </body>
"""
                else:
                    # Single mesh (either already convex or fallback)
                    mjcf += f"""
    <!-- Object {object_name} (Polygon) -->
    <body name="object_{object_name}" pos="{x} {y} {z}" euler="{0} {0} {theta}">
      <freejoint/>
      <geom type="mesh" mesh="mesh_{object_name}" mass="0.2" 
            rgba="0.8 0.3 0.3 1"/>
    </body>
"""
            else:
                raise ValueError(f"Object {object_name} type not supported")
            
        
        # Add soft connect constraints instead of rigid welds
        mjcf += "\n  </worldbody>\n\n  <equality>\n"
        for robot_name in self.robots.keys():
            mjcf += f'    <connect name="connect_{robot_name}" site1="site_mocap_{robot_name}" site2="site_robot_{robot_name}" solref="0.02 0.8" solimp="0.05 0.05 0.001"/>\n'
        mjcf += "  </equality>\n</mujoco>"
        
        self.mjcf = mjcf
        return mjcf
    
    def _update_body_id_mappings(self):
        """Update body ID mappings for efficient access"""
        self.robot_body_ids = {}
        self.object_body_ids = {}
        self.mocap_body_ids = {}
        
        for robot_name in self.robots.keys():
            body_view = self.model.body(f"robot_{robot_name}")
            self.robot_body_ids[robot_name] = body_view.id
            
            # Also track mocap body
            try:
                mocap_body_view = self.model.body(f"mocap_{robot_name}")
                self.mocap_body_ids[f"mocap_{robot_name}"] = mocap_body_view.id
            except:
                pass  # Mocap body might not exist
        
        for object_name in self.objects.keys():
            body_view = self.model.body(f"object_{object_name}")
            self.object_body_ids[object_name] = body_view.id
    
    def _reset_objects_to_world_poses(self):
        """Reset objects and obstacles to their original world poses"""
        # Reset obstacles to their original poses
        for obstacle_name, pose in self.obstacle_poses.items():
            try:
                body_view = self.model.body(f"obstacle_{obstacle_name}")
                body_id = body_view.id
                x, y = pose.get_t().cpu().numpy()
                z = self.robot_height / 2.0 + 0.001
                self.data.xpos[body_id] = [x, y, z]
            except:
                pass  # Skip if obstacle not found
        
        # Reset objects to their original poses
        for object_name, pose in self.object_poses.items():
            try:
                body_view = self.model.body(f"object_{object_name}")
                body_id = body_view.id
                x, y = pose.get_t().cpu().numpy()
                z = 0.02  # Fixed height for objects
                self.data.xpos[body_id] = [x, y, z]
                
                # Also reset orientation
                theta = pose.get_theta().cpu().numpy().item()
                # Convert to quaternion (assuming rotation around z-axis)
                quat = [0, 0, np.sin(theta/2), np.cos(theta/2)]
                self.data.xquat[body_id] = quat
            except:
                pass  # Skip if object not found
    
    def simulate_trajectories(self, 
                            robot_trajectories: Dict[str, torch.Tensor],
                            max_steps: int = None,
                            record_frequency: int = 1,
                            visualize: bool = False,
                            real_time: bool = True,
                            gui = None) -> Dict[str, Any]:
        """
        Simulate robot trajectories and return resulting world states
        
        Args:
            robot_trajectories: Dict mapping robot names to trajectories [H, 2] (x, y positions)
            max_steps: Maximum simulation steps (if None, uses trajectory length)
            record_frequency: How often to record positions (every N steps)
            visualize: Whether to visualize the simulation in real-time
            real_time: Whether to run visualization in real-time or as fast as possible
            
        Returns:
            Dict containing recorded robot and object trajectories as Transform objects
        """
        if self.model is None or self.data is None:
            raise ValueError("Simulator not initialized. Call set_world_state first.")
        
        # Extract first poses from trajectories
        robot_start_poses = {}
        for robot_name, traj in robot_trajectories.items():
            if len(traj) > 0:
                robot_start_poses[robot_name] = traj[0]
        
        # Regenerate model with given poses
        self._generate_mjcf(robot_start_poses, object_poses=None)
        self.model = mujoco.MjModel.from_xml_string(self.mjcf)
        self.data = mujoco.MjData(self.model)
        self._update_body_id_mappings()
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset objects and obstacles to their original world poses
        self._reset_objects_to_world_poses()
        
        # World state saving initialization happens automatically in _save_static_world_data()
        
        # Generate simulation ID for this run
        simulation_id = f"sim_{self.simulation_counter:04d}"
        self.simulation_counter += 1
        
        # Determine trajectory length
        traj_lengths = [len(traj) for traj in robot_trajectories.values()]
        if not traj_lengths:
            raise ValueError("No robot trajectories provided")
        
        max_traj_length = max(traj_lengths)
        if max_steps is None:
            max_steps = max_traj_length
        
        # Initialize recording arrays
        num_recorded_steps = max_steps // record_frequency
        
        # Create robot name to index mapping for mocap
        robot_names = list(self.robots.keys())
        robot_name_to_idx = {name: i for i, name in enumerate(robot_names)}
        
        # Simulation loop
        recorded_step = 0
        world_states = []
        
        # Setup visualization if requested
        external_gui = gui is not None
        if visualize and gui is None:
            gui = viewer.launch_passive(self.model, self.data)
            gui.cam.distance = 3.5
            # Track the first robot.
            # gui.cam.trackbodyid = self.robot_body_ids[robot_names[1]]
            # gui.cam.lookat = [0.0, 0.0, 0.0]
            # gui.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        
        try:
            # Run through the trajectory and then hold the last position for 20% more steps.
            for step_raw in range(max_steps + int(max_steps * 0.2)):
                step = min(step_raw, max_steps - 1)
                # Update mocap positions for robots
                for robot_name, traj in robot_trajectories.items():
                    if robot_name in robot_name_to_idx:
                        idx = robot_name_to_idx[robot_name]
                        if step < len(traj):
                            x, y = traj[step].cpu().numpy()
                            self.data.mocap_pos[idx] = [x, y, self.robot_height / 2.0 + 0.001]
                        else:
                            # Hold last position
                            x, y = traj[-1].cpu().numpy()
                            self.data.mocap_pos[idx] = [x, y, self.robot_height / 2.0 + 0.001]
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Save world state if enabled
                if self.save_world_states:
                    self._save_world_state_timestep(step, simulation_id)
                
                # Update visualization if enabled
                if visualize and gui is not None:
                    if not gui.is_running():
                        break
                    gui.sync()
                    if real_time:
                        time.sleep(self.timestep * 5)
                
                # Record positions
                if step % record_frequency == 0 and recorded_step < num_recorded_steps:
                    # Create world state for this step
                    robot_poses = {}
                    object_poses = {}
                    
                    # Record robot positions
                    for robot_name in robot_names:
                        body_id = self.robot_body_ids[robot_name]
                        pos_3d = self.data.xpos[body_id]
                        robot_poses[robot_name] = Translation2(t=torch.tensor(pos_3d[:2], device=cfg.device, dtype=torch.float32))
                    
                    # Record object positions
                    for object_name in self.objects.keys():
                        body_id = self.object_body_ids[object_name]
                        pos_3d = self.data.xpos[body_id]
                        quat = self.data.xquat[body_id]
                        quat = torch.tensor(quat, device=cfg.device, dtype=torch.float32)
                        theta = mujoco_quat_to_z_theta(quat)
                        pos_3d = torch.tensor(pos_3d, device=cfg.device, dtype=torch.float32)
                        object_poses[object_name] = Transform2(pos_3d[:2], theta.unsqueeze(0))
                    
                    world_states.append({
                        "robots": robot_poses,
                        "objects": object_poses
                    })
                    
                    recorded_step += 1
        finally:
            # Clean up visualization only if we created it
            if visualize and gui is not None and not external_gui:
                gui.close()
        
        return {
            'world_states': world_states,
            'simulation_time': max_steps * self.timestep,
            'recorded_steps': recorded_step
        }
    
    def simulate_trajectories_persistent(self, 
                                       robot_trajectories: Dict[str, torch.Tensor],
                                       max_steps: int = None,
                                       record_frequency: int = 1,
                                       visualize: bool = False,
                                       real_time: bool = True,
                                       gui = None) -> Dict[str, Any]:
        """
        Simulate robot trajectories using the existing model (for persistent simulator mode)
        
        Args:
            robot_trajectories: Dict mapping robot names to trajectories [H, 2] (x, y positions)
            max_steps: Maximum simulation steps (if None, uses trajectory length)
            record_frequency: How often to record positions (every N steps)
            visualize: Whether to visualize the simulation in real-time
            real_time: Whether to run visualization in real-time or as fast as possible
            gui: External GUI instance (optional)
            
        Returns:
            Dict containing recorded robot and object trajectories as Transform objects
        """
        if self.model is None or self.data is None:
            raise ValueError("Simulator not initialized. Call set_world_state first.")
        

        # Reset simulation data (but keep the model)
        # mujoco.mj_resetData(self.model, self.data)

        
        # Reset objects and obstacles to their original world poses
        self._reset_objects_to_world_poses()
        
        # World state saving initialization happens automatically in _save_static_world_data()
        
        # Generate simulation ID for this run
        simulation_id = f"sim_{self.simulation_counter:04d}"
        self.simulation_counter += 1
        
        # Determine trajectory length
        traj_lengths = [len(traj) for traj in robot_trajectories.values()]
        if not traj_lengths:
            raise ValueError("No robot trajectories provided")
        
        max_traj_length = max(traj_lengths)
        if max_steps is None:
            max_steps = max_traj_length
        
        # Initialize recording arrays
        num_recorded_steps = max_steps // record_frequency
        
        # Create robot name to index mapping for mocap
        robot_names = list(self.robots.keys())
        robot_name_to_idx = {name: i for i, name in enumerate(robot_names)}
        
        # Simulation loop
        recorded_step = 0
        world_states = []
        
        # Setup visualization if requested
        external_gui = gui is not None
        if visualize and gui is None:
            gui = viewer.launch_passive(self.model, self.data)
            gui.cam.distance = 3.5

        try:
            # Run through the trajectory and then hold the last position for 20% more steps.
            for step_raw in range(max_steps + int(max_steps * 0.2)):
                step = min(step_raw, max_steps - 1)
                # Update mocap positions for robots
                for robot_name, traj in robot_trajectories.items():
                    if robot_name in robot_name_to_idx:
                        idx = robot_name_to_idx[robot_name]
                        if step < len(traj):
                            x, y = traj[step].cpu().numpy()
                            self.data.mocap_pos[idx] = [x, y, self.robot_height / 2.0 + 0.001]
                        else:
                            # Hold last position
                            x, y = traj[-1].cpu().numpy()
                            self.data.mocap_pos[idx] = [x, y, self.robot_height / 2.0 + 0.001]
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Save world state if enabled
                if self.save_world_states:
                    self._save_world_state_timestep(step, simulation_id)
                
                # Update visualization if enabled
                if visualize and gui is not None:
                    if not gui.is_running():
                        break
                    gui.sync()
                    if real_time:
                        time.sleep(self.timestep * 5)
                
                # Record positions
                if step % record_frequency == 0 and recorded_step < num_recorded_steps:
                    # Create world state for this step
                    robot_poses = {}
                    object_poses = {}
                    
                    # Record robot positions
                    for robot_name in robot_names:
                        body_id = self.robot_body_ids[robot_name]
                        pos_3d = self.data.xpos[body_id]
                        robot_poses[robot_name] = Translation2(t=torch.tensor(pos_3d[:2], device=cfg.device, dtype=torch.float32))
                    
                    # Record object positions
                    for object_name in self.objects.keys():
                        body_id = self.object_body_ids[object_name]
                        pos_3d = self.data.xpos[body_id]
                        quat = self.data.xquat[body_id]
                        quat = torch.tensor(quat, device=cfg.device, dtype=torch.float32)
                        theta = mujoco_quat_to_z_theta(quat)
                        pos_3d = torch.tensor(pos_3d, device=cfg.device, dtype=torch.float32)
                        object_poses[object_name] = Transform2(pos_3d[:2], theta.unsqueeze(0))
                    
                    world_states.append({
                        "robots": robot_poses,
                        "objects": object_poses
                    })
                    
                    recorded_step += 1
        finally:
            # Clean up visualization only if we created it
            if visualize and gui is not None and not external_gui:
                gui.close()
        
        return {
            'world_states': world_states,
            'simulation_time': max_steps * self.timestep,
            'recorded_steps': recorded_step
        }
    
    def _create_convex_meshes_from_polygon(self, object_name: str, vertices_2d: np.ndarray, thickness: float) -> List[str]:
        """
        Create convex meshes from a 2D polygon using VHACD decomposition
        
        Args:
            object_name: Name of the object
            vertices_2d: 2D vertices of the polygon
            thickness: Thickness of the 3D mesh
            
        Returns:
            List of mesh names for the convex pieces
        """
        if not TRIMESH_AVAILABLE:
            # Fallback to simple triangulation without convex decomposition
            return self._create_simple_mesh_from_polygon(object_name, vertices_2d, thickness)
        
        try:
            # Create a simple triangulation first
            centroid = np.mean(vertices_2d, axis=0)
            triangles = []
            n_vertices = len(vertices_2d)
            for i in range(n_vertices):
                j = (i + 1) % n_vertices
                triangles.append([i, j, n_vertices])
            
            # Add centroid to vertices
            vertices_2d_with_centroid = np.vstack([vertices_2d, centroid])
            
            # Create 3D mesh
            mesh_vertices = []
            for z in [0.0, thickness]:
                for v in vertices_2d_with_centroid:
                    mesh_vertices.append([v[0], v[1], z])
            
            mesh_faces = []
            n = len(vertices_2d_with_centroid)
            
            # Bottom faces (z=0)
            for t in triangles:
                mesh_faces.append([t[0], t[1], t[2]])
            
            # Top faces (z=thickness)
            for t in triangles:
                mesh_faces.append([t[0]+n, t[1]+n, t[2]+n])
            
            # Side faces (quads split into two triangles)
            original_n = n_vertices
            for i in range(original_n):
                j = (i+1)%original_n
                mesh_faces.append([i, j, j+n])
                mesh_faces.append([i, j+n, i+n])
            
            # Create trimesh object
            mesh_vertices = np.array(mesh_vertices)
            mesh_faces = np.array(mesh_faces)
            concave_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
            
            # Check if mesh is already convex
            if concave_mesh.is_convex:
                # Already convex, return single mesh
                mesh_names = [f"mesh_{object_name}"]
                self.convex_mesh_names[object_name] = mesh_names
                return mesh_names
            
            # Run VHACD convex decomposition
            parts = trimesh.decomposition.convex_decomposition(
                concave_mesh,
                maxConvexHulls=8,
                maxNumVerticesPerCH=8,
                minimumVolumePercentErrorAllowed=0.5
            )
            
            # print(f"VHACD produced {len(parts)} convex pieces for {object_name}")
            
            # Create mesh names for each convex piece
            mesh_names = []
            for i, part in enumerate(parts):
                mesh_name = f"mesh_{object_name}_vhacd_{i}"
                mesh_names.append(mesh_name)
            
            # Store the mesh names for later use in geom creation
            self.convex_mesh_names[object_name] = mesh_names
            
            return mesh_names
            
        except Exception as e:
            print(f"Warning: Convex decomposition failed for {object_name}: {e}")
            print("Falling back to simple triangulation")
            return self._create_simple_mesh_from_polygon(object_name, vertices_2d, thickness)
    
    def _create_simple_mesh_from_polygon(self, object_name: str, vertices_2d: np.ndarray, thickness: float) -> List[str]:
        """
        Create a simple triangulated mesh from a 2D polygon (fallback method)
        
        Args:
            object_name: Name of the object
            vertices_2d: 2D vertices of the polygon
            thickness: Thickness of the 3D mesh
            
        Returns:
            List with single mesh name
        """
        # For simple polygons, create a fan triangulation from the centroid
        centroid = np.mean(vertices_2d, axis=0)
        
        # Create triangles from centroid to each edge
        triangles = []
        n_vertices = len(vertices_2d)
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            triangles.append([i, j, n_vertices])
        
        # Add centroid to vertices
        vertices_2d = np.vstack([vertices_2d, centroid])
        
        mesh_vertices = []
        for z in [0.0, thickness]:
            for v in vertices_2d:
                mesh_vertices.append([v[0], v[1], z])
        
        mesh_faces = []
        n = len(vertices_2d)
        
        # Bottom faces (z=0)
        for t in triangles:
            mesh_faces.append([t[0], t[1], t[2]])
        
        # Top faces (z=thickness)
        for t in triangles:
            mesh_faces.append([t[0]+n, t[1]+n, t[2]+n])
        
        # Side faces (quads split into two triangles)
        original_n = n_vertices
        for i in range(original_n):
            j = (i+1)%original_n
            mesh_faces.append([i, j, j+n])
            mesh_faces.append([i, j+n, i+n])
        
        mesh_names = [f"mesh_{object_name}"]
        self.convex_mesh_names[object_name] = mesh_names
        return mesh_names

    def _save_static_world_data(self):
        """Save static world data (obstacles, object shapes/sizes) to YAML file"""
        if not self.save_world_states or self.static_data_saved:
            return
        
        # Initialize world states data structure
        self.world_states_data = {
            'metadata': {
                'world_size': list(self.world_size),
                'timestep': self.timestep,
                'gravity': list(self.gravity),
                'timestamp': datetime.now().isoformat()
            },
            'static_data': {
                'obstacles': {},
                'object_shapes': {}
            },
            'dynamic_data': {
                'timesteps': []
            }
        }
        
        # Save obstacle configurations
        for obstacle_name, obstacle in self.obstacles.items():
            pose = self.obstacle_poses[obstacle_name]
            x, y = pose.get_t().cpu().numpy()
            theta = pose.get_theta().cpu().numpy().item()
            
            obstacle_data = {
                'type': obstacle.type,
                'position': [float(x), float(y)],
                'orientation': float(theta)
            }
            
            if obstacle.type == "circle":
                obstacle_data['radius'] = float(obstacle.radius)
            elif obstacle.type in ["rectangle", "square"]:
                obstacle_data['width'] = float(obstacle.width)
                if obstacle.type == "rectangle":
                    obstacle_data['height'] = float(obstacle.height)
            
            self.world_states_data['static_data']['obstacles'][obstacle_name] = obstacle_data
        
        # Save object shapes and sizes
        for object_name, obj in self.objects.items():
            object_data = {
                'type': type(obj).__name__
            }
            
            if isinstance(obj, ObjectRectangle):
                object_data['width'] = float(obj.width)
                object_data['height'] = float(obj.height)
            elif isinstance(obj, ObjectCircle):
                object_data['radius'] = float(obj.radius)
            elif hasattr(obj, 'vertices') or isinstance(obj, ObjectT):
                # For polygon objects, save vertices
                vertices_2d = obj.vertices.cpu().numpy()
                object_data['vertices'] = [[float(v[0]), float(v[1])] for v in vertices_2d]
            
            self.world_states_data['static_data']['object_shapes'][object_name] = object_data
        
        self.static_data_saved = True
    
    def _save_world_state_timestep(self, timestep: int, simulation_id: str = None):
        """Save current world state for a specific timestep (respects save_frequency)"""
        if not self.save_world_states or self.data is None:
            return
        
        # Always increment global counter (counts all simulator steps)
        current_global_step = self.global_timestep_counter
        self.global_timestep_counter += 1
        
        # Only save every save_frequency steps
        if current_global_step % self.save_frequency != 0:
            return
        
        # Ensure static data is saved first
        if not self.static_data_saved:
            self._save_static_world_data()
        
        timestep_data = {
            'timestep': current_global_step,
            'simulation_id': simulation_id,
            'robots': {},
            'objects': {}
        }
        
        # Save robot configurations
        for robot_name in self.robots.keys():
            if robot_name in self.robot_body_ids:
                body_id = self.robot_body_ids[robot_name]
                pos_3d = self.data.xpos[body_id]
                quat = self.data.xquat[body_id]
                
                robot_data = {
                    'position': [float(pos_3d[0]), float(pos_3d[1]), float(pos_3d[2])],
                    'orientation': [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                    'radius': float(self.robots[robot_name].radius)
                }
                timestep_data['robots'][robot_name] = robot_data
        
        # Save object configurations
        for object_name in self.objects.keys():
            if object_name in self.object_body_ids:
                body_id = self.object_body_ids[object_name]
                pos_3d = self.data.xpos[body_id]
                quat = self.data.xquat[body_id]
                
                object_data = {
                    'position': [float(pos_3d[0]), float(pos_3d[1]), float(pos_3d[2])],
                    'orientation': [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                }
                timestep_data['objects'][object_name] = object_data
        
        self.world_states_data['dynamic_data']['timesteps'].append(timestep_data)
    
    def save_world_states_to_file(self, filename: str = None):
        """Save accumulated world states data to YAML file"""
        if not self.save_world_states or self.world_states_data is None:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"world_states_{timestamp}.yaml"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        with open(filename, 'w') as f:
            yaml.dump(self.world_states_data, f, default_flow_style=False, indent=2)
        
        print(f"World states saved to {filename}")
    
    def teardown(self, filename: str = None):
        """Teardown the simulator and save all accumulated world states to file"""
        if self.save_world_states and self.world_states_data is not None:
            self.save_world_states_to_file(filename)
            print(f"Simulator teardown complete. Total simulations recorded: {self.simulation_counter}")
    
    def __del__(self):
        """Destructor to automatically save world states on simulator destruction"""
        try:
            self.teardown()
        except:
            pass  # Ignore errors during destruction

    ####################
    # Getters.
    ####################
    def get_robot_pose(self, name: str) -> Transform:
        return Translation2(t=torch.tensor(self.data.xpos[self.robot_body_ids[name]][:2], device=cfg.device, dtype=torch.float32))

    def get_object_pose(self, name: str) -> Transform:
        pos = self.data.xpos[self.object_body_ids[name]][:2]
        quat = self.data.xquat[self.object_body_ids[name]]
        quat = torch.tensor(quat, device=cfg.device, dtype=torch.float32)
        theta = mujoco_quat_to_z_theta(quat)
        return Transform2(t=torch.tensor(pos, device=cfg.device, dtype=torch.float32), theta=theta)
