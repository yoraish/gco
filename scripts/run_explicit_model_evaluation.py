"""
Comprehensive Model Evaluation using Explicit Test Scenarios

This script runs a systematic evaluation of models using explicitly defined test scenarios
that cover all combinations of:
- Robot counts (1, 2, 3)
- Object types (rectangle, circle, triangle, t_shape)
- Obstacle landscapes (empty, simple, maze, slalom)
- Start/goal positions and orientations
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import random
sys.path.append(str(Path(__file__).parent))

# Seed everything.
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from model_evaluation_benchmark import (
    ModelEvaluator, TestScenario, ModelConfig, ContactTrajectoryModelWrapper 
)
from gco.config import Config as cfg


def create_comprehensive_explicit_scenarios():
    """Create comprehensive explicit test scenarios covering all combinations."""
    scenarios = []
    
            # Define explicit test configurations
    rectangle_size1 = {"width": 0.15, "height": 0.32}
    rectangle_size2 = {"width": 0.3, "height": 0.3}
    rectangle_size3 = {"width": 0.18, "height": 0.32}
    circle_size1 = {"radius": 0.15}
    circle_size2 = {"radius": 0.18}
    circle_size3 = {"radius": 0.2}
    circle_size4 = {"radius": 0.25} 
    circle_size5 = {"radius": 0.3} 
    t_shape_size1 = {"bar_width": 0.2, "bar_height": 0.15, "stem_width": 0.15, "stem_height": 0.3}
    t_shape_size2 = {"bar_width": 0.3, "bar_height": 0.2, "stem_width": 0.25, "stem_height": 0.35}
    t_shape_size3 = {"bar_width": 0.4, "bar_height": 0.2, "stem_width": 0.2, "stem_height": 0.35}

    test_configs = []
    tasks_to_run = [
                    #   "up_env_empty",
                      "up_env_easy", 
                    #   "upmulti_env_empty",
                    #   "upmulti_env_wall",
                    ]
    task_specifications = {}

    # ================================ 
    # EMPTY ENV (fwd, bwd, left, right)
    # ================================
    empty_env_problem_names = ["up", "down", "left", "right"]
    start_pos = (0.0, 1.0, 0.0)
    for problem_name in empty_env_problem_names:
        if problem_name == "up":
            goal_pos = (0.0, 2.0, 0.0)
        elif problem_name == "down":
            goal_pos = (0.0, 0.0, 0.0)
        elif problem_name == "left":
            goal_pos = (-1.0, 1.0, 0.0)
        elif problem_name == "right":
            goal_pos = (1.0, 1.0, 0.0)

        task_specifications[f"{problem_name}_env_empty"] = {
            "obstacle_layout_name": "empty",
            # Problems will be created by randomly choosing object types and sizes from these options.
            "object_types_and_sizes": {
                "rectangle": [rectangle_size2],
            },
            # The number of objects in the scenario. Will try all in this list.
            "num_objects": [1],
            # The start and goal configurations for the objects. These are ordered pairs and will be chosen from the list in order. There must be at least max(num_objects) configurations in the list.
            "cfg_obj_start_goal_l": [
                                     (start_pos, goal_pos)],
            # The number of robots in the scenario. Will try all in this list.
            "robot_counts": [1,2,3],
            # Where the robots begin.
            "robot_start_configurations": [
                (-0.5, 0.0, 0.0),      # Robot 1
                (0.0, 0.0, 0.0),      # Robot 2
                (0.5,  0.0, 0.0),      # Robot 3
            ],
            # Number of repetitions per problem configuration
            "num_reps_per_problem": 5
        }

    # ================================ 
    # EMPTY EASY (fwd, bwd, left, right)
    # ================================
    empty_env_problem_names = ["up", "down", "left", "right"]
    start_pos = (0.0, 1.0, 0.0)
    for problem_name in empty_env_problem_names:
        if problem_name == "up":
            goal_pos = (0.0, 3.0, 2.0)
            obstacle_layout_name = "easy_up"
        elif problem_name == "down":
            goal_pos = (0.0, -1.0, -2.0)
            obstacle_layout_name = "easy_down"
        elif problem_name == "left":
            goal_pos = (-2.0, 1.0, 2.0)
            obstacle_layout_name = "easy_left"
        elif problem_name == "right":
            goal_pos = (2.0, 1.0, -2.0)
            obstacle_layout_name = "easy_right"

        task_specifications[f"{problem_name}_env_easy"] = {
            "obstacle_layout_name": obstacle_layout_name,
            # Problems will be created by randomly choosing object types and sizes from these options.
            "object_types_and_sizes": {
                "rectangle": [rectangle_size2],
            },
            # The number of objects in the scenario. Will try all in this list.
            "num_objects": [1],
            # The start and goal configurations for the objects. These are ordered pairs and will be chosen from the list in order. There must be at least max(num_objects) configurations in the list.
            "cfg_obj_start_goal_l": [
                                     (start_pos, goal_pos)],
            # The number of robots in the scenario. Will try all in this list.
            "robot_counts": [1,2,3],
            # Where the robots begin.
            "robot_start_configurations": [
                (-0.5, 3.0, 0.0),      # Robot 1
                (0.0,  3.0, 0.0),      # Robot 2
                (0.5,  3.0, 0.0),      # Robot 3
            ],
            # Number of repetitions per problem configuration
            "num_reps_per_problem": 1
        }

    # ================================ 
    # MULTI-OBJECT EMPTY (up, right)
    # ================================
    multi_object_env_problem_names = ["up", "right"]
    obstacle_layout_name = "empty"
    start_pos_l = [(-1.0, 1.0, -1.0),
                   ( 0.0, 1.0, 1.0),
                   ( 1.0, 1.0, -1.5),
                   ( -2.0, 1.0, 1.5),
                   ( 2.0, 1.0, -2.0)]
    for problem_name in multi_object_env_problem_names:
        if problem_name == "up":
            goal_pos_l = [(1.0,  2.0, 0.0),
                          (0.0,  2.0, 0.0),
                          (-1.0, 2.0, 0.0),
                          (2.0, 2.0,  0.0),
                          (-2.0, 2.0, 0.0)]
        elif problem_name == "right":
            goal_pos_l = [(2.0,  1.0, 0.0),
                          (1.0,  1.0, 0.0),
                          (0.0,  1.0, 0.0),
                          (-1.0, 1.0, 0.0),
                          (3.0,  1.0, 0.0)]

        task_specifications[f"{problem_name}multi_env_empty"] = {
            "obstacle_layout_name": obstacle_layout_name,
            "object_types_and_sizes": {
                "rectangle": [rectangle_size1, rectangle_size2, rectangle_size3],
                "circle": [circle_size2],
            },
            "robot_counts": [3,6,9],
            "num_objects": [3],
            "cfg_obj_start_goal_l": [(start_pos_l[0], goal_pos_l[0]),
                                    (start_pos_l[1], goal_pos_l[1]),
                                    (start_pos_l[2], goal_pos_l[2]),
                                    (start_pos_l[3], goal_pos_l[3]),
                                    (start_pos_l[4], goal_pos_l[4])],
            "robot_start_configurations": [
                        (-1.0 + 0.2 * i, 0.0, 0.0) for i in range(12)
                ],
            "num_reps_per_problem": 5
        }

    # ================================ 
    # MULTI-OBJECT WALL (up, left, right)
    # ================================
    multi_object_env_problem_names = ["up", "down"]
    start_pos_l = [(-1.0, 1.0,  1.0),
                   ( 0.0, 1.0,  -1.0),
                   ( 1.0, 1.0,  1.5),
                   ( -2.0, 1.0, -1.5),
                   ( 2.0, 1.0,  2.0)]
    for problem_name in multi_object_env_problem_names:
        if problem_name == "up":
            obstacle_layout_name = "wall_up"
            goal_pos_l = [(1.0,  3.0, 0.0),
                          (0.0,  3.0, 0.0),
                          (-1.0, 3.0, 0.0),
                          (2.0, 3.0, 0.0),
                          (-2.0, 3.0, 0.0)]
        task_specifications[f"{problem_name}multi_env_wall"] = {
            "obstacle_layout_name": obstacle_layout_name,
            "object_types_and_sizes": {
                "rectangle": [rectangle_size1, rectangle_size2, rectangle_size3],
                "circle": [circle_size2, circle_size3, circle_size4],
            },
            "robot_counts": [3, 6, 9],
            "num_objects": [3],
            "cfg_obj_start_goal_l": [(start_pos_l[0], goal_pos_l[0]),
                                    (start_pos_l[1], goal_pos_l[1]),
                                    (start_pos_l[2], goal_pos_l[2]),
                                    (start_pos_l[3], goal_pos_l[3]),
                                    (start_pos_l[4], goal_pos_l[4])],
            "robot_start_configurations": [
                        (-1.0 + 0.2 * i, 0.0, 0.0) for i in range(9)
                ],
            "num_reps_per_problem": 10
        }

    if tasks_to_run:
        task_specifications = {spec_name:task_specifications[spec_name] for spec_name in tasks_to_run}

    # Generate scenarios using random sampling
    import random
    scenario_id = 0
    
    for spec_name, spec in task_specifications.items():
        obstacle_type = spec["obstacle_layout_name"]  # Get obstacle layout from spec
        for num_objects in spec["num_objects"]:
            for robot_count in spec["robot_counts"]:
                # Get available object types and their sizes
                object_types = list(spec["object_types_and_sizes"].keys())
                
                # Generate the specified number of repetitions for this problem configuration
                for rep in range(spec["num_reps_per_problem"]):
                    # Randomly sample object types and sizes for this scenario
                    object_configurations = []
                    
                    for i in range(num_objects):

                        random.seed(42 + rep + i)
                        # Randomly choose object type
                        obj_type = random.choice(object_types)
                        
                        # Randomly choose size for this object type
                        available_sizes = spec["object_types_and_sizes"][obj_type]
                        obj_size = random.choice(available_sizes)
                        
                        # Get start and goal positions (cycling through available configurations)
                        start_pos_ori, goal_pos_ori = spec["cfg_obj_start_goal_l"][i % len(spec["cfg_obj_start_goal_l"])]
                        start_pos, start_orientation = start_pos_ori[:2], start_pos_ori[2]
                        goal_pos, goal_orientation = goal_pos_ori[:2], goal_pos_ori[2]


                        object_configurations.append({
                            "type": obj_type,
                            "size": obj_size,
                            "start_position": start_pos,
                            "goal_position": goal_pos,
                            "start_orientation": start_orientation,
                            "goal_orientation": goal_orientation
                        })
                    
                    # Create scenario
                    scenario = TestScenario(
                        scenario_id=f"scenario_{scenario_id:03d}",
                        num_robots=robot_count,
                        robot_start_configurations=spec["robot_start_configurations"][:robot_count],
                        obstacle_type=obstacle_type,  # Use obstacle layout from spec
                        object_type=object_configurations[0]["type"],  # Use the actual object type
                        object_size=object_configurations[0]["size"],  # Use the actual object size
                        start_position=object_configurations[0]["start_position"][:2],
                        goal_position=object_configurations[0]["goal_position"][:2],
                        start_orientation=object_configurations[0]["start_orientation"],
                        goal_orientation=object_configurations[0]["goal_orientation"],
                        seed=42 + rep,  # Different seed for each repetition
                        problem_name=spec_name,  # Store the problem specification name
                        is_multi_object= len(object_configurations) > 1,
                        object_configurations=object_configurations
                    )
                    if scenario.num_robots > 3 and not scenario.is_multi_object:
                        print(f"SKIPPING Scenario {scenario.scenario_id} has {scenario.num_robots} robots but is not a multi-object scenario")
                        continue
                    scenarios.append(scenario)
                    scenario_id += 1
    
    print(f"Generated {len(scenarios)} test scenarios")
    return scenarios, tasks_to_run


def main():
    """Main function to run explicit model evaluation."""
    parser = argparse.ArgumentParser(description="Run explicit model evaluation")
    parser.add_argument("-v", "--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--output-dir", default="output/explicit_model_evaluation", help="Output directory")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Maximum number of scenarios to run")
    parser.add_argument("--model-checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--model-types", nargs="+", default=["dc"], choices=["dc"], help="Model types to evaluate: dc")
    
    args = parser.parse_args()
    
    print("=== Explicit Model Evaluation ===")
    print(f"Visualization: {args.visualize}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model types: {args.model_types}")
    
    # Create evaluator
    evaluator = ModelEvaluator(output_dir=args.output_dir)
    
    # Generate scenarios
    print("Generating test scenarios...")
    scenarios, tasks_to_run = create_comprehensive_explicit_scenarios()
    
    if args.max_scenarios:
        scenarios = scenarios[:args.max_scenarios]
        print(f"Limited to {len(scenarios)} scenarios")
    
    # Map short codes to full model types and checkpoints
    model_type_mapping = {
        "dc": ("discrete", cfg.checkpoints.c_r_flex),
    }
    
    # Map model type to method name
    method_mapping = {
        "discrete": "discrete-continuous",
    }

    
    all_results = []
    model_names = []
    
    # Iterate over each model type.
    # for scenario_idx, scenario in enumerate(scenarios):
    for model_code in args.model_types:
        model_type, default_checkpoint = model_type_mapping[model_code]
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()} model ({model_code})")
        print(f"{'='*60}")
        
        # Get model checkpoint
        if args.model_checkpoint:
            model_checkpoint = args.model_checkpoint
        else:
            model_checkpoint = default_checkpoint
        
        if model_checkpoint is not None:
            print(f"Using model checkpoint: {model_checkpoint}")
        else:
            print("Using heuristic model (no checkpoint required)")
        
        # Create model configuration
        model_config = ModelConfig(
            name=f"explicit_eval_{model_type}",
            checkpoint_path=model_checkpoint,
            model_type=model_type,
            visualize=args.visualize,
            max_iterations=100,
            goal_tolerance_position=0.15,
            goal_tolerance_orientation=0.5
        )
        
        # Run evaluation
        method = method_mapping.get(model_type, model_type)
        results = evaluator.evaluate_model(model_config, scenarios, "", method)
        
        # Store results
        all_results.extend(results)
        model_names.append(model_config.name)
        
    
    # Save CSV results
    print(f"\n{'='*60}")
    print("Saving CSV results...")
    csv_filepath = evaluator.save_results_csv(all_results)
    
    # Generate summary report
    print("\nGenerating summary report...")
    summary_stats = evaluator.generate_summary_report(model_names)
    
    # Create visualization
    if all_results:
        print("Creating visualization...")
        evaluator.create_visualization(model_names)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")
    print(f"CSV results saved to: file://{csv_filepath.absolute()}")
    print(f"Evaluated {len(model_names)} model types: {', '.join(model_names)}")


if __name__ == "__main__":
    main()
    