from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch

def generate_launch_description():
    # Build MoveIt config with ALL necessary components
    moveit_config = MoveItConfigsBuilder("humanoid", 
                                         package_name="robot_moveit_config"
    ).robot_description(
        file_path="config/humanoid.urdf.xacro"
    ).robot_description_semantic(
        file_path="config/humanoid.srdf"
    ).trajectory_execution(
        file_path="config/moveit_controllers.yaml"
    ).planning_pipelines(
        pipelines=["ompl"]  # Load planning pipelines
    ).planning_scene_monitor(
        publish_planning_scene=True,
        publish_geometry_updates=True,
        publish_state_updates=True
    ).to_moveit_configs()
    
    return generate_move_group_launch(moveit_config)