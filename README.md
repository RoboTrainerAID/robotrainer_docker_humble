
# RoboTrainer Bayesian Optimization
A ROS 2 Humble package for Bayesian Optimization User Study.

## How to start

1. Clone the repository:
```bash
git clone https://github.com/RoboTrainerAID/robotrainer_docker_humble.git
cd robotrainer_docker_humble
```
2. Build the Docker image:
```bash
./build_docker.sh
```
3. Start the Docker container:
```bash
./start_docker.sh
ros2 launch robotrainer_bayesian_optimization launch.py

# Or with autostart
./autostart.sh
```
### Testing the Service
Trigger an optimization update:
```bash
ros2 service call /robotrainer_bayesian_optimization/update std_srvs/srv/Trigger
```
A detailed description of how to start all the necessary RoboTrainer services and how to use this node can be found [here](https://github.com/RoboTrainerAID/robotrainer_user_performance/blob/melodic/robotrainer_study_bayesian_optimization/README.md).

## Architecture
The system consists of three main components:
1. **Bayesian Optimization Node** (`bayesian_optimization_node.py`): Core ROS 2 node managing the optimization loop
2. **Custom Acquisition Functions**: 
   - `bayesian_optimization_qNIPV.py`: Implements exploration-focused qNIPV
   - `bayesian_optimization_qUCB.py`: Implements UCB-based acquisition function
3. **Data Processing Utilities** (`utils.py`): Handles ROS bag reading, scenario generation, and data analysis
### System Workflow
```
Initialize with historical data
         ↓
Generate next parameter suggestion
         ↓
Create scenario YAML file
         ↓
Wait for trial completion (triggered via ROS service)
         ↓
Read and process bag file data
         ↓
Update Bayesian model with results
         ↓
Save model state and predictions
         ↓
Repeat optimization loop
```


## Configuration
### Initial Data Configuration
To add initial data for the Bayesian Optimization edit `src/robotrainer_bayesian_optimization/robotrainer_bayesian_optimization/config.yaml`:
```yaml
initial_data_files:
  0: "/path/to/bag/file/with/force_0.bag"
  50: "/path/to/bag/file/with/force_50.bag"
  100: "/path/to/bag/file/with/force_100.bag"
```
Keys represent virtual force values, values are paths to bag files with observed outcomes.

### Optimization Objective
The default optimization objective is `wrench_force_y`. This can be modified in `bayesian_optimization_node.py` in `__init__` function:
```python
self.optimization_objective = "wrench_force_y"
```
Available metrics include:
- `wrench_force_x`, `wrench_force_y`, `wrench_force_z`: Force measurements
- `wrench_torque_x`, `wrench_torque_y`, `wrench_torque_z`: Torque measurements
- `twist_linear_x`, `twist_linear_y`, `twist_linear_z`: Linear velocity
- `hr`: Heart rate (from Polar OH1 sensor)
- `front`, `left`, `right`: User deviation measurements

### Parameter Bounds
Virtual force parameter range is configured in `bayesian_optimization_node.py`:
```python
virtual_force = RangeParameterConfig(
    name="virtual_force", 
    parameter_type="int", 
    bounds=(0, 100)
)
```
## Usage
### Standard Optimization Mode
1. Ensure initial data bag files are configured in `config.yaml`
2. Start the optimization node
3. The system will:
   - Load initial data to warm-start the model
   - Suggest the first parameter value
   - Create an initial scenario YAML file
   - Wait for service call to process each trial
   - Continuously optimize based on feedback
### Ground Truth Mode
For generating random baseline data. To set the node in this mode edit `is_ground_thruth` paramether in `main` function:
```python
node = BayesianOptimizationNode(is_ground_truth=True)
```
This mode randomly samples parameters without Bayesian Optimization, useful for comparison studies.

### Service Interface
The node exposes a service at `/robotrainer_bayesian_optimization/update` (Trigger type):
- **Request**: Empty trigger
- **Response**: 
  - `success`: Boolean indicating successful processing
  - `message`: Scenario ID for the next trial
### Topic Subscriptions
The node subscribes to:
- `/robotrainer_user_study_manager/study_status`: Current study status identifier

## Technical Details
### Bayesian Optimization Implementation
The package uses the **Ax Platform** (Meta's adaptive experimentation framework) with **BoTorch** (Bayesian optimization in PyTorch) for the optimization backend.
#### Generation Strategy
The optimization uses a two-phase strategy:
1. **Center Node**: Initial point generation at the parameter space center
2. **Modular BoTorch Node**: Bayesian optimization with custom acquisition functions
```python
generation_strategy = construct_generation_strategy(
    generator_spec=GeneratorSpec(
        model_enum=Generators.BOTORCH_MODULAR,
        model_kwargs={"botorch_acqf_class": qNegIntegratedPosteriorVariance}
    ),
    node_name="BoTorch w/ Custom Components",
)
```
### Data Processing Pipeline
#### Bag File Reading
The `BagReader` class processes ROS 1 bag files containing:
- Force/torque measurements (`geometry_msgs/WrenchStamped`)
- Velocity data (`geometry_msgs/TwistStamped`)
- User deviation (`robotrainer_deviation/msg/RobotrainerUserDeviation`)
- Heart rate (`std_msgs/Int32`)
- Virtual force status and magnitude
Selected topics:
```python
selected_topics = [
    "/base/output_data",
    "/base/virtual_forces/modalities_debug/resulting_force",
    "/base/virtual_forces/modalities_debug/status",
    "/robotrainer_deviation/robotrainer_deviation",
    "/biosensors/polar_oh1/hr",
    "/base/fts_adaptive_force_controller/debug/velocity_output",
]
```
#### Data Processing Steps
1. **Read bag file**: Deserialize all messages from selected topics
2. **Convert to DataFrame**: Organize data with timestamps
3. **Filter virtual force period**: Extract data only when virtual force is active
4. **Compute aggregates**: Calculate mean/median/max values per second
5. **Return metrics**: Provide summary statistics for optimization
### Scenario Generation
Scenarios are dynamically created with:
- **Virtual force magnitude**: Scaled arrow vector based on optimization suggestion
- **Random location**: Force area randomly placed along the path (avoiding start/end)
- **Consistent geometry**: Margin maintained relative to area based on default scenario
- **Unique identifiers**: Timestamp-based scenario IDs

### Adding New Acquisition Functions
1. Create a new file following the pattern of `bayesian_optimization_qNIPV.py`
2. Implement the acquisition function inheriting from appropriate BoTorch base class
3. Register with `@acqf_input_constructor` decorator
4. Create a `construct_generation_strategy` function
5. Update the node to use the new acquisition function

