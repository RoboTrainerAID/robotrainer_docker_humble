# RoboTrainer Bayesian Optimization

A ROS 2 Humble package for automated parameter optimization in robotic training scenarios using Bayesian Optimization with custom acquisition functions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Package Structure](#package-structure)
- [Technical Details](#technical-details)
- [Data Processing](#data-processing)
- [Acquisition Functions](#acquisition-functions)
- [Docker Environment](#docker-environment)

## Overview

This package implements an intelligent Bayesian Optimization framework for optimizing virtual force parameters in robotic training applications. It uses advanced machine learning techniques (specifically Gaussian Processes and custom acquisition functions) to efficiently explore the parameter space and identify optimal configurations based on real-time sensor feedback from human-robot interaction experiments.

The system is designed for use with the RoboTrainer platform, automatically adjusting virtual force parameters to optimize training effectiveness while monitoring human participant responses through force/torque sensors, velocity measurements, and biometric data.

## Features

- **Advanced Bayesian Optimization**: Implements custom acquisition functions including:
  - qNIPV (q-Negative Integrated Posterior Variance) for exploration-focused optimization
  - qUCB (q-Upper Confidence Bound) for balanced exploration-exploitation
- **ROS 2 Humble Integration**: Full native ROS 2 support with service-based interface
- **Automated Scenario Generation**: Dynamically creates training scenarios based on optimization suggestions
- **Comprehensive Data Processing**: Reads and processes ROS bag files with multiple sensor modalities
- **Docker Support**: Fully containerized environment with all dependencies pre-configured
- **Ground Truth Mode**: Optional random sampling mode for baseline comparisons
- **Prediction Tracking**: Stores model predictions and optimization state for analysis
- **Initial Data Integration**: Supports pre-existing trial data for warm-starting the optimization

## Architecture

The system consists of three main components:

1. **Bayesian Optimization Node** (`bayesian_optimization_node.py`): Core ROS 2 node managing the optimization loop
2. **Custom Acquisition Functions**: 
   - `bayesian_optimization_qNIPV.py`: Implements exploration-focused qNIPV
   - `bayesian_optimization_qUCB.py`: Implements UCB-based acquisition
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

## Installation

### Prerequisites

- Docker and Docker Compose
- X11 server (for GUI applications)
- Linux/macOS operating system

### Building the Docker Container

1. Clone the repository:
```bash
git clone https://github.com/RoboTrainerAID/robotrainer_docker_humble.git
cd robotrainer_docker_humble
```

2. Build the Docker image:
```bash
./build_docker.sh
```

This creates a Docker image with:
- ROS 2 Humble Desktop
- Python 3.10
- BoTorch and Ax Platform for Bayesian optimization
- Required dependencies (pandas, rosbags, numpy, seaborn)

## Quick Start

### Option 1: Interactive Mode

```bash
# Start the Docker container
./start_docker.sh

# Inside the container, build the workspace
colcon build --packages-select robotrainer_bayesian_optimization

# Source the workspace
source install/setup.bash

# Launch the optimization node
ros2 launch robotrainer_bayesian_optimization launch.py
```

### Option 2: Autostart Mode

```bash
# Automatically start the container and launch the node
./autostart.sh
```

### Testing the Service

Trigger an optimization update:
```bash
ros2 service call /robotrainer_bayesian_optimization/update std_srvs/srv/Trigger
```

## Configuration

### Initial Data Configuration

Edit `src/robotrainer_bayesian_optimization/robotrainer_bayesian_optimization/config.yaml`:

```yaml
initial_data_files:
  0: "/path/to/bag/file/with/force_0.bag"
  50: "/path/to/bag/file/with/force_50.bag"
  100: "/path/to/bag/file/with/force_100.bag"
```

Keys represent virtual force values, values are paths to bag files with observed outcomes.

### Optimization Objective

The default optimization objective is `wrench_force_y` (minimizing Y-axis force). This can be modified in `bayesian_optimization_node.py`:

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

For generating random baseline data:

```python
node = BayesianOptimizationNode(is_ground_truth=True)
```

This mode randomly samples parameters without Bayesian optimization, useful for comparison studies.

### Service Interface

The node exposes a service at `/robotrainer_bayesian_optimization/update` (Trigger type):

- **Request**: Empty trigger
- **Response**: 
  - `success`: Boolean indicating successful processing
  - `message`: Scenario ID for the next trial

### Topic Subscriptions

The node subscribes to:
- `/robotrainer_user_study_manager/study_status`: Current study status identifier

## Package Structure

```
robotrainer_docker_humble/
├── Dockerfile                          # Container definition
├── build_docker.sh                     # Build script
├── start_docker.sh                     # Container launch script
├── autostart.sh                        # Automatic launch script
├── dds_profile.xml                     # DDS configuration
├── data/
│   ├── bags/                          # ROS bag files storage
│   │   ├── raw/                       # Raw bag files from experiments
│   │   └── filter_bags.sh            # Bag filtering utility
│   ├── initial_data/                  # Pre-existing trial data
│   ├── scenarios/                     # Generated scenario YAML files
│   │   └── default_scenario.yaml     # Template scenario
│   ├── models/                        # Saved optimization models (JSON)
│   └── predictions/                   # Model predictions (pickle files)
└── src/
    └── robotrainer_bayesian_optimization/
        ├── package.xml                # ROS 2 package manifest
        ├── setup.py                   # Python package setup
        ├── setup.cfg                  # Setup configuration
        ├── config/
        │   └── params.yaml           # ROS parameters (currently empty)
        ├── launch/
        │   └── launch.py             # ROS 2 launch file
        ├── robotrainer_bayesian_optimization/
        │   ├── __init__.py
        │   ├── bayesian_optimization_node.py      # Main optimization node
        │   ├── bayesian_optimization_qNIPV.py     # qNIPV acquisition function
        │   ├── bayesian_optimization_qUCB.py      # qUCB acquisition function
        │   ├── config.yaml                        # Initial data configuration
        │   └── utils.py                           # Data processing utilities
        └── test/                                  # Unit tests
            ├── test_copyright.py
            ├── test_flake8.py
            └── test_pep257.py
```

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

#### Gaussian Process Model

- Uses a standard Gaussian Process regression model
- Automatically handles noise estimation
- Supports multi-output objectives (though current implementation uses single output)

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

## Acquisition Functions

### qNIPV (q-Negative Integrated Posterior Variance)

**Purpose**: Maximizes information gain about the entire function landscape.

**Characteristics**:
- Pure exploration strategy
- Selects points that reduce global model uncertainty
- Useful early in optimization or for function mapping

**Implementation**: `bayesian_optimization_qNIPV.py`

The acquisition function quantifies the integrated posterior variance using Monte Carlo integration:

$$\text{qNIPV}(X) = -\int_{D} \mathbb{V}[f(x) | \mathcal{D}, X] \, dx$$

where $D$ is the domain, $f$ is the objective function, $\mathcal{D}$ is observed data, and $X$ are candidate points.

### qUCB (q-Upper Confidence Bound)

**Purpose**: Balances exploration and exploitation with a tunable parameter.

**Characteristics**:
- Controlled by beta parameter (exploration-exploitation tradeoff)
- Higher beta → more exploration
- Standard choice for efficient optimization

**Implementation**: `bayesian_optimization_qUCB.py`

The acquisition function is:

$$\text{qUCB}(X) = \mathbb{E}[\max(\mu(X) + |Y - \mu(X)|)]$$

where $Y \sim \mathcal{N}(\mu, \beta \frac{\pi}{2} \Sigma)$ and $\beta$ controls the confidence bound.

## Docker Environment

### Container Configuration

- **Base Image**: `osrf/ros:humble-desktop`
- **OS**: Ubuntu 22.04
- **User**: `docker` (UID/GID 1000 by default)
- **Network**: Host networking for ROS communication
- **Display**: X11 forwarding enabled
- **DDS Profile**: FastDDS configuration for ROS 2

### Volume Mounts

- `./src` → `/home/docker/ros_ws/src`: Source code
- `./data` → `/home/docker/ros_ws/data`: Persistent data storage
- `/dev` → `/dev`: Device access (sensors)

### Environment Variables

- `ROS_DOMAIN_ID`: Set to 36 (configurable in `start_docker.sh`)
- `PYTHONPATH`: Includes ROS 2 workspace
- `DISPLAY`: X11 display for GUI applications

## Development and Testing

### Running Tests

```bash
# Inside the container
colcon test --packages-select robotrainer_bayesian_optimization
```

Tests include:
- Copyright checks
- PEP 257 docstring compliance
- Flake8 linting

### Adding New Acquisition Functions

1. Create a new file following the pattern of `bayesian_optimization_qNIPV.py`
2. Implement the acquisition function inheriting from appropriate BoTorch base class
3. Register with `@acqf_input_constructor` decorator
4. Create a `construct_generation_strategy` function
5. Update the node to use the new acquisition function

### Debugging

Enable verbose logging:
```bash
ros2 launch robotrainer_bayesian_optimization launch.py --log-level debug
```

Check service availability:
```bash
ros2 service list | grep robotrainer
```

Monitor topics:
```bash
ros2 topic echo /robotrainer_user_study_manager/study_status
```

## Troubleshooting

### Container Won't Start

- Check Docker daemon is running
- Verify X11 forwarding: `xhost +local:docker`
- Check port conflicts (if not using host networking)

### Bag Files Not Found

- Ensure bag files are in `data/bags/raw/`
- Verify study status matches bag filename prefix
- Check file permissions

### Optimization Not Updating

- Verify service is callable: `ros2 service call /robotrainer_bayesian_optimization/update std_srvs/srv/Trigger`
- Check initial data is properly loaded in `config.yaml`
- Review node logs for errors

### Model Prediction Errors

- Ensure sufficient initial data (minimum 2-3 trials)
- Check that optimization objective exists in bag file data
- Verify data quality (no NaN or infinite values)

## Citation

If you use this package in your research, please cite:

```bibtex
@misc{robotrainer_bayesian_optimization,
  title={RoboTrainer Bayesian Optimization},
  author={RoboTrainerAID},
  year={2025},
  url={https://github.com/RoboTrainerAID/robotrainer_docker_humble}
}
```

## License

[License information to be added]

## Maintainer

- **Email**: docker@todo.todo
- **Repository**: https://github.com/RoboTrainerAID/robotrainer_docker_humble
- **Branch**: bayesian_optimization

## Acknowledgments

This package builds upon:
- [Ax Platform](https://ax.dev/) - Adaptive experimentation framework
- [BoTorch](https://botorch.org/) - Bayesian optimization in PyTorch
- [ROS 2](https://docs.ros.org/en/humble/) - Robot Operating System
- [rosbags](https://gitlab.com/ternaris/rosbags) - Pure Python ROS bag reader

## Usage

### Basic Workflow

The package provides automated optimization loops that can be triggered via ROS 2 services. The `update` service runs one iteration of the Bayesian optimization process.

### Parameters

Configure optimization parameters in the launch file (`launch.py`) before starting.

## Troubleshooting

Ensure ROS 2 is properly sourced and the service is running before making service calls.


