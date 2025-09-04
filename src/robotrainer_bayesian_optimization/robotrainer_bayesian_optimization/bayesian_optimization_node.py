import os

import rclpy
import yaml
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.modelbridge.registry import Generators
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .bayesian_optimisation import construct_generation_strategy, qUpperConfidenceBound
from .utils import BagReader, create_new_scenario, get_mean_values_from_bag


class BayesianOptimizationNode(Node):

    def __init__(self):
        super().__init__("robotrainer_bayesian_optimization")
        
        self.srv = self.create_service(
            Trigger, "/robotrainer_bayesian_optimization/update", self.update_callback
        )

        self.subscriber = self.create_subscription(
            String,
            "/robotrainer_user_study_manager/study_status",
            self.study_status_callback,
            10,
        )

        self.study_status = ""
        self.bags_folder = "/home/docker/ros_ws/data/bags/"
        # Load initial data from config file
        self.config_file_path = "/home/docker/ros_ws/src/robotrainer_bayesian_optimization/robotrainer_bayesian_optimization/config.yaml"
        self.initial_data = {}
        with open(self.config_file_path, "r") as file:
            config = yaml.safe_load(file)
            self.initial_data = config["initial_data_files"]

        self.preexisting_trials = []
        
        self.bag_reader = BagReader()
        
        for key, value in self.initial_data.items():
            self.get_logger().info(f"Loaded initial data: {key} -> {value}")
            mean_values = get_mean_values_from_bag(value, self.bag_reader)
            trial_data = (
                {"virtual_force": key},
                {"wrench_force": mean_values["wrench_force_y"]},
            )
            self.preexisting_trials.append(trial_data)

        self.get_logger().info(f"Loaded initial values: {self.preexisting_trials}")

        # Init BO model
        generation_strategy = construct_generation_strategy(
            generator_spec=GeneratorSpec(
                model_enum=Generators.BOTORCH_MODULAR,
                model_kwargs={"botorch_acqf_class": qUpperConfidenceBound},
            ),
            node_name="BoTorch w/ Custom Components",
        )

        self.client = Client()

        virtual_force = RangeParameterConfig(
            name="virtual_force", parameter_type="int", bounds=(0, 120)
        )
        parameters = [virtual_force]
        self.client.configure_experiment(
            parameters=parameters,
            # The following arguments are only necessary when saving to the DB
            name="virtual_force_impact_experimant",
            description="Find out the impact of virtual force on human participant",
            owner="developer",
        )

        first_metric_name = (
            "wrench_force"  # this name is used during the optimization loop
        )
        objective = (
            f"{first_metric_name}"  # minimization is specified by the negative sign
        )
        self.client.configure_optimization(objective=objective)
        self.client.set_generation_strategy(
            generation_strategy=generation_strategy,
        )

        for parameters, data in self.preexisting_trials:
            trial_index = self.client.attach_trial(parameters=parameters)
            self.client.complete_trial(trial_index=trial_index, raw_data=data)

        next_trial = self.client.get_next_trials(max_trials=1)
        self.next_index, self.next_parameters = next(iter(next_trial.items()))
        self.get_logger().info(f"Next suggestion:\n - index: {self.next_index}, \n - parameters: {self.next_parameters}")
        
        # Create Scenario
        new_scenario = create_new_scenario(self.next_parameters["virtual_force"], "initial_scenario")

        # Save yaml
        scenario_name = f"initial_scenario.yaml"
        scenario_path_in_container = (
            "/home/docker/ros_ws/data/scenarios/" + scenario_name
        )
        with open(scenario_path_in_container, "w") as file:
            yaml.dump(new_scenario, file)
        self.get_logger().info(f"Created new scenario file: {scenario_name}")

        self.get_logger().info("Successfully started robotrainer_bayesian_optimization")

    def study_status_callback(self, msg):
        if not (self.study_status == msg.data):
            self.get_logger().info(f"Study status updated to: {msg.data}")
        self.study_status = msg.data

    def update_callback(self, request, response):
        self.get_logger().info("Incoming request...")
        matches = [os.path.join(self.bags_folder, f) for f in os.listdir(self.bags_folder) if f.startswith(self.study_status) and f.endswith(".bag")]
        bag_file_path = matches[0] if len(matches) == 1 else ""

        if not bag_file_path:
            response.success = False
            response.message = "Can not find bag file path or there are two of them!"
            return response
        self.get_logger().info(f"Bag file path: {bag_file_path}")

        # Process Data
        mean_values = get_mean_values_from_bag(bag_file_path, self.bag_reader)

        # Update BO model
        parameters, raw_data = (
                self.next_parameters,
                {"wrench_force": mean_values["wrench_force_y"]},
            )
        self.client.complete_trial(trial_index=self.next_index, raw_data=raw_data)
        self.get_logger().info(
            f"Completed trial {self.next_index} with parameters {parameters} and data {raw_data}"
        )
        trials = self.client.get_next_trials(max_trials=1)
        self.next_index, self.next_parameters = next(iter(trials.items()))
        self.get_logger().info(f"Next suggestion:\n - index: {self.next_index}, \n - parameters: {self.next_parameters}")
        
        # Create Scenario
        scenario_id = f"trial_{self.next_index}_force_{self.next_parameters['virtual_force']}"
        new_scenario = create_new_scenario(self.next_parameters["virtual_force"], scenario_id)

        # Save yaml
        scenario_name = f"{scenario_id}.yaml"
        scenario_path_in_container = (
            "/home/docker/ros_ws/data/scenarios/" + scenario_name
        )
        with open(scenario_path_in_container, "w") as file:
            yaml.dump(new_scenario, file)
        
        response.success = True
        response.message = scenario_id
        self.get_logger().info(f"Created new scenario file: {scenario_name}")
        return response


def main():
    rclpy.init()

    node = BayesianOptimizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        return

    rclpy.shutdown()


if __name__ == "__main__":
    main()
