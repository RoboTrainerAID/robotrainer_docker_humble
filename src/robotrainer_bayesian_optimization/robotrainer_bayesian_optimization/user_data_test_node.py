import os
import random

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .utils import create_new_scenario


class UserDataTestNode(Node):

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
        self.bags_folder = "/home/docker/ros_ws/data/bags/raw/"

        self.experiments = []
        with open("/home/docker/ros_ws/data/experiments.yaml", "r") as file:
            experiments_dict = yaml.safe_load(file)
            # Convert YAML dict to list for index-based access
            self.experiments = list(experiments_dict.values())

        random.seed(32)
        self.random_trial_indices = random.sample(
            range(len(self.experiments)), k=len(self.experiments)
        )

        self.next_index = 0
        self.next_experiment = self.random_trial_indices.pop(0)

        # Create Scenario
        new_scenario = create_new_scenario(
            self.experiments[self.next_experiment]["parameters"]["force"],
            self.experiments[self.next_experiment]["name"],
            self.experiments[self.next_experiment]["parameters"]["force_direction"],
        )

        # Save yaml
        scenario_name = f"{self.experiments[self.next_experiment]['name']}.yaml"
        scenario_path_in_container = (
            "/home/docker/ros_ws/data/scenarios/" + scenario_name
        )
        with open(scenario_path_in_container, "w") as file:
            yaml.dump(new_scenario, file)
        self.get_logger().info(f"Created new scenario file: {scenario_name}")

        parameters_lines = [
            f"{key}: {value}"
            for key, value in self.experiments[self.next_experiment]["parameters"].items()
        ]
        self.get_logger().info(
            "New scenario parameters are: "
            f"\nScenario name: {self.experiments[self.next_experiment]['name']}"
            f"\n{chr(10).join(parameters_lines)}"
        )

        self.get_logger().info("Successfully started data test node. ")

    def study_status_callback(self, msg):
        if not (self.study_status == msg.data):
            self.get_logger().info(f"Study status updated to: {msg.data}")
        self.study_status = msg.data

    def update_callback(self, request, response):
        self.get_logger().info("Incoming request...")
        matches = [
            os.path.join(self.bags_folder, f)
            for f in os.listdir(self.bags_folder)
            if f.startswith(self.study_status) and f.endswith(".bag")
        ]
        bag_file_path = matches[0] if len(matches) == 1 else ""

        if not bag_file_path:
            response.success = False
            response.message = (
                "Can not find bag file path or there are two of them! Number of matches: "
                + str(len(matches))
            )
            return response
        self.get_logger().info(f"Bag file path: {bag_file_path}")

        if not self.random_trial_indices:
            response.success = False
            response.message = "All test trials completed!"
            self.get_logger().info("All test trials completed!")
            return response

        # Create Scenario
        self.next_index += 1
        self.next_experiment = self.random_trial_indices.pop(0)
        scenario_id = f"{self.experiments[self.next_experiment]['name']}"
        new_scenario = create_new_scenario(
            self.experiments[self.next_experiment]["parameters"]["force"],
            scenario_id,
            self.experiments[self.next_experiment]["parameters"]["force_direction"],
        )

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
        parameters_lines = [
            f"{key}: {value}"
            for key, value in self.experiments[self.next_experiment]["parameters"].items()
        ]
        self.get_logger().info(
            "New scenario parameters are: "
            f"\nScenario name: {self.experiments[self.next_experiment]['name']}"
            f"\n{chr(10).join(parameters_lines)}"
        )

        self.get_logger().info(f"Successfully processed request: {response.success}.")
        self.get_logger().info(f"Response message: {response.message}")
        return response


def main():
    rclpy.init()

    node = UserDataTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        return

    rclpy.shutdown()


if __name__ == "__main__":
    main()
