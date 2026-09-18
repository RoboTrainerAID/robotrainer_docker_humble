import os
import re
import requests

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .utils import create_new_scenario, create_new_static_scenario


class UserStudyDifficultyNode(Node):

    def __init__(self):
        super().__init__("robotrainer_bayesian_optimization")

        self.familiarization_mode = False
        self.standard_study_mode = True
        self.nasa_tlx_plus_mode = False
        self.bayesian_optimization_mode = False
        self.first_bayesian_optimization_experiment = False
        self.questionnaire_needed = False
        self.bag_per_force = {}
        self.bo_force = [15] # Example force levels for Bayesian Optimization

        self.BACKEND_URL = os.environ.get("BACKEND_URL", "http://192.168.1.5:5001")
        self.srv = self.create_service(
            Trigger, "/robotrainer_bayesian_optimization/update", self.update_callback
        )

        self.study_status = ""
        self.subscriber = self.create_subscription(
            String,
            "/robotrainer_user_study_manager/study_status",
            self.study_status_callback,
            10,
        )

        if self.study_status:
            self.user_id = self.extract_user_id(self.study_status)
        else:
            self.user_id = "U001"

        participants_response = requests.get(
            f'{self.BACKEND_URL}/participants/{self.user_id}', 
            headers={'accept': 'application/json'}
        )

        if participants_response.status_code == 200:
            self.participant_id = participants_response.json()
        else:
            self.get_logger().info(f"Error: {participants_response.status_code}, {participants_response.text}") 

        self.bags_folder = "/home/docker/ros_ws/data/bags/raw/"

        self.standard_experiments = []
        with open("/home/docker/ros_ws/data/experiments_user_study_difficulty_default.yaml", "r") as file:
            experiments_dict = yaml.safe_load(file)
            # Convert YAML dict to list for index-based access
            self.standard_experiments = list(experiments_dict.values())

        # TODO: initialize BO experiment

        self.standard_study_trial_indices = list(range(0, len(self.standard_experiments), 1))

        self.next_experiment_number = self.standard_study_trial_indices.pop(0)

        # Create Scenario
        new_scenario = create_new_static_scenario(
            self.standard_experiments[self.next_experiment_number]["parameters"]["force"],
            self.standard_experiments[self.next_experiment_number]["name"],
            self.standard_experiments[self.next_experiment_number]["parameters"]["force_direction"],
        )

        # Save yaml
        scenario_name = f"{self.standard_experiments[self.next_experiment_number]['name']}.yaml"
        scenario_path_in_container = (
            "/home/docker/ros_ws/data/scenarios/" + scenario_name
        )
        with open(scenario_path_in_container, "w") as file:
            yaml.dump(new_scenario, file)
        self.get_logger().info(f"Created new scenario file: {scenario_name}")

        data = {   
            "details": {},
            "experiment_number": self.standard_experiments[self.next_experiment_number]["name"],
            "extra_step": self.standard_experiments[self.next_experiment_number]["parameters"]["extra_nasa_tlx_questions"],
            "flag": "" ,
            "force": self.standard_experiments[self.next_experiment_number]["parameters"]["force"],
            "method": self.standard_experiments[self.next_experiment_number]["parameters"]["experiment_method"],
            "status": "waiting",
        }
        self.post_new_experiment(data)

        parameters_lines = [
            f"{key}: {value}"
            for key, value in self.standard_experiments[self.next_experiment_number]["parameters"].items()
        ]
        self.get_logger().info(
            "New scenario parameters are: "
            f"\nScenario name: {self.standard_experiments[self.next_experiment_number]['name']}"
            f"\n{chr(10).join(parameters_lines)}"
        )

        self.get_logger().info("Successfully started user study difficulty node. ")

    def post_new_experiment(self, data):
        url = f"{self.BACKEND_URL}/participants/{self.participant_id}/new-experiment"
        
        response = requests.post(url, json=data, headers={'accept': 'application/json'})
        if response.status_code == 200:
            self.get_logger().info("Successfully posted new experiment")
        else:
            self.get_logger().info(f"Error posting new experiment: {response.status_code}, {response.text}")
    
    def study_status_callback(self, msg):
        if not (self.study_status == msg.data):
            self.get_logger().info(f"Study status updated to: {msg.data}")
        self.study_status = msg.data

    def extract_user_id(self, status):
        match = re.search(r'_(U\d+)_', status)
        if not match:
            raise ValueError(f"Could not find user ID in filename: {status}")
        return match.group(1)

    def create_nasa_tlx_plus_experiments(self, bag_per_force):
         # TODO: implement logic to create NASA-TLX+ experiments based on bag_per_force
         with open("/home/docker/ros_ws/data/experiments_user_study_difficulty_nasa_tlx_plus.yaml", "r") as file:
            experiments_dict = yaml.safe_load(file)
            # Convert YAML dict to list for index-based access
            experiments = list(experiments_dict.values())
            return experiments

    def update_callback(self, request, response):

        next_experiment = None
        # seting status of last experiment to ready
        waiting_experiment_response = requests.get(
            f"{self.BACKEND_URL}/participants/{self.participant_id}/waiting-experiment", 
            headers={'accept': 'application/json'}
        )
        if waiting_experiment_response.status_code == 200:
            waiting_experiment = waiting_experiment_response.json()
        else:
            self.get_logger().info(f"Error: {waiting_experiment_response.status_code}, {waiting_experiment_response.text}") 

        status_url = f"{self.BACKEND_URL}/experiments/{waiting_experiment}/change-status"
        status_response = requests.post(status_url, json={"status": "ready"}, headers={'accept': 'application/json'})
        if status_response.status_code == 200:
            self.get_logger().info("Successfully updated experiment status")
        else:
            self.get_logger().info(f"Error updating experiment status: {status_response.status_code}, {status_response.text}")

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

        if self.familiarization_mode:
            # TODO: implement familiarization mode
            pass

        if self.standard_study_mode:
            self.bag_per_force[self.standard_experiments[self.next_experiment_number]["parameters"]["force"]] = bag_file_path

            if not self.standard_study_trial_indices:
                self.standard_study_mode = False
                self.nasa_tlx_plus_mode = True
                self.nasa_tlx_plus_experiments = self.create_nasa_tlx_plus_experiments(self.bag_per_force)
                self.nasa_tlx_plus_experiment_indices = list(range(
                    0,
                    len(self.nasa_tlx_plus_experiments), 
                    1)
                )
                self.get_logger().info("All standard study experiments completed. Switching to NASA-TLX+ mode.")
            else:
                # Create Scenario
                self.next_experiment_number = self.standard_study_trial_indices.pop(0)
                next_experiment = self.standard_experiments[self.next_experiment_number]
                self.questionnaire_needed = True

        if self.nasa_tlx_plus_mode:
            if not self.nasa_tlx_plus_experiment_indices:
                self.nasa_tlx_plus_mode = False
                self.bayesian_optimization_mode = True
                self.first_bayesian_optimization_experiment = False
                self.get_logger().info("All NASA-TLX+ experiments completed. Switching to Bayesian Optimization mode.")
                # TODO: implement logic to initialize Bayesian Optimization experiments

            else:
                # Create Scenario
                self.next_experiment_number = self.nasa_tlx_plus_experiment_indices.pop(0)
                next_experiment = self.nasa_tlx_plus_experiments[self.next_experiment_number]
                self.questionnaire_needed = True

        if self.bayesian_optimization_mode:
            if self.first_bayesian_optimization_experiment:
                self.get_logger().info("Starting Bayesian Optimization mode.")
                self.first_bayesian_optimization_experiment = False
                # TODO: initialize Bayesian Optimization experiments
            else:
                self.get_logger().info("Starting second+ Bayesian Optimization run.")
                # TODO: implement logic for Bayesian Optimization experiments
                # TODO: implement end of experiments logic, e.g., send a message to the user study manager that all experiments are completed
                self.next_bo_force = self.bo_force.pop(0) if self.bo_force else None
                if self.next_bo_force:
                    next_experiment = {
                        "name": f"bayesian_optimization_{self.next_bo_force}",
                        "parameters": {
                            "force": self.next_bo_force,
                            "force_direction": 1,  # Example direction
                            "experiment_type": "bo",
                            "extra_nasa_tlx_questions": False,
                            "experiment_method": "bayesian_optimization",
                        },
                    }
                    self.questionnaire_needed = True
                else:
                    self.get_logger().info("All Bayesian Optimization experiments completed.")
                    end_experiment_url = f"{self.BACKEND_URL}/participants/{self.user_id}"
                    end_response = requests.patch(end_experiment_url, json={"is_finished": True}, headers={'accept': 'application/json'})
                    if end_response.status_code == 200:
                        self.get_logger().info("Successfully marked participant as finished.")
                    else:
                        self.get_logger().info(f"Error marking participant as finished: {end_response.status_code}, {end_response.text}")

                    response.success = True
                    response.message = "All experiments completed."
                    return response

        scenario_id = f"{next_experiment['name']}"
        new_scenario = create_new_scenario(
            next_experiment["parameters"]["force"],
            scenario_id,
            next_experiment["parameters"]["force_direction"],
        )

        if self.questionnaire_needed:
            # create next experiment in the database
            data = {
                "details": {},
                "experiment_number": next_experiment["name"],
                "extra_step": next_experiment["parameters"]["extra_nasa_tlx_questions"],
                "flag": next_experiment.get("flag", ""),
                "force": next_experiment["parameters"]["force"],
                "method": next_experiment["parameters"]["experiment_method"],
                "status": "waiting",
            }
            self.post_new_experiment(data)

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
            for key, value in next_experiment["parameters"].items()
        ]
        self.get_logger().info(
            "New scenario parameters are: "
            f"\nScenario name: {next_experiment['name']}"
            f"\n{chr(10).join(parameters_lines)}"
        )

        self.get_logger().info(f"Successfully processed request: {response.success}.")
        self.get_logger().info(f"Response message: {response.message}")
        return response


def main():
    rclpy.init()

    node = UserStudyDifficultyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt")
        return

    rclpy.shutdown()


if __name__ == "__main__":
    main()
