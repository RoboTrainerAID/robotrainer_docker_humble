import os
import re
import sys

import rclpy
import requests
import yaml
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.modelbridge.registry import Generators
from ax.storage.botorch_modular_registry import register_acquisition_function
from rclpy.node import Node
from .scripts.optimal_force_models import (
    optimal_force_linear,
    optimal_force_quadratic,
    optimal_force_quadratic_maxforce,
    optimal_force_sensor_proxy,
)
from .scripts.sensor_feature_extraction import extract_user_features_from_bags
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .bayesian_optimization_qNIPV_and_qLNEI import (
    construct_generation_strategy,
    qLogNoisyExpectedImprovement,
    qNegIntegratedPosteriorVariance,
)
from .utils import create_new_scenario


class UserStudyDifficultyNode(Node):

    def __init__(self):
        super().__init__("robotrainer_bayesian_optimization")

        self.familiarization_mode = False
        self.standard_study_mode = False
        self.nasa_tlx_plus_mode = False
        self.same_force_mode = False
        self.bayesian_optimization_mode = False
        self.number_of_exploration_experiments = 5
        self.number_of_exploitation_experiments = 7
        self.current_force = 0
        self.bag_per_force = {}
        self.BACKEND_URL = os.environ.get("BACKEND_URL", "http://192.168.1.5:5001")
        self.study_status = ""
        self.user_id = "" # needed to be filled in if starting not from the beginning of the study, e.g., if the robotrainer is restarted in the middle of a study
        self.participant_id = "" # needed to be filled in if starting not from the beginning of the study, e.g., if the robotrainer is restarted in the middle of a study
        self.bags_folder = "/home/docker/ros_ws/data/bags/raw/"
        self.scenarios_folder = "/home/docker/ros_ws/data/scenarios/"

        self.srv = self.create_service(
            Trigger, "/robotrainer_bayesian_optimization/update", self.update_callback
        )

        self.subscriber = self.create_subscription(
            String,
            "/robotrainer_user_study_manager/study_status",
            self.study_status_callback,
            10,
        )

        self.familiarization_experiments = []
        self.familiarization_mode = True
        with open("/home/docker/ros_ws/data/experiments_user_study_difficulty_familiarization.yaml", "r") as file:
            experiments_dict = yaml.safe_load(file)
            # Convert YAML dict to list for index-based access
            self.familiarization_experiments = list(experiments_dict.values())
        experiment = self.familiarization_experiments.pop(0)
        self.current_force = experiment["parameters"]["force"]

        # self.standard_experiments = []
        # self.standard_study_mode = True
        # with open("/home/docker/ros_ws/data/experiments_user_study_difficulty_standard.yaml", "r") as file:
        #     experiments_dict = yaml.safe_load(file)
        #     # Convert YAML dict to list for index-based access
        #     self.standard_study_experiments = list(experiments_dict.values())
        # experiment = self.standard_experiments.pop(0)

        # self.nasa_tlx_plus_experiments = []
        # self.nasa_tlx_plus_mode = True
        # with open("/home/docker/ros_ws/data/experiments_user_study_difficulty_nasa_tlx_plus.yaml", "r") as file:
        #     experiments_dict = yaml.safe_load(file)
        #     # Convert YAML dict to list for index-based access
        #     self.nasa_tlx_plus_experiments = list(experiments_dict.values())
        # experiment = self.nasa_tlx_plus_experiments.pop(0)

        # new_experiment = {
        #     "name": "same_force_check_0013",
        #     "scenario_id": "6_green_line_force_left_60",
        #     "parameters": {
        #         "force": 60,
        #         "force_direction": 1,
        #         "experiment_type": "standard", 
        #         "experiment_method": "default", 
        #         "nasa_tlx_needed": True,
        #         "extra_nasa_tlx_questions": False,
        #     },
        # }
        # self.same_force_experiments = [new_experiment]

        # self.bayesian_optimization_experiments = []
        # self.bayesian_optimization_mode = True
        # with open("/home/docker/ros_ws/data/experiments_user_study_difficulty_bayesian_optimization.yaml", "r") as file:
        #     experiments_dict = yaml.safe_load(file)
        #     # Convert YAML dict to list for index-based access
        #     self.bayesian_optimization_experiments = list(experiments_dict.values())
        # experiment = self.bayesian_optimization_experiments.pop(0)

        if experiment["parameters"]["nasa_tlx_needed"]:
            data = {
                "details": experiment.get("details", {}),
                "experiment_name": experiment["name"],
                "status": "waiting",
                "extra_step": experiment["parameters"]["extra_nasa_tlx_questions"],
                "force": experiment["parameters"]["force"],
                "type": experiment["parameters"]["experiment_type"],
                "method": experiment["parameters"]["experiment_method"],
                "flag": experiment.get("flag", ""),
                "scenario_id": experiment["scenario_id"],
            }
            self.post_new_experiment(data)

        self.get_logger().info("Successfully started user study difficulty node.")

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

    def get_user_and_partisipant_id_from_study_status(self):
        if self.study_status:
            self.user_id = self.extract_user_id(self.study_status)
        else:
            self.user_id = "U001"
        participants_response = requests.get(
            f'{self.BACKEND_URL}/participants/{self.user_id}', 
            headers={'accept': 'application/json'}
        )
        if participants_response.status_code == 200:
            participant = participants_response.json()
            self.participant_id = participant["_id"]
        else:
            self.get_logger().info(f"Error: {participants_response.status_code}, {participants_response.text}")

    def set_last_experiment_status_to_ready(self):
        waiting_experiment_response = requests.get(
            f"{self.BACKEND_URL}/participants/{self.participant_id}/waiting-experiment", 
            headers={'accept': 'application/json'}
        )
        if waiting_experiment_response.status_code == 200:
            waiting_experiment = waiting_experiment_response.json()
        else:
            self.get_logger().info(f"Error: {waiting_experiment_response.status_code}, {waiting_experiment_response.text}") 
            waiting_experiment = 0

        status_url = f"{self.BACKEND_URL}/experiments/{waiting_experiment}/change-status"
        status_response = requests.post(status_url, json={"status": "ready"}, headers={'accept': 'application/json'})
        if status_response.status_code == 200:
            self.get_logger().info("Successfully updated experiment status")
        else:
            self.get_logger().info(f"Error updating experiment status: {status_response.status_code}, {status_response.text}")

    def check_if_bag_file_exists(self, experiment_name):
        matches = [
            os.path.join(self.bags_folder, f)
            for f in os.listdir(self.bags_folder)
            if f.startswith(experiment_name) and f.endswith(".bag")
        ]
        bag_file_path = matches[0] if len(matches) == 1 else ""
        return bag_file_path

    def create_nasa_tlx_plus_experiments(self, bag_per_force):
        # get all experiments from database
        experiments_response = requests.get(
            f"{self.BACKEND_URL}/participants/{self.participant_id}/experiments", 
            headers={'accept': 'application/json'}
        )
        if experiments_response.status_code == 200:
            experiments = experiments_response.json()
        else:
            self.get_logger().info(f"Error: {experiments_response.status_code}, {experiments_response.text}")

        tlx_performance = {}
        for experiment in experiments:
            tlx_performance[experiment["force"]] = experiment["scores"]["performance"]
        participants_response = requests.get(
            f'{self.BACKEND_URL}/participants/{self.user_id}', 
            headers={'accept': 'application/json'}
        )
        if participants_response.status_code == 200:
            participant = participants_response.json()
            max_force_n = participant["clinical_scales"]["robotrainer_max_force_right"]
        else:
            self.get_logger().info(f"Error: {participants_response.status_code}, {participants_response.text}")

        timeseries_features = extract_user_features_from_bags(bag_per_force)

        m1 = optimal_force_linear(tlx_performance, max_force_n)
        m1_scenario_id = f"8_green_line_force_left_{int(m1.optimal_force_n)}"
        m1_scenario = create_new_scenario(int(m1.optimal_force_n),m1_scenario_id)
        with open(f"{self.scenarios_folder}{m1_scenario_id}.yaml", "w") as file:
            yaml.dump(m1_scenario, file)
        m1_details = m1.details
        m1_details["force_cap_n"] = m1.force_cap_n
        m1_details["raw_crossing_n"] = m1.raw_crossing_n
        m1_details["within_device_window"] = m1.within_device_window
        m1_details["coefficients"] = m1.coefficients
        m1_details["fit_rmse"] = m1.fit_rmse
        m1_experiment = {
            "name": "nasa_tlx_plus_experiment_0008",
            "scenario_id": m1_scenario_id,
            "parameters": {
                "force": m1.optimal_force_n,
                "force_direction": 1,
                "experiment_type": "nasa_tlx_plus", 
                "experiment_method": m1.method, 
                "nasa_tlx_needed": True,
                "extra_nasa_tlx_questions": True,
                "flag": m1.flag,
                "details": m1_details,
            },
        }

        m2 = optimal_force_quadratic(tlx_performance, max_force_n)
        m2_scenario_id = f"9_green_line_force_left_{int(m2.optimal_force_n)}"
        m2_scenario = create_new_scenario(int(m2.optimal_force_n),m2_scenario_id)
        with open(f"{self.scenarios_folder}{m2_scenario_id}.yaml", "w") as file:
            yaml.dump(m2_scenario, file)
        m2_details = m2.details
        m2_details["force_cap_n"] = m2.force_cap_n
        m2_details["raw_crossing_n"] = m2.raw_crossing_n
        m2_details["within_device_window"] = m2.within_device_window
        m2_details["coefficients"] = m2.coefficients
        m2_details["fit_rmse"] = m2.fit_rmse
        m2_experiment = {
                    "name": "nasa_tlx_plus_experiment_0009",
                    "scenario_id": m2_scenario_id,
                    "parameters": {
                        "force": m2.optimal_force_n,
                        "force_direction": 1,
                        "experiment_type": "nasa_tlx_plus", 
                        "experiment_method": m2.method, 
                        "nasa_tlx_needed": True,
                        "extra_nasa_tlx_questions": True,
                        "flag": m2.flag,
                        "details": m2_details,
                    },
                }
        
        m3 = optimal_force_quadratic_maxforce(tlx_performance, max_force_n)
        m3_scenario_id = f"10_green_line_force_left_{int(m3.optimal_force_n)}"
        m3_scenario = create_new_scenario(int(m3.optimal_force_n), m3_scenario_id)
        with open(f"{self.scenarios_folder}{m3_scenario_id}.yaml", "w") as file:
            yaml.dump(m3_scenario, file)
        m3_details = m3.details
        m3_details["force_cap_n"] = m3.force_cap_n
        m3_details["raw_crossing_n"] = m3.raw_crossing_n
        m3_details["within_device_window"] = m3.within_device_window
        m3_details["coefficients"] = m3.coefficients
        m3_details["fit_rmse"] = m3.fit_rmse
        m3_experiment = {
                    "name": "nasa_tlx_plus_experiment_0010",
                    "scenario_id": m3_scenario_id,
                    "parameters": {
                        "force": m3.optimal_force_n,
                        "force_direction": 1,
                        "experiment_type": "nasa_tlx_plus", 
                        "experiment_method": m3.method, 
                        "nasa_tlx_needed": True,
                        "extra_nasa_tlx_questions": True,
                        "flag": m3.flag,
                        "details": m3_details,
                    },
                }
        
        m4 = optimal_force_sensor_proxy(timeseries_features, max_force_n)
        m4_scenario_id = f"11_green_line_force_left_{int(m4.optimal_force_n)}"
        m4_scenario = create_new_scenario(int(m4.optimal_force_n), m4_scenario_id)
        with open(f"{self.scenarios_folder}{m4_scenario_id}.yaml", "w") as file:
            yaml.dump(m4_scenario, file)
        m4_details = m4.details
        m4_details["force_cap_n"] = m4.force_cap_n
        m4_details["raw_crossing_n"] = m4.raw_crossing_n
        m4_details["within_device_window"] = m4.within_device_window
        m4_details["coefficients"] = m4.coefficients
        m4_details["fit_rmse"] = m4.fit_rmse
        m4_experiment = {
                    "name": "nasa_tlx_plus_experiment_0011",
                    "scenario_id": m4_scenario_id,
                    "parameters": {
                        "force": m4.optimal_force_n,
                        "force_direction": 1,
                        "experiment_type": "nasa_tlx_plus", 
                        "experiment_method": m4.method, 
                        "nasa_tlx_needed": True,
                        "extra_nasa_tlx_questions": True,
                        "flag": m4.flag,
                        "details": m4_details,
                    },
                }

        self.nasa_tlx_plus_experiments = [m1_experiment, m2_experiment, m3_experiment, m4_experiment]
        with open("/home/docker/ros_ws/data/experiments_user_study_difficulty_nasa_tlx_plus.yaml", "w") as file:
            yaml.dump({exp["name"]: exp for exp in self.nasa_tlx_plus_experiments}, file)

    def initialize_bayesian_optimization_experiments(self):
        self.optimization_objective = "tlx_performance"

        # load existing experiments from the backend to initialize the Bayesian optimization process
        self.initial_bo_data = []
        experiments_response = requests.get(
            f"{self.BACKEND_URL}/participants/{self.participant_id}/experiments", 
            headers={'accept': 'application/json'}
        )
        if experiments_response.status_code == 200:
            experiments = experiments_response.json()
        else:
            self.get_logger().info(f"Error: {experiments_response.status_code}, {experiments_response.text}")
        for experiment in experiments:
            trial_data = (
                {"virtual_force": experiment["force"]},
                {self.optimization_objective: experiment["scores"]["performance"]},
            )
            self.initial_bo_data.append(trial_data)
        self.get_logger().info(f"Loaded initial values: {self.initial_bo_data}")
        
        # Init BO model
        generation_strategy = construct_generation_strategy(
            node_name="BoTorch w/ Custom Components",
            num_exploration=self.number_of_exploration_experiments,
        )
        register_acquisition_function(qNegIntegratedPosteriorVariance)
        register_acquisition_function(qLogNoisyExpectedImprovement)
        self.client = Client()
        virtual_force = RangeParameterConfig(
            name="virtual_force", parameter_type="int", bounds=(0, 100)
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
            self.optimization_objective  # this name is used during the optimization loop
        )
        objective = (
            f"{first_metric_name}"  # minimization is specified by the negative sign
        )
        self.client.configure_optimization(objective=objective)
        self.client.set_generation_strategy(
            generation_strategy=generation_strategy,
        )
        # Add the initial data to the experiment
        for parameters, data in self.initial_bo_data:
            trial_index = self.client.attach_trial(parameters=parameters)
            self.client.complete_trial(trial_index=trial_index, raw_data=data)

        self.bayesian_optimization_experiments = []
        next_experiment = 13
        for i in range(self.number_of_exploration_experiments):
            self.bayesian_optimization_experiments.append(f"bayesian_optimization_exploration_experiment_{next_experiment:04d}")
            next_experiment += 1
        for i in range(self.number_of_exploitation_experiments):
            self.bayesian_optimization_experiments.append(f"bayesian_optimization_exploitation_experiment_{next_experiment:04d}")
            next_experiment += 1
        self.bo_index = 12

    def get_next_bayesian_optimization_experiment(self, experiment_name):
        next_trial = self.client.get_next_trials(max_trials=1)
        next_index, next_parameters = next(iter(next_trial.items()))
        self.get_logger().info(
            f"Next suggestion:\n - index: {next_index}, \n - parameters: {next_parameters}"
        )
        self.bo_index += 1
        scenario_id = f"{self.bo_index}_green_line_force_left_{next_parameters['virtual_force']}"
        new_scenario = create_new_scenario(
            next_parameters["virtual_force"], scenario_id
        )
        with open(f"{self.scenarios_folder}{scenario_id}.yaml", "w") as file:
            yaml.dump(new_scenario, file)
        method = (
            "exploration"
            if "exploration" in experiment_name
            else "exploitation"
        )
        experiment = {
            "name": experiment_name,
            "scenario_id": scenario_id,
            "parameters": {
                "force": next_parameters["virtual_force"],
                "force_direction": 1,
                "experiment_type": "bayesian_optimization", 
                "experiment_method": method, 
                "nasa_tlx_needed": True,
                "extra_nasa_tlx_questions": False,
                "flag": None,
                "details": {},
            },
        }
        return experiment

    def update_callback(self, request, response):

        self.get_logger().info("Incoming request...")

        self.get_user_and_partisipant_id_from_study_status()

        next_experiment = None
        # seting status of last experiment to ready
        self.get_logger().info("Setting last experiment status to ready...")
        self.set_last_experiment_status_to_ready()

        # check if there is a bag file for the current study status
        bag_file_path = self.check_if_bag_file_exists(self.study_status)
        if not bag_file_path:
            response.success = False
            response.message = (
                "Can not find bag file path or there are two of them!"
            )
            return response
        self.get_logger().info(f"Bag file path: {bag_file_path}")

        if self.familiarization_mode:
            if not self.familiarization_experiments:
                self.familiarization_mode = False
                self.standard_study_mode = True
                self.get_logger().info("All familiarization experiments completed. Switching to standard study mode.")
                with open("/home/docker/ros_ws/data/experiments_user_study_difficulty_standard.yaml", "r") as file:
                    experiments_dict = yaml.safe_load(file)
                    # Convert YAML dict to list for index-based access
                    self.standard_study_experiments = list(experiments_dict.values())
            else:
                # Get the next experiment from the familiarization experiments list
                next_experiment = self.familiarization_experiments.pop(0)
                self.current_force = next_experiment["parameters"]["force"]

        if self.standard_study_mode:
            self.bag_per_force[self.current_force] = bag_file_path

            if not self.standard_study_experiments:
                self.standard_study_mode = False
                self.nasa_tlx_plus_mode = True
                self.get_logger().info("All standard study experiments completed. Switching to NASA-TLX+ mode.")
                try:
                    print("\n" + "=" * 50)
                    sys.stdout.flush()  # Ensure that the prompt is printed before waiting for input
                    user_info = input("Whaiting for user input. If user finished the questionnaire, press ENTER: ")
                    print("=" * 50 + "\n")
                except Exception as e:
                    self.get_logger().error(f"Failed to capture terminal input: {str(e)}")
                self.create_nasa_tlx_plus_experiments(self.bag_per_force) 
            else:
                # Get the next experiment from the standard study experiments list
                next_experiment = self.standard_study_experiments.pop(0)
                self.current_force = next_experiment["parameters"]["force"]

        if self.nasa_tlx_plus_mode:
            if not self.nasa_tlx_plus_experiments:
                self.nasa_tlx_plus_mode = False
                self.same_force_mode = True
                self.get_logger().info("All NASA-TLX+ experiments completed.")

                new_experiment = {
                    "name": "same_force_check_0012",
                    "scenario_id": "6_green_line_force_left_60",
                    "parameters": {
                        "force": 60,
                        "force_direction": 1,
                        "experiment_type": "standard", 
                        "experiment_method": "default", 
                        "nasa_tlx_needed": True,
                        "extra_nasa_tlx_questions": False,
                    },
                }
                self.same_force_experiments = [new_experiment]

            else:
                # Create Scenario
                next_experiment = self.nasa_tlx_plus_experiments.pop(0)
                self.current_force = next_experiment["parameters"]["force"]

        if self.same_force_mode:
            if not self.same_force_experiments:
                self.same_force_mode = False
                self.bayesian_optimization_mode = True
                self.get_logger().info("All same-force experiments completed.")
                try:
                    print("\n" + "=" * 50)
                    user_info = input("Whaiting for user input. If user finished the questionnaire, press ENTER: ")
                    print("=" * 50 + "\n")
                except Exception as e:
                    self.get_logger().error(f"Failed to capture terminal input: {str(e)}")
                self.initialize_bayesian_optimization_experiments()
            else:
                next_experiment = self.same_force_experiments.pop(0)
                self.current_force = next_experiment["parameters"]["force"]

        if self.bayesian_optimization_mode:
            if self.bayesian_optimization_experiments:
                experiment_name = self.bayesian_optimization_experiments.pop(0)
                next_experiment = self.get_next_bayesian_optimization_experiment(experiment_name)
                self.current_force = next_experiment["parameters"]["force"]
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

        if next_experiment["parameters"]["nasa_tlx_needed"]:
            # create next experiment in the database
            data = {
                "details": next_experiment.get("details", {}),
                "experiment_name": next_experiment["name"],
                "status": "waiting",
                "extra_step": next_experiment["parameters"]["extra_nasa_tlx_questions"],
                "force": next_experiment["parameters"]["force"],
                "type": next_experiment["parameters"]["experiment_type"],
                "method": next_experiment["parameters"]["experiment_method"],
                "flag": next_experiment.get("flag", ""),
                "scenario_id": next_experiment["scenario_id"],
            }
            self.post_new_experiment(data)

        parameters_lines = [
            f"{key}: {value}"
            for key, value in next_experiment["parameters"].items()
        ]
        self.get_logger().info(
            "New scenario parameters are: "
            f"\nScenario name: {next_experiment['name']}"
            f"\n{chr(10).join(parameters_lines)}"
        )

        response.success = True
        scenario_id = next_experiment['scenario_id']
        response.message = scenario_id
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
