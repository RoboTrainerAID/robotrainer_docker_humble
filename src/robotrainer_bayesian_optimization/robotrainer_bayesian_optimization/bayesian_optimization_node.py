import rclpy
import yaml
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.modelbridge.registry import Generators
from rclpy.node import Node
from std_srvs.srv import Trigger

from .bayesian_optimisation import construct_generation_strategy, qUpperConfidenceBound
from .utils import BagReader, DataFrameProcessing, create_new_scenario


class BayesianOptimizationNode(Node):

    def __init__(self):
        super().__init__("robotrainer_bayesian_optimization")
        self.srv = self.create_service(
            Trigger, "/robotrainer_bayesian_optimization/update", self.update_callback
        )

        # Load initial data from config file
        self.config_file_path = "/home/docker/ros_ws/src/robotrainer_bayesian_optimization/robotrainer_bayesian_optimization/config.yaml"
        self.initial_data = {}
        with open(self.config_file_path, "r") as file:
            config = yaml.safe_load(file)
            self.initial_data = config["initial_data_files"]

        self.preexisting_trials = []
        self.selected_topics = [
            "/base/output_data",
            "/base/virtual_forces/modalities_debug/resulting_force",
            "/base/virtual_forces/modalities_debug/status",
            "/robotrainer_deviation/robotrainer_deviation",
            "/biosensors/polar_oh1/hr",
        ]
        self.bag_reader = BagReader(selected_topics=self.selected_topics)
        for key, value in self.initial_data.items():
            self.get_logger().info(f"Loaded initial data: {key} -> {value}")
            bag_data_df = self.bag_reader.read_bag_file(value)
            df_processing = DataFrameProcessing(bag_data_df)
            bag_data_df_virtual_force = df_processing.create_virtual_force_df()
            virtual_force_df_processing = DataFrameProcessing(bag_data_df_virtual_force)
            mean_values = virtual_force_df_processing.mean_values()
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
            # Attach the parameterization to the Client as a trial and immediately complete it with the preexisting data
            trial_index = self.client.attach_trial(parameters=parameters)
            self.client.complete_trial(trial_index=trial_index, raw_data=data)

        trials = self.client.get_next_trials(max_trials=1)
        self.next_index, self.next_parameters = next(iter(trials.items()))
        self.get_logger().info(f"Next suggestion:\n - index: {self.next_index}, \n - parameters: {self.next_parameters}")

        self.get_logger().info("Successfully started robotrainer_bayesian_optimization")

    def update_callback(self, request, response):
        self.get_logger().info("Incoming request...")

        bag_file_path = ""
        with open(self.config_file_path, "r") as file:
            config = yaml.safe_load(file)
            bag_file_path = config["default_bag_file"]

        if not bag_file_path:
            response.success = False
            response.message = "No bag file path provided."
            return response
        self.get_logger().info(f"Bag file path: {bag_file_path}")

        # Read selected topics from bag file in pandas DataFrame
        bag_data_df = self.bag_reader.read_bag_file(bag_file_path)

        # Process Data
        df_processing = DataFrameProcessing(bag_data_df)
        bag_data_df_virtual_force = df_processing.create_virtual_force_df()
        virtual_force_df_processing = DataFrameProcessing(bag_data_df_virtual_force)
        mean_values = virtual_force_df_processing.mean_values()

        # Update BO model
        parameters, raw_data = (
                self.next_parameters,
                {"wrench_force": mean_values["wrench_force_y"]},
            )
        trial_index = self.client.attach_trial(parameters=parameters)
        self.client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        self.get_logger().info(
            f"Completed trial {trial_index} with parameters {parameters} and data {raw_data}"
        )
        trials = self.client.get_next_trials(max_trials=1)
        self.next_index, self.next_parameters = next(iter(trials.items()))
        self.get_logger().info(f"Next suggestion:\n - index: {self.next_index}, \n - parameters: {self.next_parameters}")
        # Create Scenario
        default_scenario_path = "/home/docker/ros_ws/data/scenarios/default_scenario.yaml"
        with open(default_scenario_path, "r") as file:
            scenario_data = yaml.safe_load(file)

        new_scenario = create_new_scenario(scenario_data, self.next_parameters["virtual_force"])

        # Save yaml
        scenario_name = f"trial_{self.next_index}_force_{self.next_parameters['virtual_force']}.yaml"
        scenario_path_in_container = (
            "/home/docker/ros_ws/data/scenarios/" + scenario_name
        )
        scenario_path_on_robotrainer = (
            "/home/robotrainer/workspace/docker/robotrainer_bayesian_optimization/data/scenarios/"
            + scenario_name
        )

        with open(scenario_path_in_container, "w") as file:
            yaml.dump(new_scenario, file)
        
        response.success = True
        response.message = scenario_path_on_robotrainer
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
