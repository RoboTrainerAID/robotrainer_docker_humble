from std_srvs.srv import Trigger

import rclpy
from rclpy.node import Node


class BayesianOptimizationNode(Node):

    def __init__(self):
        super().__init__('robotrainer_bayesian_optimization')
        self.srv = self.create_service(Trigger, '/robotrainer_bayesian_optimization/update', self.update_callback)

        # Init BO model

        self.get_logger().info('Successfully started robotrainer_bayesian_optimization')


    def update_callback(self, request, response):
        self.get_logger().info('Incoming request...')

        # Load Bag
        # Process Data
        # Update BO model
        # Create Scenario
        # Save yaml

        scenario_name = 'force_XXX.yaml'
        scenario_path_in_container = '/home/docker/ros_ws/data/scenarios/' + scenario_name
        scenario_path_on_robotrainer = '/home/robotrainer/workspace/docker/robotrainer_bayesian_optimization/data/scenarios/' + scenario_name

        response.success = True
        response.message = scenario_path_on_robotrainer

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


if __name__ == '__main__':
    main()