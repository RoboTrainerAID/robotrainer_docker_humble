import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.msg import get_types_from_msg


class MessageReader:

    def __init__(self):

        self.readers = {
            "geometry_msgs/msg/WrenchStamped": self.read_wrench_stamped, # output_data topic
            "geometry_msgs/msg/TwistStamped": self.read_twist_stamped, # velocity_output topic
            "geometry_msgs/msg/Vector3": self.read_vector3, # position, velocity_in, velocity_out, resulting_velocity, resulting_force topics
            "robotrainer_deviation/msg/RobotrainerUserDeviation": self.read_robotrainer_deviation, # robotrainer_deviation topic
            "std_msgs/msg/String": self.read_data, # status topic
            "std_msgs/msg/Int32": self.read_data, # hr topic
            "visualization_msgs/msg/Marker": self.read_marker, # visualization_marker topic
            "ipr_helpers/msg/Pose2DStamped": self.read_pose2d_stamped, # mobile_robot_pose topic
        }

    def read_pose2d_stamped(self, topic, msg):

        record = {
            "topic": topic,
            "pose_x": msg.pose.x,
            "pose_y": msg.pose.y,
            "pose_theta": msg.pose.theta,
        }
        return record

    def read_marker(self, topic, msg):

        record = {
            "topic": topic,
            "marker_ns": msg.ns,
            "marker_id": msg.id,
            "marker_type": msg.type,
            "marker_action": msg.action,
            "marker_scale_x": msg.scale.x,
            "marker_scale_y": msg.scale.y,
            "marker_scale_z": msg.scale.z,
            "marker_color_r": msg.color.r,
            "marker_color_g": msg.color.g,
            "marker_color_b": msg.color.b,
            "marker_color_a": msg.color.a,
            "marker_pose_position": msg.pose.position,
            "marker_pose_orientation_x": msg.pose.orientation.x,
            "marker_pose_orientation_y": msg.pose.orientation.y,
            "marker_pose_orientation_z": msg.pose.orientation.z,
            "marker_pose_orientation_w": msg.pose.orientation.w,
        }
        return record

    def read_wrench_stamped(self, topic, msg):

        record = {
            "topic": topic,
            "wrench_force_x": msg.wrench.force.x,
            "wrench_force_y": msg.wrench.force.y,
            "wrench_force_z": msg.wrench.force.z,
            "wrench_torque_x": msg.wrench.torque.x,
            "wrench_torque_y": msg.wrench.torque.y,
            "wrench_torque_z": msg.wrench.torque.z,
        }
        return record

    def read_twist_stamped(self, topic, msg):

        record = {
            "topic": topic,
            "twist_linear_x": msg.twist.linear.x,
            "twist_linear_y": msg.twist.linear.y,
            "twist_linear_z": msg.twist.linear.z,
            "twist_angular_x": msg.twist.angular.x,
            "twist_angular_y": msg.twist.angular.y,
            "twist_angular_z": msg.twist.angular.z,
        }
        return record

    def read_vector3(self, topic, msg):

        key = topic.split("/")[-1]
        record = {
            "topic": topic,
            key + "_x": msg.x,
            key + "_y": msg.y,
            key + "_z": msg.z,
        }
        return record

    def read_robotrainer_deviation(self, topic, msg):

        record = {
            "topic": topic,
            "front": msg.front,
            "left": msg.left,
            "right": msg.right,
        }
        return record

    def read_data(self, topic, msg):

        key = topic.split("/")[-1]
        record = {
            "topic": topic,
            key: msg.data,
        }
        return record

    def read_message(self, msgtype, topic, msg):

        try:
            record = self.readers[msgtype](topic, msg)
        except KeyError:
            print(
                f"Cannot read the message of type {msgtype}. This message type has not yet been implemented."
            )
            record = {}
        except Exception as e:
            print(f"Cannot read the message of type {msgtype}. The error is {e}")
            record = {}
        return record


class DataFrameProcessing:

    def __init__(self, df):
        self.df = df
        try:
            self.status_df = self.df[self.df["status"].notna()].copy()
        except:
            pass

    def create_mean_df(self):
        try:
            mean_df = self.df.drop(columns=["status"])
        except:
            mean_df = self.df.copy()
        mean_df["timestamp"] = mean_df["timestamp"] // 1_000_000_000
        mean_df = mean_df.groupby(["topic", "timestamp"]).mean()
        mean_df.reset_index(inplace=True)
        mean_df["timestamp"] = (mean_df["timestamp"] + 0.5) * 1_000_000_000
        return mean_df

    def create_median_df(self):
        try:
            median_df = self.df.drop(columns=["status"])
        except:
            median_df = self.df.copy()
        median_df["timestamp"] = median_df["timestamp"] // 1_000_000_000
        median_df = median_df.groupby(["topic", "timestamp"]).median()
        median_df.reset_index(inplace=True)
        median_df["timestamp"] = (median_df["timestamp"] + 0.5) * 1_000_000_000
        return median_df

    def create_virtual_force_df(self):
        if "resulting_force_x" in self.df.columns:
            start_index = self.df.apply(pd.Series.first_valid_index)[
                "resulting_force_x"
            ]
            end_index = self.df.apply(pd.Series.last_valid_index)["resulting_force_x"]
            force_started = self.df["timestamp"][start_index]
            force_stopped = self.df["timestamp"][end_index]
            virtual_force_df = self.df.drop(
                self.df.loc[
                    (self.df["timestamp"] < force_started)
                    | (self.df["timestamp"] > force_stopped)
                ].index
            )
            return virtual_force_df
        else:
            return self.df

    def create_df_without_virtual_force(self):
        if "resulting_force_x" in self.df.columns:
            start_index = self.df.apply(pd.Series.first_valid_index)[
                "resulting_force_x"
            ]
            force_started = self.df["timestamp"][start_index]
            df_without_virtual_force = self.df.drop(
                self.df.loc[(self.df["timestamp"] >= force_started)].index
            )
            return df_without_virtual_force
        else:
            return self.df

    def mean_values(self):
        mean_values = {}
        columns = self.df.columns
        for key in columns:
            try:
                mean_values[key] = self.df[key].mean()
            except:
                continue
        return mean_values

    def median_values(self):
        median_values = {}
        columns = self.df.columns
        for key in columns:
            try:
                median_values[key] = self.df[key].median()
            except:
                continue
        return median_values


class BagReader:

    ROBOTRAINER_DEVIATION_PATHINDEX_MSG = """
    int16 front
    int16 left
    int16 right
    """
    ROBOTRAINER_DEVIATION_ROBOTRAINERUSERDEVIATION_MSG = """
    float64 front
    float64 left
    float64 right
    """
    IPR_HELPERS_POSE2DSTAMPED_MSG = """
    std_msgs/Header header
    geometry_msgs/Pose2D pose
    """
    selected_topics = [
        "/base/output_data", # WrenchStamped
        "/base/virtual_forces/modalities_debug/resulting_force", # Vector3
        "/base/virtual_forces/modalities_debug/status", # String
        "/robotrainer_deviation/robotrainer_deviation", # RobotrainerUserDeviation
        "/biosensors/polar_oh1/hr", # Int32
        "/base/fts_adaptive_force_controller/debug/velocity_output", # TwistStamped
        "/base/virtual_forces/modalities_debug/velocity_in", # Vector3
        "/base/virtual_forces/modalities_debug/velocity_out", # Vector3
        "/base/virtual_forces/modalities_debug/resulting_velocity", # Vector3
        # "/scenario_publisher/visualization_marker", # VisualizationMarker
        "/base/virtual_forces/modalities_debug/position", # Vector3
        "/mobile_robot_pose", # Pose2DStamped
    ]

    def __init__(self):
        self.message_reader = MessageReader()

        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        self.typestore.register(
            get_types_from_msg(
                self.ROBOTRAINER_DEVIATION_PATHINDEX_MSG,
                "robotrainer_deviation/msg/PathIndex",
            )
        )
        self.typestore.register(
            get_types_from_msg(
                self.ROBOTRAINER_DEVIATION_ROBOTRAINERUSERDEVIATION_MSG,
                "robotrainer_deviation/msg/RobotrainerUserDeviation",
            )
        )
        self.typestore.register(
            get_types_from_msg(
                self.IPR_HELPERS_POSE2DSTAMPED_MSG, "ipr_helpers/msg/Pose2DStamped"
            )
        )

    def read_bag_file(self, bag_file_path):
        bag_data = []
        with Reader(bag_file_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic in self.selected_topics:
                    msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                    record = self.message_reader.read_message(
                        connection.msgtype, connection.topic, msg
                    )
                    record["timestamp"] = timestamp
                    bag_data.append(record)
        bag_data_df = pd.DataFrame(bag_data)
        try:
            bag_data_df["resulting_force_x"] = np.where(
                bag_data_df["topic"]
                == "/base/virtual_forces/modalities_debug/resulting_force",
                bag_data_df["resulting_force_x"] * 100,
                bag_data_df["resulting_force_x"],
            )
            bag_data_df["resulting_force_y"] = np.where(
                bag_data_df["topic"]
                == "/base/virtual_forces/modalities_debug/resulting_force",
                bag_data_df["resulting_force_y"] * 100,
                bag_data_df["resulting_force_y"],
            )
            bag_data_df["resulting_force_z"] = np.where(
                bag_data_df["topic"]
                == "/base/virtual_forces/modalities_debug/resulting_force",
                bag_data_df["resulting_force_z"] * 100,
                bag_data_df["resulting_force_z"],
            )
            bag_data_df["resulting_force"] = np.where(
                bag_data_df["topic"]
                == "/base/virtual_forces/modalities_debug/resulting_force",
                np.sqrt(
                    bag_data_df["resulting_force_x"] ** 2
                    + bag_data_df["resulting_force_y"] ** 2
                ),
                np.nan,
            )
        except:
            pass
        try:
            bag_data_df["twist_linear"] = np.nan
            bag_data_df["twist_linear"] = np.where(
                bag_data_df["topic"]
                == "/base/fts_adaptive_force_controller/debug/velocity_output",
                np.sqrt(
                    bag_data_df["twist_linear_x"] ** 2
                    + bag_data_df["twist_linear_y"] ** 2
                ),
                np.nan,
            )
        except:
            pass
        try:
            bag_data_df["full_force"] = np.nan
            bag_data_df["full_force"] = np.where(
                bag_data_df["topic"] == "/base/output_data",
                np.sqrt(
                    bag_data_df["wrench_force_x"] ** 2
                    + bag_data_df["wrench_force_y"] ** 2
                    + (bag_data_df["wrench_torque_z"] / 0.35) ** 2
                ),
                np.nan,
            )
        except:
            pass
        try:
            bag_data_df["full_velocity"] = np.nan
            bag_data_df["full_velocity"] = np.where(
                bag_data_df["topic"]
                == "/base/fts_adaptive_force_controller/debug/velocity_output",
                np.sqrt(
                    bag_data_df["twist_linear_x"] ** 2
                    + bag_data_df["twist_linear_y"] ** 2
                    + (bag_data_df["twist_angular_z"] * 0.35) ** 2
                ),
                np.nan,
            )
        except:
            pass
        return bag_data_df


def create_new_scenario(new_force, scenario_name, direction=1):

    default_scenario_path = "/home/docker/ros_ws/data/scenarios/default_scenario.yaml"
    with open(default_scenario_path, "r") as file:
        existing_scenario = yaml.safe_load(file)

    force_name = existing_scenario["force"]["config"]["force_names"][0]

    area = np.array(
        [
            existing_scenario["force"]["data"][force_name]["area"]["x"],
            existing_scenario["force"]["data"][force_name]["area"]["y"],
            existing_scenario["force"]["data"][force_name]["area"]["z"],
        ]
    )
    arrow = np.array(
        [
            existing_scenario["force"]["data"][force_name]["arrow"]["x"],
            existing_scenario["force"]["data"][force_name]["arrow"]["y"],
            existing_scenario["force"]["data"][force_name]["arrow"]["z"],
        ]
    )
    margin = np.array(
        [
            existing_scenario["force"]["data"][force_name]["margin"]["x"],
            existing_scenario["force"]["data"][force_name]["margin"]["y"],
            existing_scenario["force"]["data"][force_name]["margin"]["z"],
        ]
    )

    radius = margin - area

    original_length = np.linalg.norm(arrow)
    scale = new_force / original_length
    new_arrow = arrow * scale
    np.set_printoptions(suppress=True, precision=17)
    start = 60
    end = len(existing_scenario["path"]["points"]) - 60
    random_location = existing_scenario["path"]["points"][random.randint(start, end)]
    new_area = existing_scenario["path"][random_location]
    new_area = np.array(
        [
            existing_scenario["path"][random_location]["x"],
            existing_scenario["path"][random_location]["y"],
            existing_scenario["path"][random_location]["z"],
        ]
    )
    new_margin = new_area + radius

    existing_scenario["force"]["data"][force_name]["area"]["x"] = new_area.tolist()[0]
    existing_scenario["force"]["data"][force_name]["area"]["y"] = new_area.tolist()[1]
    existing_scenario["force"]["data"][force_name]["area"]["z"] = new_area.tolist()[2]

    existing_scenario["force"]["data"][force_name]["arrow"]["x"] = new_arrow.tolist()[0] * direction
    existing_scenario["force"]["data"][force_name]["arrow"]["y"] = new_arrow.tolist()[1] * direction
    existing_scenario["force"]["data"][force_name]["arrow"]["z"] = new_arrow.tolist()[2] * direction

    existing_scenario["force"]["data"][force_name]["margin"]["x"] = new_margin.tolist()[0]
    existing_scenario["force"]["data"][force_name]["margin"]["y"] = new_margin.tolist()[1]
    existing_scenario["force"]["data"][force_name]["margin"]["z"] = new_margin.tolist()[2]

    scenario_path_on_robotrainer = (
        "/home/robotrainer/workspace/docker/robotrainer_docker_bayesian_optimization/data/scenarios/"
        + scenario_name
    )
    existing_scenario["scenario"] = scenario_path_on_robotrainer
    scenario_id = "scenario_id" + datetime.now().strftime("%Y%m%d%H%M")
    existing_scenario["scenario_id"] = scenario_id

    return existing_scenario


def get_mean_values_from_bag(bag_file_path, bag_reader):
    # Read selected topics from bag file in pandas DataFrame
    bag_data_df = bag_reader.read_bag_file(bag_file_path)

    # Process Data
    df_processing = DataFrameProcessing(bag_data_df)
    bag_data_df_virtual_force = df_processing.create_virtual_force_df()
    force_started_timestamp = bag_data_df_virtual_force["timestamp"].min()
    force_stopped_timestamp = bag_data_df_virtual_force["timestamp"].max()
    duration = (force_stopped_timestamp - force_started_timestamp) * 1e-9
    try:
        mean_df = bag_data_df_virtual_force.drop(
            columns=[
                "status",
            ]
        )
    except:
        mean_df = bag_data_df_virtual_force.copy()
    mean_df = mean_df.drop(
        columns=[
            "marker_ns",
            "marker_id",
            "marker_type",
            "marker_action",
            "marker_scale_x",
            "marker_scale_y",
            "marker_scale_z",
            "marker_color_r",
            "marker_color_g",
            "marker_color_b",
            "marker_color_a",
            "marker_pose_position",
            "marker_pose_orientation_x",
            "marker_pose_orientation_y",
            "marker_pose_orientation_z",
            "marker_pose_orientation_w",
            "topic",
        ]
    )
    mean_df["timestamp"] = mean_df["timestamp"] // 100_000_000
    mean_df = mean_df.groupby(["timestamp"]).mean()
    mean_df.reset_index(inplace=True)
    mean_df["timestamp"] = (mean_df["timestamp"]) * 100_000_000
    mean_df["power"] = (
        mean_df["wrench_force_x"] * mean_df["twist_linear_x"]
        + mean_df["wrench_force_y"] * mean_df["twist_linear_y"]
        + mean_df["wrench_torque_z"] * mean_df["twist_angular_z"]
    ).fillna(0)
    mean_df["power_velocity_in"] = (
        mean_df["wrench_force_x"] * mean_df["velocity_in_x"]
        + mean_df["wrench_force_y"] * mean_df["velocity_in_y"]
        + mean_df["wrench_torque_z"] * mean_df["velocity_in_z"]
    ).fillna(0)
    dt = 0.1
    total_work = np.trapz(mean_df["power"], dx=dt)
    total_work_velocity_in = np.trapz(mean_df["power_velocity_in"], dx=dt)
    positive_work = np.trapz(np.maximum(mean_df["power"], 0), dx=dt)
    negative_work = np.trapz(np.minimum(mean_df["power"], 0), dx=dt)
    absolute_work = np.trapz(np.abs(mean_df["power"]), dx=dt)
    absolute_work_velocity_in = np.trapz(np.abs(mean_df["power_velocity_in"]), dx=dt)
    positive_mean_power = mean_df.loc[mean_df["power"] > 0, "power"].mean()
    positive_mean_power = 0 if np.isnan(positive_mean_power) else positive_mean_power
    negative_mean_power = mean_df.loc[mean_df["power"] <= 0, "power"].mean()
    negative_mean_power = 0 if np.isnan(negative_mean_power) else negative_mean_power
    mean_power = mean_df["power"].mean()
    mean_power_velocity_in = mean_df["power_velocity_in"].mean()
    virtual_force_df_processing = DataFrameProcessing(bag_data_df_virtual_force)
    mean_values = virtual_force_df_processing.mean_values()

    bag_data_df_deviation = (
        bag_data_df[
            bag_data_df["topic"] == "/robotrainer_deviation/robotrainer_deviation"
        ]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    deviation_front_sum = 0
    for i in range(0, len(bag_data_df_deviation) - 2):
        deviation_front_sum += (
            (
                (
                    bag_data_df_deviation["front"][i]
                    + bag_data_df_deviation["front"][i + 1]
                )
                / 2
            )
            * (
                bag_data_df_deviation["timestamp"][i + 1]
                - bag_data_df_deviation["timestamp"][i]
            )
            * 1e-9
        )
    mean_values["duration"] = duration
    mean_values["total_work"] = total_work
    mean_values["positive_work"] = positive_work
    mean_values["negative_work"] = negative_work
    mean_values["absolute_work"] = absolute_work
    mean_values["positive_mean_power"] = positive_mean_power
    mean_values["negative_mean_power"] = negative_mean_power
    mean_values["mean_power"] = mean_power
    mean_values["mean_power_velocity_in"] = mean_power_velocity_in
    mean_values["deviation_front_sum"] = deviation_front_sum
    mean_values["absolute_work_velocity_in"] = absolute_work_velocity_in
    mean_values["total_work_velocity_in"] = total_work_velocity_in
    return mean_values


def get_median_values_from_bag(bag_file_path, bag_reader):
    # Read selected topics from bag file in pandas DataFrame
    bag_data_df = bag_reader.read_bag_file(bag_file_path)

    # Process Data
    df_processing = DataFrameProcessing(bag_data_df)
    bag_data_df_virtual_force = df_processing.create_virtual_force_df()
    force_started_timestamp = bag_data_df_virtual_force["timestamp"].min()
    force_stopped_timestamp = bag_data_df_virtual_force["timestamp"].max()
    duration = (force_stopped_timestamp - force_started_timestamp) * 1e-9
    virtual_force_df_processing = DataFrameProcessing(bag_data_df_virtual_force)
    median_values = virtual_force_df_processing.median_values()
    median_values["duration"] = duration
    return median_values


def get_max_values_from_bag(bag_file_path, bag_reader):
    # Read selected topics from bag file in pandas DataFrame
    bag_data_df = bag_reader.read_bag_file(bag_file_path)

    # Process Data
    df_processing = DataFrameProcessing(bag_data_df)
    bag_data_df_virtual_force = df_processing.create_virtual_force_df()
    virtual_force_df_processing = DataFrameProcessing(bag_data_df_virtual_force)
    bag_data_df_virtual_force_mean_per_second = (
        virtual_force_df_processing.create_mean_df()
    )
    max_values = {}
    columns = bag_data_df_virtual_force_mean_per_second.columns
    for key in columns:
        try:
            max_values[key] = bag_data_df_virtual_force_mean_per_second[key].max()
        except:
            continue
    return max_values


def get_data_root():
    """Get data root path - auto-detects Docker vs local environment."""
    docker_path = Path("/home/docker/ros_ws/data")
    local_path = Path("../../data").resolve()
    
    
    # Use Docker path if it exists, otherwise use local path
    if docker_path.exists():
        return docker_path
    return local_path

def normalize_raw_values(df):

    data_root = get_data_root()
    scenario_file = data_root / "scenarios" / "default_scenario.yaml"
    radius = 0.0
    with open(scenario_file, "r") as file:
        scenario = yaml.safe_load(file)
        force_name = scenario["force"]["config"]["force_names"][0]

        area = np.array(
            [
                scenario["force"]["data"][force_name]["area"]["x"],
                scenario["force"]["data"][force_name]["area"]["y"],
                scenario["force"]["data"][force_name]["area"]["z"],
            ]
        )
        margin = np.array(
            [
                scenario["force"]["data"][force_name]["margin"]["x"],
                scenario["force"]["data"][force_name]["margin"]["y"],
                scenario["force"]["data"][force_name]["margin"]["z"],
            ]
        )
        radius = ((area[0] - margin[0]) ** 2 + (area[1] - margin[1]) ** 2) ** 0.5

    # Flip signs for rows where resulting_force_y < 0
    mask = df["resulting_force_y"] > 0

    df.loc[mask, "resulting_force_z"] *= -1
    df.loc[mask, "wrench_force_y"] *= -1
    df.loc[mask, "wrench_torque_x"] *= -1
    df.loc[mask, "wrench_torque_y"] *= -1
    df.loc[mask, "wrench_torque_z"] *= -1
    df.loc[mask, "twist_angular_z"] *= -1
    df.loc[mask, "twist_linear_y"] *= -1

    offset = 0.49

    # Create deviation_in_force_direction column based on resulting_force_y
    df["deviation_in_force_direction"] = df.apply(
        lambda row: row["left"] if row["resulting_force_y"] > 0 else row["right"],
        axis=1,
    )

    # Create deviation_opposite_force_direction column based on resulting_force_y
    df["deviation_opposite_force_direction"] = df.apply(
        lambda row: row["right"] if row["resulting_force_y"] > 0 else row["left"],
        axis=1,
    )
    # Normalize the deviation columns
    df["deviation_in_force_direction"] = df["deviation_in_force_direction"] - offset
    df["deviation_opposite_force_direction"] = (
        df["deviation_opposite_force_direction"] - offset
    )

    df["deviation_in_force_direction_squared"] = df["deviation_in_force_direction"] ** 2

    df["full_force_diff"] = df["full_force"] - df["full_force"][0]

    df.loc[mask, "resulting_force_y"] *= -1
    return df
