import random

import numpy as np
import pandas as pd
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.msg import get_types_from_msg


class MessageReader:

    def __init__(self):

        self.readers = {
            "geometry_msgs/msg/WrenchStamped": self.read_wrench_stamped,
            "geometry_msgs/msg/Vector3": self.read_vector3,
            "robotrainer_deviation/msg/RobotrainerUserDeviation": self.read_robotrainer_deviation,
            "std_msgs/msg/String": self.read_data,
            "std_msgs/msg/Int32": self.read_data,
        }

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

    def read_vector3(self, topic, msg):

        record = {
            "topic": topic,
            "x": msg.x,
            "y": msg.y,
            "z": msg.z,
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
            start_index = self.df.apply(pd.Series.first_valid_index)['resulting_force_x']
            end_index = self.df.apply(pd.Series.last_valid_index)['resulting_force_x']
            force_started = self.df['timestamp'][start_index]
            force_stopped = self.df['timestamp'][end_index]
            virtual_force_df = self.df.drop(self.df.loc[(self.df['timestamp']<force_started) | (self.df['timestamp']>force_stopped)].index)
            return virtual_force_df
        else:
            return self.df

    def create_df_without_virtual_force(self):
        force_started = self.df[self.df["status"] == "force 0 started"]["timestamp"]
        force_started.reset_index(drop=True, inplace=True)
        df_without_virtual_force = self.df.drop(
            self.df.loc[(self.df["timestamp"] >= force_started[0].item())].index
        )
        return df_without_virtual_force

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

    def __init__(self, selected_topics):
        self.selected_topics = selected_topics
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
            bag_data_df["x"] = np.where(
                bag_data_df["topic"]
                == "/base/virtual_forces/modalities_debug/resulting_force",
                bag_data_df["x"] * 100,
                bag_data_df["x"],
            )
            bag_data_df["y"] = np.where(
                bag_data_df["topic"]
                == "/base/virtual_forces/modalities_debug/resulting_force",
                bag_data_df["y"] * 100,
                bag_data_df["y"],
            )
            bag_data_df["z"] = np.where(
                bag_data_df["topic"]
                == "/base/virtual_forces/modalities_debug/resulting_force",
                bag_data_df["z"] * 100,
                bag_data_df["z"],
            )
            bag_data_df['resulting_force'] = np.where(
                bag_data_df['topic'] == '/base/virtual_forces/modalities_debug/resulting_force', 
                np.sqrt(bag_data_df['resulting_force_x'] ** 2 + bag_data_df['resulting_force_y'] ** 2), 
                np.nan
            )
        except:
            pass
        try:
            bag_data_df['wrench_force'] = np.nan
            bag_data_df['wrench_force'] =  np.where(
                bag_data_df['topic'] == '/base/output_data', 
                np.sqrt(bag_data_df['wrench_force_x'] ** 2 + bag_data_df['wrench_force_y'] ** 2), 
                np.nan
            )
        except:
            pass

        return bag_data_df


def create_new_scenario(existing_scenario, new_force):
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
    start = 50
    end = len(existing_scenario["path"]["points"])
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
    print(
        f"new_point: {random_location},\nnew area: {new_area},\nnew margin: {new_margin}"
    )

    existing_scenario["force"]["data"][force_name]["area"]["x"] = new_area.tolist()[0]
    existing_scenario["force"]["data"][force_name]["area"]["y"] = new_area.tolist()[1]
    existing_scenario["force"]["data"][force_name]["area"]["z"] = new_area.tolist()[2]

    existing_scenario["force"]["data"][force_name]["arrow"]["x"] = new_arrow.tolist()[0]
    existing_scenario["force"]["data"][force_name]["arrow"]["y"] = new_arrow.tolist()[1]
    existing_scenario["force"]["data"][force_name]["arrow"]["z"] = new_arrow.tolist()[2]

    existing_scenario["force"]["data"][force_name]["margin"]["x"] = new_margin.tolist()[0]
    existing_scenario["force"]["data"][force_name]["margin"]["y"] = new_margin.tolist()[1]
    existing_scenario["force"]["data"][force_name]["margin"]["z"] = new_margin.tolist()[2]

    return existing_scenario