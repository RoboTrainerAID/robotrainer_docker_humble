#!/usr/bin/env python3
"""
ROS2-equivalent of the ROS1 rosbag TF extractor.

Reads /tf_static and /tf from every ROS1 .bag file in a folder, rebuilds the
TF tree, and writes the pose of --source-frame in --target-frame as
PoseStamped messages to a new bag alongside the original (suffix
'_transformed_to_map_frame').

Key differences from the ROS1 version:
  - rosbag.Bag          -> rosbags.rosbag1.Reader  (ros2 bag info/play can't open .bag)
  - tf2_ros.Buffer      -> tf2_py.BufferCore        (no running node needed)
  - set_transform input -> requires geometry_msgs.msg.TransformStamped (ROS2 type),
                           so each rosbags transform is converted before feeding it in

Usage (run after sourcing the workspace):
    ros2 run robotrainer_bayesian_optimization bag_tf_to_map \
        --folder /path/to/bags/ \
        --source-frame base_link \
        --target-frame map
"""

import argparse
import struct
import sys
from pathlib import Path

from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

import rclpy
import rclpy.duration
import rclpy.time
import tf2_py
from geometry_msgs.msg import TransformStamped


def to_ros2_transform(rb_tf) -> TransformStamped:
    """Convert a rosbags TransformStamped to a ROS2 TransformStamped."""
    ts = TransformStamped()
    ts.header.stamp.sec = int(rb_tf.header.stamp.sec)
    ts.header.stamp.nanosec = int(getattr(rb_tf.header.stamp, 'nanosec',
                                          getattr(rb_tf.header.stamp, 'nsec', 0)))
    ts.header.frame_id = str(rb_tf.header.frame_id)
    ts.child_frame_id = str(rb_tf.child_frame_id)
    ts.transform.translation.x = float(rb_tf.transform.translation.x)
    ts.transform.translation.y = float(rb_tf.transform.translation.y)
    ts.transform.translation.z = float(rb_tf.transform.translation.z)
    ts.transform.rotation.x = float(rb_tf.transform.rotation.x)
    ts.transform.rotation.y = float(rb_tf.transform.rotation.y)
    ts.transform.rotation.z = float(rb_tf.transform.rotation.z)
    ts.transform.rotation.w = float(rb_tf.transform.rotation.w)
    return ts


def serialize_pose_stamped(seq, stamp_sec, stamp_nsec, frame_id,
                           x, y, z, qx, qy, qz, qw) -> bytes:
    """Serialize geometry_msgs/PoseStamped to ROS1 binary format."""
    frame_bytes = frame_id.encode('utf-8')
    header = struct.pack('<III', seq, stamp_sec, stamp_nsec)
    header += struct.pack('<I', len(frame_bytes)) + frame_bytes
    pose = struct.pack('<7d', x, y, z, qx, qy, qz, qw)
    return header + pose


def list_tf_frames(bagfile):
    """Return the sorted set of all TF frame IDs seen in the bag."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    frames = set()

    with Reader(bagfile) as bag:
        seen = set()
        for c in bag.connections:
            if c.topic in ('/tf', '/tf_static') and c.msgtype not in seen:
                seen.add(c.msgtype)
                _, msgdef_text = c.msgdef
                typestore.register(get_types_from_msg(msgdef_text, c.msgtype))

        tf_conns = [c for c in bag.connections if c.topic in ('/tf', '/tf_static')]
        for connection, _ts, rawdata in bag.messages(connections=tf_conns):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            for transform in msg.transforms:
                frames.add(str(transform.header.frame_id))
                frames.add(str(transform.child_frame_id))

    return sorted(frames)


def extract_map_frame_pose(bagfile, source_frame, target_frame):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    tf_buffer = tf2_py.BufferCore(rclpy.duration.Duration(seconds=7200.0))
    poses = []

    with Reader(bagfile) as bag:

        # Register types from the bag's own embedded definitions so that
        # the ROS1 binary layout (including the seq field in Header) is correct.
        seen = set()
        for c in bag.connections:
            if c.topic in ('/tf', '/tf_static') and c.msgtype not in seen:
                seen.add(c.msgtype)
                _, msgdef_text = c.msgdef
                typestore.register(get_types_from_msg(msgdef_text, c.msgtype))

        # --- pass 1: load ALL transforms into the buffer first ---
        # Doing lookups only after the full buffer is populated avoids missing
        # poses at the start of the bag (e.g. map->odom not yet published).
        static_conns = [c for c in bag.connections if c.topic == '/tf_static']
        for connection, _ts, rawdata in bag.messages(connections=static_conns):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            for transform in msg.transforms:
                tf_buffer.set_transform_static(to_ros2_transform(transform), 'bag')

        tf_conns = [c for c in bag.connections if c.topic == '/tf']
        lookup_times = []
        for connection, _ts, rawdata in bag.messages(connections=tf_conns):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            for transform in msg.transforms:
                tf_buffer.set_transform(to_ros2_transform(transform), 'bag')
            stamp = msg.transforms[0].header.stamp
            t_sec = int(stamp.sec)
            t_nsec = int(getattr(stamp, 'nanosec', getattr(stamp, 'nsec', 0)))
            lookup_times.append(rclpy.time.Time(seconds=t_sec, nanoseconds=t_nsec))

        # --- pass 2: look up the transform at every recorded /tf timestamp ---
        for lookup_time in lookup_times:
            try:
                result = tf_buffer.lookup_transform_core(
                    target_frame, source_frame, lookup_time
                )
                t = result.transform.translation
                r = result.transform.rotation
                poses.append({
                    't_sec': result.header.stamp.sec,
                    't_nsec': result.header.stamp.nanosec,
                    'x': t.x,
                    'y': t.y,
                    'z': t.z,
                    'qx': r.x,
                    'qy': r.y,
                    'qz': r.z,
                    'qw': r.w,
                })
            except (tf2_py.LookupException,
                    tf2_py.ExtrapolationException,
                    tf2_py.ConnectivityException):
                pass

    return poses


_POSE_STAMPED_MSGDEF = (
    'Header header\n'
    'geometry_msgs/Pose pose\n'
    '================================================================================\n'
    'MSG: std_msgs/Header\n'
    'uint32 seq\n'
    'time stamp\n'
    'string frame_id\n'
    '================================================================================\n'
    'MSG: geometry_msgs/Pose\n'
    'geometry_msgs/Point position\n'
    'geometry_msgs/Quaternion orientation\n'
    '================================================================================\n'
    'MSG: geometry_msgs/Point\n'
    'float64 x\n'
    'float64 y\n'
    'float64 z\n'
    '================================================================================\n'
    'MSG: geometry_msgs/Quaternion\n'
    'float64 x\n'
    'float64 y\n'
    'float64 z\n'
    'float64 w\n'
)


def write_pose_bag(output_path, poses, target_frame, output_topic):
    with Writer(output_path) as writer:
        conn = writer.add_connection(
            output_topic,
            'geometry_msgs/msg/PoseStamped',
            msgdef=_POSE_STAMPED_MSGDEF,
            md5sum='d3812c3cbc69362b77dc0b19b345f8f5',
        )
        for seq, p in enumerate(poses):
            t_ns = p['t_sec'] * 10**9 + p['t_nsec']
            data = serialize_pose_stamped(
                seq, p['t_sec'], p['t_nsec'], target_frame,
                p['x'], p['y'], p['z'],
                p['qx'], p['qy'], p['qz'], p['qw'],
            )
            writer.write(conn, t_ns, data)

        print(f'  Wrote {len(poses)} PoseStamped messages to {output_topic}')


def main():
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', required=True,
                        help='Folder containing ROS1 .bag files')
    parser.add_argument('--source-frame', '-s', default='robotrainer_front_marker')
    parser.add_argument('--target-frame', '-t', default='map')
    parser.add_argument('--output-topic', default=None)
    parser.add_argument('--list-frames', action='store_true',
                        help='Print all TF frames found in each bag and exit')
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f'Error: {folder} is not a directory.', file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    bag_files = sorted(
        f for f in folder.glob('*.bag')
        if not f.stem.endswith('_transformed_to_map_frame')
    )
    if not bag_files:
        print(f'No .bag files found in {folder}.', file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    if args.list_frames:
        all_frames = set()
        for bagfile in bag_files:
            frames = list_tf_frames(bagfile)
            all_frames.update(frames)
            print(f'\n{bagfile.name}:')
            for f in frames:
                print(f'  {f}')
        print(f'\nAll frames across all bags ({len(all_frames)} unique):')
        for f in sorted(all_frames):
            print(f'  {f}')
        rclpy.shutdown()
        return

    output_topic = args.output_topic or f'/{args.source_frame}_in_{args.target_frame}'

    for bagfile in bag_files:
        output_path = bagfile.parent / (bagfile.stem + '_transformed_to_map_frame.bag')
        if output_path.exists():
            print(f'Skipping {bagfile.name}: output already exists ({output_path.name})')
            continue

        print(f'\nProcessing: {bagfile.name}')
        print(f'  Transform: {args.target_frame} <- {args.source_frame}')

        poses = extract_map_frame_pose(str(bagfile), args.source_frame, args.target_frame)
        print(f'  Collected {len(poses)} poses.')

        if not poses:
            print(f'  WARNING: no poses found — skipping. Check that the TF tree '
                  f'connects "{args.source_frame}" to "{args.target_frame}".',
                  file=sys.stderr)
            continue

        write_pose_bag(str(output_path), poses, args.target_frame, output_topic)
        print(f'  Written -> {output_path.name}')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
