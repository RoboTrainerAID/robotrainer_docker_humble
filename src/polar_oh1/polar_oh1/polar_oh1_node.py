#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Hochschule Karlsruhe IRAS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ROS 2 driver for the Polar OH1+ optical heart rate sensor.

Publishes the same topics as the original version (``hr``, ``ppg_ch0..3``,
``ppi``, ``hrv``) with the same message types.

BLE and the Polar Measurement Data control protocol are handled by
`bleakheart <https://github.com/fsmeraldi/bleakheart>`_.  bleakheart has no
decoder for PP intervals and none for uncompressed PPG, so those two payloads
are unpacked here; everything else -- connection, control point commands and
their error codes, the sensor-to-host time offset -- comes from the library.

Two limitations are documented Polar behaviour, not driver faults:

* enabling the PPI stream switches the sensor to a different algorithm, so
  heart rate then updates only every 5 s and the first PPI batch takes about
  25 s to arrive (polar-ble-sdk, ``documentation/products/PolarOH1.md``);
* PPI-derived HRV "cannot be measured accurately during activity, it should
  only be used at complete rest" (``documentation/PPIData.md``).  For a moving
  subject use the Polar H10 and ``polar_h10_node``, which derives HRV from
  ECG R-R intervals.
"""

from __future__ import annotations

import asyncio
import struct
import threading
from typing import List, Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

import bleakheart
from bleak import BleakClient, BleakScanner

from std_msgs.msg import Float32MultiArray, Int32

from polar_oh1.polar_h10_node import HrvAnalyzer, _wait_or_shutdown, adapter_kwargs

# PPI sample layout (PMD specification, table 22): heart rate, interval,
# error estimate, flags. bleakheart passes PPI frames through undecoded.
PPI_SAMPLE = struct.Struct('<BHHB')
PPI_FLAG_INVALID = 0x01
PPI_FLAG_SKIN_CONTACT = 0x02
PPI_FLAG_CONTACT_UNSUPPORTED = 0x04

PPG_CHANNELS = 4
PPG_RESOLUTION = 22
# The OH1 must be asked for 130 Hz even though it actually samples at 135 Hz
# (polar-ble-sdk, documentation/KnownIssues.md, OH1 issue 2).
PPG_REQUEST_RATE = 130


def decode_ppg_uncompressed(payload: bytes) -> List[List[int]]:
    """Unpack PPG frame type 0: four 24-bit signed channels per sample."""
    stride = PPG_CHANNELS * 3
    if len(payload) % stride:
        raise ValueError('PPG payload length {} is not a multiple of {}'.format(
            len(payload), stride))
    return [
        [int.from_bytes(payload[off + 3 * c:off + 3 * c + 3], 'little', signed=True)
         for c in range(PPG_CHANNELS)]
        for off in range(0, len(payload), stride)
    ]


class PolarOH1Node(Node):

    def __init__(self) -> None:
        super().__init__('polar_oh1_node')

        self.declare_parameter('Device_Mac_Address', 'A0:9E:1A:E0:BC:97')
        self.declare_parameter('adapter', '')
        self.declare_parameter('publish_ppg', True)
        self.declare_parameter('publish_ppi', True)
        self.declare_parameter('ppg_decimation', 1)
        self.declare_parameter('reconnect_delay', 5.0)
        self.declare_parameter('ppi_max_error_ms', 30.0)
        self.declare_parameter('hrv_window_sec', 60.0)
        self.declare_parameter('hrv_min_beats', 10)

        self.device_mac = self.get_parameter('Device_Mac_Address').value
        self.adapter = self.get_parameter('adapter').value
        self.publish_ppg_enabled = self.get_parameter('publish_ppg').value
        self.publish_ppi_enabled = self.get_parameter('publish_ppi').value
        self.ppg_decimation = max(1, int(self.get_parameter('ppg_decimation').value))
        self.reconnect_delay = float(self.get_parameter('reconnect_delay').value)
        self.ppi_max_error_ms = float(self.get_parameter('ppi_max_error_ms').value)

        # Topic names and message types are unchanged from the original node.
        self.pub_hr = self.create_publisher(Int32, 'biosensors/polar_oh1/hr', 10)
        self.pub_ppg = [
            self.create_publisher(Float32MultiArray,
                                  'biosensors/polar_oh1/ppg_ch{}'.format(i), 10)
            for i in range(PPG_CHANNELS)
        ]
        self.pub_ppi = self.create_publisher(Int32, 'biosensors/polar_oh1/ppi', 10)
        self.pub_hrv = self.create_publisher(Float32MultiArray, 'biosensors/polar_oh1/hrv', 10)

        self.hrv = HrvAnalyzer(
            window_sec=float(self.get_parameter('hrv_window_sec').value),
            min_beats=int(self.get_parameter('hrv_min_beats').value),
            rr_min_ms=300.0,
            rr_max_ms=2000.0,
            max_rel_change=0.25,
        )

        self.stats = {'ppg': 0, 'ppi': 0, 'ppi_rejected': 0, 'hr': 0, 'errors': 0}
        self._warned_compressed_ppg = False
        self.create_timer(10.0, self._log_stats)

        if self.publish_ppi_enabled:
            self.get_logger().warn(
                'PPI streaming is enabled: the OH1 then updates heart rate only every 5 s '
                'and the first PPI batch takes about 25 s to arrive. This is documented '
                'sensor behaviour, not a driver fault.')

    def _log_stats(self) -> None:
        self.get_logger().info(
            'PPG samples {ppg} | PPI accepted {ppi} rejected {ppi_rejected} | '
            'HR frames {hr} | decode errors {errors}'.format(**self.stats))

    # -- link callbacks -----------------------------------------------------

    def on_heartrate(self, frame) -> None:
        """
        Handle a bleakheart HeartRate callback with unpack=False.

        Frame layout is ``('HR', t_ns, (avg_hr, rr_list), energy)``.  Unpacking
        is off because the OH1 reports no R-R intervals while PPI streaming is
        active, and with unpack=True bleakheart would then emit nothing at all.
        """
        try:
            _, _host_ns, (avg_hr, _rr_list), _ = frame
            self.stats['hr'] += 1
            if avg_hr > 0:
                self.pub_hr.publish(Int32(data=int(avg_hr)))
        except Exception as exc:  # noqa: BLE001
            self.stats['errors'] += 1
            self.get_logger().error('HR callback failed: {}'.format(exc),
                                    throttle_duration_sec=5.0)

    def on_measurement(self, frame) -> None:
        """
        Handle a bleakheart PolarMeasurementData callback.

        PPI frames and uncompressed PPG frames arrive as raw bytearrays
        because bleakheart has no decoder for them; compressed PPG arrives
        already decoded.
        """
        try:
            kind, _host_ns, payload = frame
            if kind == 'PPG':
                self._handle_ppg(payload)
            elif kind == 'PPI':
                self._handle_ppi(payload)
            else:
                self.get_logger().warn('Ignoring unexpected {} frame'.format(kind),
                                       throttle_duration_sec=30.0)
        except Exception as exc:  # noqa: BLE001
            self.stats['errors'] += 1
            self.get_logger().error('{} frame handling failed: {}'.format(frame[0], exc),
                                    throttle_duration_sec=5.0)

    def _handle_ppg(self, payload) -> None:
        if isinstance(payload, (bytes, bytearray)):
            frame_type = payload[9] & 0x7F
            if frame_type != 0:
                raise ValueError('unsupported PPG frame type 0x{:02x}'.format(frame_type))
            samples = decode_ppg_uncompressed(bytes(payload[10:]))
        else:
            # bleakheart already decoded a delta-compressed frame. Note that
            # its delta unpacking reads the bit stream MSB first, while the
            # official Polar SDK (BlePMDClient.parseDeltaFrame) reads it LSB
            # first, so these values are worth sanity checking against the
            # uncompressed path before you trust them.
            if not self._warned_compressed_ppg:
                self._warned_compressed_ppg = True
                self.get_logger().warn(
                    'Sensor is sending delta-compressed PPG; decoding is done by '
                    'bleakheart, whose delta bit order differs from the official '
                    'Polar SDK. Verify the values before using them.')
            samples = [list(sample) for sample in payload]

        if self.ppg_decimation > 1:
            samples = samples[::self.ppg_decimation]
        if not samples:
            return

        self.stats['ppg'] += len(samples)
        for channel in range(PPG_CHANNELS):
            values = [float(sample[channel]) for sample in samples]
            self.pub_ppg[channel].publish(Float32MultiArray(data=values))

    def _handle_ppi(self, payload) -> None:
        body = bytes(payload[10:])
        frame_type = payload[9] & 0x7F
        if frame_type != 0:
            raise ValueError('unsupported PPI frame type 0x{:02x}'.format(frame_type))
        if len(body) % PPI_SAMPLE.size:
            raise ValueError('PPI payload length {} is not a multiple of {}'.format(
                len(body), PPI_SAMPLE.size))

        # Polar gives PPI samples no usable acquisition time ("the sample time
        # is either zero or it is missing from the stream"), so receive time is
        # the best available reference.
        now_ns = self.get_clock().now().nanoseconds
        for offset in range(0, len(body), PPI_SAMPLE.size):
            _hr, interval_ms, error_ms, flags = PPI_SAMPLE.unpack_from(body, offset)

            # Polar: "If skin contact flag is 0 or blocker flag is 1, the
            # sample should not be treated as valid and discarded."
            if flags & PPI_FLAG_INVALID:
                self.stats['ppi_rejected'] += 1
                continue
            contact_supported = not (flags & PPI_FLAG_CONTACT_UNSUPPORTED)
            if contact_supported and not (flags & PPI_FLAG_SKIN_CONTACT):
                self.stats['ppi_rejected'] += 1
                continue
            if error_ms > self.ppi_max_error_ms:
                self.stats['ppi_rejected'] += 1
                continue

            self.stats['ppi'] += 1
            self.pub_ppi.publish(Int32(data=int(interval_ms)))

            metrics = self.hrv.add_beat(now_ns, float(interval_ms))
            if not self.hrv.last_accepted:
                self.stats['ppi_rejected'] += 1
            elif metrics is not None:
                self.pub_hrv.publish(Float32MultiArray(
                    data=[float(metrics['rmssd']), float(metrics['sdnn'])]))


class PolarOH1Link:
    """Connect / stream / reconnect, driving bleakheart's Polar classes."""

    def __init__(self, node: PolarOH1Node) -> None:
        self._node = node
        self._log = node.get_logger()
        self._client: Optional[BleakClient] = None
        self._hr: Optional[bleakheart.HeartRate] = None
        self._pmd: Optional[bleakheart.PolarMeasurementData] = None
        self._streams: List[str] = []
        self._disconnected = asyncio.Event()
        self.shutdown = asyncio.Event()

    def _on_disconnect(self, _client) -> None:
        if not self.shutdown.is_set():
            self._disconnected.set()

    async def _connect_and_stream(self) -> None:
        kwargs = {'timeout': 20.0}
        kwargs.update(adapter_kwargs(self._node.adapter))

        self._log.info('Scanning for Polar OH1+ at {} ...'.format(self._node.device_mac))
        device = await BleakScanner.find_device_by_address(self._node.device_mac, **kwargs)
        if device is None:
            raise RuntimeError('device {} not found'.format(self._node.device_mac))

        self._client = BleakClient(device, disconnected_callback=self._on_disconnect, **kwargs)
        await self._client.connect()
        if not self._client.is_connected:
            raise RuntimeError('connect() returned but client is not connected')
        self._log.info('Connected to Polar OH1+ [{}]'.format(device.address))

        self._node.hrv.reset()
        self._streams = []

        self._hr = bleakheart.HeartRate(
            self._client, callback=self._node.on_heartrate, unpack=False)
        await self._hr.start_notify()

        self._pmd = bleakheart.PolarMeasurementData(
            self._client, callback=self._node.on_measurement)

        if self._node.publish_ppg_enabled:
            await self._start('PPG', SAMPLE_RATE=PPG_REQUEST_RATE, RESOLUTION=PPG_RESOLUTION)
        if self._node.publish_ppi_enabled:
            await self._start('PPI')

    async def _start(self, measurement: str, **settings) -> None:
        code, message, _ = await self._pmd.start_streaming(measurement, **settings)
        if code != 0:
            raise RuntimeError('start {} stream failed: {} (code {})'.format(
                measurement, message, code))
        self._streams.append(measurement)
        self._log.info('{} stream started'.format(measurement))

    async def _teardown(self) -> None:
        if self._client is not None and self._client.is_connected:
            for measurement in reversed(self._streams):
                try:
                    code, message = await self._pmd.stop_streaming(measurement)
                    if code != 0:
                        self._log.warn('Stopping {} returned: {}'.format(measurement, message))
                    else:
                        self._log.info('{} stream stopped'.format(measurement))
                except Exception as exc:  # noqa: BLE001
                    self._log.warn('Could not stop {} stream: {}'.format(measurement, exc))
            if self._hr is not None:
                try:
                    await self._hr.stop_notify()
                except Exception:  # noqa: BLE001
                    pass
            try:
                await self._client.disconnect()
                self._log.info('Disconnected from sensor')
            except Exception as exc:  # noqa: BLE001
                self._log.warn('Disconnect failed: {}'.format(exc))

        self._streams = []
        self._hr = None
        self._pmd = None
        self._client = None

    async def run(self) -> None:
        try:
            while not self.shutdown.is_set():
                try:
                    self._disconnected.clear()
                    await self._connect_and_stream()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self._log.error('Connection attempt failed: {} - retrying in {:.1f}s'.format(
                        exc, self._node.reconnect_delay))
                    await self._teardown()
                    await _wait_or_shutdown(self.shutdown, self._node.reconnect_delay)
                    continue

                while not self.shutdown.is_set() and not self._disconnected.is_set():
                    await _wait_or_shutdown(self.shutdown, 1.0)

                if self._disconnected.is_set() and not self.shutdown.is_set():
                    self._log.warn('Sensor disconnected - reconnecting in {:.1f}s'.format(
                        self._node.reconnect_delay))
                    await self._teardown()
                    await _wait_or_shutdown(self.shutdown, self._node.reconnect_delay)
        finally:
            await self._teardown()


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def main(args=None) -> None:
    rclpy.init(args=args)

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
    loop_thread.start()

    node = PolarOH1Node()
    link = PolarOH1Link(node)
    task = asyncio.run_coroutine_threadsafe(link.run(), loop)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C reaches us as either of these depending on whether rclpy's
        # signal handler got there first; both are a normal shutdown.
        node.get_logger().info('Shutting down ...')
    finally:
        loop.call_soon_threadsafe(link.shutdown.set)
        try:
            task.result(timeout=15.0)
        except Exception as exc:  # noqa: BLE001
            node.get_logger().warn('BLE task did not stop cleanly: {}'.format(exc))

        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=3.0)
        loop.close()

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
