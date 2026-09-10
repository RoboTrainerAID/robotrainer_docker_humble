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
ROS 2 driver for the Polar H10 heart rate sensor.

Streams heart rate, R-R intervals, raw ECG, accelerometer, battery and skin
contact and publishes each of them on its own topic using only standard ROS
message types, so the data survives a ros1_bridge hop.

The BLE and Polar Measurement Data protocol work is done by `bleakheart
<https://github.com/fsmeraldi/bleakheart>`_ (``pip install bleakheart``).
This module only adds what bleakheart does not provide:

* per-sample timestamps inside a PMD frame (bleakheart reports one timestamp
  per frame, which refers to the frame's last sample);
* anchoring those timestamps to the ROS clock, plus a guard for the sensor
  losing its own epoch between subjects;
* time-domain HRV with artifact rejection;
* the ROS interface.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import statistics
import threading
from collections import deque
from typing import Dict, List, Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclTime

import bleakheart
from bleak import BleakClient, BleakScanner

from rcl_interfaces.msg import ParameterDescriptor

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PointStamped, Vector3Stamped
from std_msgs.msg import Bool, Float32

STANDARD_GRAVITY = 9.80665  # m/s^2, for the mG -> m/s^2 conversion

# bleak 3.0 moved the BlueZ adapter selection from a top-level ``adapter=``
# argument into ``bluez={"adapter": ...}``.  The old spelling is now absorbed
# by ``**kwargs`` and silently ignored, so probe the installed version instead
# of guessing.  BleakClient and BleakScanner changed together, so one probe
# covers both.
_BLEAK_USES_BLUEZ_ARGS = "bluez" in inspect.signature(BleakClient.__init__).parameters


def adapter_kwargs(adapter: str) -> dict:
    """Return bleak keyword arguments selecting a BlueZ adapter, e.g. hci1."""
    if not adapter:
        return {}
    if _BLEAK_USES_BLUEZ_ARGS:
        return {"bluez": {"adapter": adapter}}
    return {"adapter": adapter}


# ---------------------------------------------------------------------------
# Battery over BlueZ
# ---------------------------------------------------------------------------

# bluetoothd's battery plugin claims the GATT Battery Service (0x180f) for
# itself and does not re-export its Battery Level characteristic (0x2a19) on
# the GATT D-Bus API.  On such a host bleak cannot find the characteristic at
# all -- "Characteristic 00002a19-0000-1000-8000-00805f9b34fb was not found!"
# -- even though the H10 does provide it.  The level is then only readable as
# the ``Percentage`` property of ``org.bluez.Battery1`` on the device object:
#
#   dbus-send --system --print-reply --dest=org.bluez \
#       /org/bluez/hci1/dev_24_AC_AC_1E_C2_03 \
#       org.freedesktop.DBus.Properties.GetAll string:org.bluez.Battery1
#
# dbus-fast is bleak's own BlueZ dependency, so it is installed wherever this
# matters; the import stays guarded so the node still imports on a host
# without BlueZ.
try:
    from dbus_fast import BusType, Message, MessageType
    from dbus_fast.aio import MessageBus
except ImportError:  # no BlueZ backend -- GATT is the only source
    MessageBus = None

BLUEZ_SERVICE = "org.bluez"
BLUEZ_BATTERY_INTERFACE = "org.bluez.Battery1"
BLUEZ_DEVICE_INTERFACE = "org.bluez.Device1"
DBUS_PROPERTIES_INTERFACE = "org.freedesktop.DBus.Properties"
DBUS_OBJECT_MANAGER_INTERFACE = "org.freedesktop.DBus.ObjectManager"


async def _bluez_call(bus, path: str, interface: str, member: str,
                      signature: str = "", body: Optional[list] = None) -> Optional[list]:
    """Send one method call to bluetoothd; return its reply body, or None."""
    reply = await bus.call(Message(destination=BLUEZ_SERVICE, path=path,
                                   interface=interface, member=member,
                                   signature=signature, body=body or []))
    if reply is None or reply.message_type != MessageType.METHOD_RETURN:
        return None
    return reply.body


async def read_bluez_battery_percent(address: str,
                                     device_path: Optional[str] = None) -> Optional[int]:
    """
    Read ``org.bluez.Battery1.Percentage`` for a connected device.

    Returns None whenever that interface is not there to be read: a host
    without BlueZ, a bluetoothd without the battery plugin, or a device that is
    not currently connected.  ``device_path`` is queried directly when it is
    known; otherwise the BlueZ object tree is searched for a device with a
    matching address, which also covers not knowing which adapter it sits on.

    A short-lived bus connection per read keeps this clear of the connection
    bleak maintains, at a cost that is irrelevant once a minute.
    """
    if MessageBus is None:
        return None

    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    try:
        if device_path:
            body = await _bluez_call(bus, device_path, DBUS_PROPERTIES_INTERFACE,
                                     "Get", "ss",
                                     [BLUEZ_BATTERY_INTERFACE, "Percentage"])
            if body:
                return int(body[0].value)

        body = await _bluez_call(bus, "/", DBUS_OBJECT_MANAGER_INTERFACE,
                                 "GetManagedObjects")
        if not body:
            return None

        wanted = address.upper()
        for interfaces in body[0].values():
            battery = interfaces.get(BLUEZ_BATTERY_INTERFACE)
            found = interfaces.get(BLUEZ_DEVICE_INTERFACE, {}).get("Address")
            if not battery or found is None or found.value.upper() != wanted:
                continue
            percentage = battery.get("Percentage")
            if percentage is not None:
                return int(percentage.value)
        return None
    finally:
        bus.disconnect()


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

class RosTimeAnchor:
    """
    Maps bleakheart timestamps onto the ROS clock.

    bleakheart stamps both heart rate beats and PMD frames in the
    ``time.time_ns()`` domain, so a single delta converts the lot.  In the
    normal case ROS time *is* system time and the delta is zero, but taking it
    from ``node.get_clock()`` keeps the driver on the same clock as every
    other publisher and as the bag recorder.

    The guard matters for a multi-subject session: the H10 forgets its own
    time about a minute after being taken off the strap, and bleakheart
    computes its sensor-to-host offset only once per
    :class:`~bleakheart.PolarMeasurementData` instance.  A fresh instance is
    built on every connection, so a handover normally re-anchors by itself;
    if a mapping ever does come out implausible, this catches it instead of
    stamping samples years away.
    """

    def __init__(self, max_error_ns: int = 5_000_000_000) -> None:
        self._delta: Optional[int] = None
        self._max_error = max_error_ns
        self.reanchor_count = 0

    def reset(self) -> None:
        self._delta = None

    @property
    def delta_ns(self) -> Optional[int]:
        return self._delta

    def to_ros(self, host_ns: int, ros_now_ns: int) -> int:
        if self._delta is None:
            self._delta = ros_now_ns - host_ns
        mapped = host_ns + self._delta
        if abs(ros_now_ns - mapped) > self._max_error:
            self._delta = ros_now_ns - host_ns
            self.reanchor_count += 1
            mapped = ros_now_ns
        return mapped


class FrameTimebase:
    """
    Reconstructs per-sample timestamps inside a PMD frame.

    bleakheart hands over a whole frame with a single timestamp, which refers
    to its *last* sample.  Following the official Polar SDK
    (``PmdDataModelUtils.kt``), the inter-sample delta is measured from
    consecutive frame timestamps -- which tracks the sensor's real rate rather
    than the nominal one -- and falls back to the nominal rate for the first
    frame or whenever the measured value is implausible.
    """

    def __init__(self, nominal_rate_hz: float, tolerance: float = 0.5) -> None:
        self._nominal_delta = int(round(1e9 / nominal_rate_hz))
        self._tolerance = tolerance
        self._prev_ts: Optional[int] = None

    def reset(self) -> None:
        self._prev_ts = None

    def sample_times(self, last_sample_ts: int, n_samples: int) -> List[int]:
        if n_samples <= 0:
            return []

        delta = self._nominal_delta
        if self._prev_ts is not None and last_sample_ts > self._prev_ts:
            measured = (last_sample_ts - self._prev_ts) // n_samples
            lo = self._nominal_delta * (1.0 - self._tolerance)
            hi = self._nominal_delta * (1.0 + self._tolerance)
            if lo <= measured <= hi:
                delta = measured
        self._prev_ts = last_sample_ts

        first = last_sample_ts - delta * (n_samples - 1)
        return [first + i * delta for i in range(n_samples)]


# ---------------------------------------------------------------------------
# HRV
# ---------------------------------------------------------------------------

class HrvAnalyzer:
    """
    Sliding-window time-domain HRV with artifact rejection.

    There is no publishing latency: a new value is emitted on every accepted
    beat and stamped with that beat's own time.  What the window does mean is
    that the value *looks backward* over ``window_sec`` -- see the README for
    how to size it against a short experiment.

    A single missed or doubled beat is enough to wreck RMSSD, so beats are
    screened first:

    * absolute plausibility (``rr_min_ms`` .. ``rr_max_ms``);
    * relative change against the median of the recent accepted beats
      (Malik-style), which catches ectopics and dropped beats.

    Successive differences are only formed between beats that are both
    accepted *and* adjacent, so a rejected beat breaks the chain instead of
    silently creating a huge fake difference.
    """

    def __init__(self, window_sec: float, min_beats: int, rr_min_ms: float,
                 rr_max_ms: float, max_rel_change: float) -> None:
        self._window_ns = int(window_sec * 1e9)
        self._min_beats = min_beats
        self._rr_min = rr_min_ms
        self._rr_max = rr_max_ms
        self._max_rel_change = max_rel_change
        self._beats: deque = deque()  # (t_ns, rr_ms, accepted)
        self._recent_ok: deque = deque(maxlen=8)
        #: Whether the most recent beat passed the artifact filter.  ``add_beat``
        #: returns None both for a rejected beat and for a window that is not
        #: full yet, so callers need this to tell the two apart.
        self.last_accepted = False

    def reset(self) -> None:
        self._beats.clear()
        self._recent_ok.clear()
        self.last_accepted = False

    def _accept(self, rr_ms: float) -> bool:
        if not (self._rr_min <= rr_ms <= self._rr_max):
            return False
        if self._recent_ok:
            reference = statistics.median(self._recent_ok)
            if reference > 0 and abs(rr_ms - reference) / reference > self._max_rel_change:
                return False
        return True

    def add_beat(self, t_ns: int, rr_ms: float) -> Optional[Dict[str, float]]:
        """Add one R-R interval; return HRV metrics if the window is valid."""
        accepted = self._accept(rr_ms)
        self.last_accepted = accepted
        if accepted:
            self._recent_ok.append(rr_ms)
        self._beats.append((t_ns, rr_ms, accepted))

        cutoff = t_ns - self._window_ns
        while self._beats and self._beats[0][0] < cutoff:
            self._beats.popleft()

        if not accepted:
            return None

        rr_values = [rr for _, rr, ok in self._beats if ok]
        if len(rr_values) < self._min_beats:
            return None

        # Successive differences between adjacent accepted beats only.
        diffs: List[float] = []
        prev_ok: Optional[float] = None
        for _, rr, ok in self._beats:
            if ok:
                if prev_ok is not None:
                    diffs.append(rr - prev_ok)
                prev_ok = rr
            else:
                prev_ok = None

        if len(diffs) < 2:
            return None

        rmssd = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        sdnn = statistics.stdev(rr_values)
        nn50 = sum(1 for d in diffs if abs(d) > 50.0)
        mean_rr = statistics.fmean(rr_values)
        span_sec = (self._beats[-1][0] - self._beats[0][0]) * 1e-9

        return {
            "rmssd": rmssd,
            "sdnn": sdnn,
            "pnn50": 100.0 * nn50 / len(diffs),
            "mean_rr": mean_rr,
            "mean_hr": 60000.0 / mean_rr if mean_rr > 0 else 0.0,
            "quality": len(rr_values) / len(self._beats),
            # Provenance: exactly which beats and how much time this value
            # covers, so a short experiment can be filtered on it afterwards.
            "n_beats": float(len(rr_values)),
            "window_sec": span_sec,
        }


HRV_KEYS = ("rmssd", "sdnn", "pnn50", "mean_rr", "mean_hr",
            "quality", "n_beats", "window_sec")


# ---------------------------------------------------------------------------
# BLE session
# ---------------------------------------------------------------------------

class PolarH10Link:
    """Connect / stream / reconnect, driving bleakheart's Polar classes."""

    def __init__(self, node: "PolarH10Node") -> None:
        self._node = node
        self._log = node.get_logger()
        self._client: Optional[BleakClient] = None
        self._hr: Optional[bleakheart.HeartRate] = None
        self._pmd: Optional[bleakheart.PolarMeasurementData] = None
        self._streams: List[str] = []
        #: Which of the two battery sources answered last, so a host whose
        #: bluetoothd owns the battery service is not asked for a hidden
        #: characteristic once a minute.  Reset whenever neither answers.
        self._battery_source: Optional[str] = None
        self._disconnected = asyncio.Event()
        self.shutdown = asyncio.Event()

    # -- discovery ----------------------------------------------------------

    async def _find_device(self):
        address = self._node.p_address.strip()
        kwargs = {"timeout": self._node.p_scan_timeout}
        kwargs.update(adapter_kwargs(self._node.p_adapter))

        if ":" in address:
            self._log.info("Scanning for Polar H10 at address {} ...".format(address))
            return await BleakScanner.find_device_by_address(address, **kwargs)

        needle = address.lower() if address else "polar h10"
        self._log.info("Scanning for a device whose name contains '{}' ...".format(needle))
        devices = await BleakScanner.discover(**kwargs)
        matches = sorted(
            (d for d in devices if d.name and needle in d.name.lower()),
            key=lambda d: (d.name or "", d.address),
        )
        for device in matches:
            self._log.info("  found {} [{}]".format(device.name, device.address))
        return matches[0] if matches else None

    def _on_disconnect(self, _client) -> None:
        if not self.shutdown.is_set():
            self._disconnected.set()

    # -- session ------------------------------------------------------------

    async def _connect_and_stream(self) -> None:
        device = await self._find_device()
        if device is None:
            raise RuntimeError("no matching Polar H10 found")

        kwargs = {"disconnected_callback": self._on_disconnect, "timeout": 20.0}
        kwargs.update(adapter_kwargs(self._node.p_adapter))

        self._client = BleakClient(device, **kwargs)
        await self._client.connect()
        if not self._client.is_connected:
            raise RuntimeError("connect() returned but client is not connected")

        self._log.info("Connected to {} [{}]".format(
            getattr(device, "name", "?"), device.address))
        # Fresh state: a new subject means a new sensor epoch and a new
        # baseline, and the beat chain from the previous wearer is meaningless.
        self._node.on_connected(device.address)

        self._streams = []
        await self._read_battery()

        # unpack=True makes bleakheart split each frame into individual beats
        # and estimate a timestamp per beat.  Consequence worth knowing: a
        # frame carrying no R-R interval produces no callback at all, so heart
        # rate pauses while the strap has no skin contact.
        self._hr = bleakheart.HeartRate(
            self._client,
            callback=self._node.on_heartbeat,
            contact_callback=self._node.on_contact_gained,
            contact_lost_callback=self._node.on_contact_lost,
            instant_rate=False,
            unpack=True,
        )
        await self._hr.start_notify()
        self._log.info("Heart rate / R-R stream started")

        # A new PolarMeasurementData instance per connection, so bleakheart
        # recomputes its sensor-to-host time offset for this session.
        self._pmd = bleakheart.PolarMeasurementData(
            self._client, callback=self._node.on_measurement)

        try:
            offered = await self._pmd.available_measurements()
            self._log.info("Sensor offers: {}".format(", ".join(offered) or "nothing"))
        except Exception as exc:  # noqa: BLE001
            self._log.warn("Could not read available measurements: {}".format(exc))

        if self._node.p_publish_ecg:
            await self._start("ECG",
                              SAMPLE_RATE=int(self._node.p_ecg_sample_rate),
                              RESOLUTION=14)
        if self._node.p_publish_acc:
            await self._start("ACC",
                              SAMPLE_RATE=int(self._node.p_acc_sample_rate),
                              RESOLUTION=16,
                              RANGE=int(self._node.p_acc_range_g))

    async def _start(self, measurement: str, **settings) -> None:
        code, message, _ = await self._pmd.start_streaming(measurement, **settings)
        if code != 0:
            raise RuntimeError("start {} stream failed: {} (code {})".format(
                measurement, message, code))
        self._streams.append(measurement)
        self._log.info("{} stream started ({})".format(
            measurement, ", ".join("{}={}".format(k, v) for k, v in settings.items())))

    def _bluez_device_path(self) -> Optional[str]:
        """
        BlueZ object path of the connected device, if bleak's BlueZ backend
        is in use.  Private bleak state, hence the defensive lookup: without it
        :func:`read_bluez_battery_percent` searches by address instead.
        """
        return getattr(getattr(self._client, "_backend", None), "_device_path", None)

    async def _read_battery(self) -> None:
        """
        Publish the battery level, from GATT or from BlueZ.

        The Battery Level characteristic is the portable source and is tried
        first; ``org.bluez.Battery1`` covers the hosts where bluetoothd has
        taken that characteristic over (see :func:`read_bluez_battery_percent`).
        """
        gatt_error: Optional[Exception] = None
        if self._battery_source != "bluez":
            try:
                level = int(await bleakheart.BatteryLevel(self._client).read())
            except Exception as exc:  # noqa: BLE001
                gatt_error = exc
            else:
                self._battery_source = "gatt"
                self._node.on_battery(level)
                return

        percent: Optional[int] = None
        try:
            percent = await read_bluez_battery_percent(
                self._client.address, self._bluez_device_path())
        except Exception as exc:  # noqa: BLE001
            self._log.debug("org.bluez.Battery1 read failed: {}".format(exc))

        if percent is not None:
            if self._battery_source != "bluez":
                self._log.info(
                    "Battery Level characteristic not exposed by bluetoothd - "
                    "reading org.bluez.Battery1 instead")
                self._battery_source = "bluez"
            self._node.on_battery(percent)
            return

        self._battery_source = None
        self._log.warn("Battery read failed: {}".format(
            gatt_error if gatt_error is not None else "no battery level available"),
            throttle_duration_sec=60.0)

    async def _teardown(self) -> None:
        """
        Stop the streams and disconnect.

        Polar documents that the H10 keeps running until its battery is flat
        if ECG/ACC streaming is not stopped by the host (polar-ble-sdk
        KnownIssues, H10 issue 2), so this has to run.
        """
        if self._client is not None and self._client.is_connected:
            for measurement in reversed(self._streams):
                try:
                    code, message = await self._pmd.stop_streaming(measurement)
                    if code != 0:
                        self._log.warn("Stopping {} returned: {}".format(measurement, message))
                    else:
                        self._log.info("{} stream stopped".format(measurement))
                except Exception as exc:  # noqa: BLE001
                    self._log.warn("Could not stop {} stream: {}".format(measurement, exc))
            if self._hr is not None:
                try:
                    await self._hr.stop_notify()
                except Exception:  # noqa: BLE001
                    pass
            try:
                await self._client.disconnect()
                self._log.info("Disconnected from sensor")
            except Exception as exc:  # noqa: BLE001
                self._log.warn("Disconnect failed: {}".format(exc))

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
                    self._node.on_disconnected()
                    self._log.error("Connection attempt failed: {} - retrying in {:.1f}s".format(
                        exc, self._node.p_reconnect_delay))
                    await self._teardown()
                    await _wait_or_shutdown(self.shutdown, self._node.p_reconnect_delay)
                    continue

                next_battery = self._node.p_battery_period
                while not self.shutdown.is_set() and not self._disconnected.is_set():
                    await _wait_or_shutdown(self.shutdown, 1.0)
                    next_battery -= 1.0
                    if next_battery <= 0.0 and self._client and self._client.is_connected:
                        await self._read_battery()
                        next_battery = self._node.p_battery_period

                if self._disconnected.is_set() and not self.shutdown.is_set():
                    self._node.on_disconnected()
                    self._log.warn("Sensor disconnected - reconnecting in {:.1f}s".format(
                        self._node.p_reconnect_delay))
                    await self._teardown()
                    await _wait_or_shutdown(self.shutdown, self._node.p_reconnect_delay)
        finally:
            self._node.on_disconnected()
            await self._teardown()


async def _wait_or_shutdown(shutdown: asyncio.Event, delay: float) -> None:
    try:
        await asyncio.wait_for(shutdown.wait(), timeout=delay)
    except asyncio.TimeoutError:
        pass


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class PolarH10Node(Node):
    """Publishes every Polar H10 signal on its own topic, standard types only."""

    def __init__(self) -> None:
        super().__init__("polar_h10_node")

        params = (
            ("address", "", "MAC address or name substring. Empty = first Polar H10 found."),
            ("adapter", "", "Bluetooth adapter, e.g. hci0. Empty = system default."),
            ("frame_id", "polar_h10", "frame_id written into every header."),
            ("topic_prefix", "biosensors/polar_h10", "Prefix for all published topics."),
            ("scan_timeout", 15.0, "BLE scan timeout in seconds."),
            ("reconnect_delay", 5.0, "Seconds to wait between reconnection attempts."),
            ("battery_period", 60.0, "Battery level poll period in seconds."),
            ("publish_ecg", True, "Stream raw ECG at 130 Hz."),
            ("publish_acc", True, "Stream the accelerometer."),
            ("publish_hrv", True, "Compute and publish time-domain HRV."),
            ("ecg_sample_rate", 130, "ECG sample rate in Hz (the H10 supports 130)."),
            ("acc_sample_rate", 200, "ACC sample rate in Hz (25/50/100/200)."),
            ("acc_range_g", 8, "ACC range in g (2/4/8)."),
            ("acc_in_g", False, "Publish acceleration in g instead of m/s^2."),
            ("hrv_window_sec", 30.0, "Sliding window length for HRV in seconds."),
            ("hrv_min_beats", 10, "Minimum accepted beats before HRV is published."),
            ("rr_min_ms", 300.0, "Reject R-R intervals shorter than this."),
            ("rr_max_ms", 2000.0, "Reject R-R intervals longer than this."),
            ("rr_max_rel_change", 0.25,
             "Max deviation from the recent median before a beat is rejected."),
            ("high_rate_queue_depth", 400, "QoS depth for the ECG and ACC topics."),
            ("low_rate_queue_depth", 50, "QoS depth for HR / R-R / HRV / status."),
        )
        for name, default, description in params:
            self.declare_parameter(name, default, ParameterDescriptor(description=description))
            setattr(self, "p_" + name, self.get_parameter(name).value)

        # -- publishers -----------------------------------------------------
        fast = QoSProfile(depth=int(self.p_high_rate_queue_depth),
                          reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST)
        slow = QoSProfile(depth=int(self.p_low_rate_queue_depth),
                          reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST)

        def topic(name: str) -> str:
            prefix = self.p_topic_prefix.strip("/")
            return "{}/{}".format(prefix, name) if prefix else name

        self.pub_hr = self.create_publisher(PointStamped, topic("heart_rate"), slow)
        self.pub_rr = self.create_publisher(PointStamped, topic("rr_interval"), slow)
        self.pub_rr_filtered = self.create_publisher(
            PointStamped, topic("rr_interval_filtered"), slow)
        self.pub_ecg = self.create_publisher(PointStamped, topic("ecg"), fast)
        self.pub_acc = self.create_publisher(Vector3Stamped, topic("acc"), fast)
        self.pub_contact = self.create_publisher(Bool, topic("skin_contact"), slow)
        self.pub_battery = self.create_publisher(Float32, topic("battery"), slow)
        self.pub_status = self.create_publisher(DiagnosticArray, topic("status"), slow)
        self.pub_hrv = {
            key: self.create_publisher(PointStamped, topic("hrv/" + key), slow)
            for key in HRV_KEYS
        }

        # -- state ----------------------------------------------------------
        self.anchor = RosTimeAnchor()
        self.ecg_timebase = FrameTimebase(float(self.p_ecg_sample_rate))
        self.acc_timebase = FrameTimebase(float(self.p_acc_sample_rate))
        self.hrv = HrvAnalyzer(
            window_sec=float(self.p_hrv_window_sec),
            min_beats=int(self.p_hrv_min_beats),
            rr_min_ms=float(self.p_rr_min_ms),
            rr_max_ms=float(self.p_rr_max_ms),
            max_rel_change=float(self.p_rr_max_rel_change),
        )

        self._acc_scale = 1e-3 if self.p_acc_in_g else 1e-3 * STANDARD_GRAVITY
        self._connected = False
        self._device_address = ""
        self._battery_percent: Optional[int] = None
        self._contact: Optional[bool] = None
        self._counts = {"ecg": 0, "acc": 0, "beats": 0, "rr_rejected": 0, "frame_errors": 0}
        self._last_diag_ns = self._now_ns()
        self._last_counts = dict(self._counts)

        self.create_timer(1.0, self._on_timer)

        self.get_logger().info(
            "Polar H10 driver ready - publishing under '{}/' (ECG={}, ACC={}, HRV={})".format(
                self.p_topic_prefix.strip("/"), self.p_publish_ecg,
                self.p_publish_acc, self.p_publish_hrv))

    # -- helpers ------------------------------------------------------------

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _stamp(self, ns: Optional[int]):
        if ns is None or ns < 0:
            ns = self._now_ns()
        return RclTime(nanoseconds=int(ns)).to_msg()

    def _point(self, ns: Optional[int], value: float) -> PointStamped:
        msg = PointStamped()
        msg.header.stamp = self._stamp(ns)
        msg.header.frame_id = self.p_frame_id
        msg.point.x = float(value)
        return msg

    # -- link callbacks -----------------------------------------------------

    def on_connected(self, address: str) -> None:
        self._connected = True
        self._device_address = address
        self._contact = None
        self.anchor.reset()
        self.ecg_timebase.reset()
        self.acc_timebase.reset()
        self.hrv.reset()

    def on_disconnected(self) -> None:
        self._connected = False

    def on_contact_gained(self) -> None:
        self._contact = True
        self.get_logger().info("Skin contact detected")
        self.pub_contact.publish(Bool(data=True))

    def on_contact_lost(self) -> None:
        self._contact = False
        self.get_logger().warn("Skin contact lost")
        self.pub_contact.publish(Bool(data=False))

    def on_battery(self, percent: int) -> None:
        """Publish the battery level as a percentage of full charge, 0..100."""
        self._battery_percent = percent
        self.pub_battery.publish(Float32(data=float(percent)))
        self.get_logger().info("Battery level {}%".format(percent))

    def on_heartbeat(self, frame) -> None:
        """
        Handle a bleakheart HeartRate callback (unpack=True): one call per beat.

        Frame layout is ``('HR', t_ns, (hr_bpm, rr_ms), energy)``.
        """
        try:
            _, host_ns, (hr_bpm, rr_ms), _ = frame
            beat_ns = self.anchor.to_ros(host_ns, self._now_ns())
            self._counts["beats"] += 1

            self.pub_hr.publish(self._point(beat_ns, hr_bpm))
            self.pub_rr.publish(self._point(beat_ns, rr_ms))

            if not self.p_publish_hrv:
                return

            metrics = self.hrv.add_beat(beat_ns, float(rr_ms))
            if not self.hrv.last_accepted:
                self._counts["rr_rejected"] += 1
                return

            self.pub_rr_filtered.publish(self._point(beat_ns, rr_ms))
            if metrics is None:
                return  # window not full yet
            for key, publisher in self.pub_hrv.items():
                publisher.publish(self._point(beat_ns, metrics[key]))
        except Exception as exc:  # noqa: BLE001
            self._counts["frame_errors"] += 1
            self.get_logger().error("Heart rate callback failed: {}".format(exc),
                                    throttle_duration_sec=5.0)

    def on_measurement(self, frame) -> None:
        """
        Handle a bleakheart PolarMeasurementData callback, ``(kind, t_ns, payload)``.

        ``t_ns`` refers to the *last* sample of the frame; the remaining
        sample times are reconstructed by :class:`FrameTimebase`.
        """
        try:
            kind, host_ns, payload = frame
            ros_now = self._now_ns()
            last_sample_ns = self.anchor.to_ros(host_ns, ros_now)

            if kind == "ECG":
                times = self.ecg_timebase.sample_times(last_sample_ns, len(payload))
                for micro_volt, stamp_ns in zip(payload, times):
                    self.pub_ecg.publish(self._point(stamp_ns, micro_volt))
                self._counts["ecg"] += len(payload)

            elif kind == "ACC":
                times = self.acc_timebase.sample_times(last_sample_ns, len(payload))
                for (x, y, z), stamp_ns in zip(payload, times):
                    msg = Vector3Stamped()
                    msg.header.stamp = self._stamp(stamp_ns)
                    msg.header.frame_id = self.p_frame_id
                    msg.vector.x = x * self._acc_scale
                    msg.vector.y = y * self._acc_scale
                    msg.vector.z = z * self._acc_scale
                    self.pub_acc.publish(msg)
                self._counts["acc"] += len(payload)

            else:
                self.get_logger().warn("Ignoring unexpected {} frame".format(kind),
                                       throttle_duration_sec=30.0)
        except Exception as exc:  # noqa: BLE001
            self._counts["frame_errors"] += 1
            self.get_logger().error("Measurement callback failed: {}".format(exc),
                                    throttle_duration_sec=5.0)

    # -- periodic -----------------------------------------------------------

    def _on_timer(self) -> None:
        # Skin contact is edge-triggered by bleakheart. Republishing the
        # latched state once a second guarantees that a bag started at an
        # arbitrary moment still contains it.
        if self._contact is not None:
            self.pub_contact.publish(Bool(data=bool(self._contact)))
        self._publish_diagnostics()

    def _publish_diagnostics(self) -> None:
        now_ns = self._now_ns()
        dt = max(1e-6, (now_ns - self._last_diag_ns) * 1e-9)
        rates = {k: (self._counts[k] - self._last_counts[k]) / dt for k in self._counts}
        self._last_counts = dict(self._counts)
        self._last_diag_ns = now_ns

        status = DiagnosticStatus()
        status.name = "polar_h10"
        status.hardware_id = self._device_address or "unknown"
        if not self._connected:
            status.level = DiagnosticStatus.ERROR
            status.message = "disconnected"
        elif self._contact is False:
            status.level = DiagnosticStatus.WARN
            status.message = "connected, no skin contact"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "streaming"

        delta = self.anchor.delta_ns
        delta_text = "n/a" if delta is None else "{:.6f}".format(delta * 1e-9)
        status.values = [
            KeyValue(key="connected", value=str(self._connected)),
            KeyValue(key="skin_contact", value=str(self._contact)),
            KeyValue(key="battery_percent", value=str(self._battery_percent)),
            KeyValue(key="ecg_hz", value="{:.1f}".format(rates["ecg"])),
            KeyValue(key="acc_hz", value="{:.1f}".format(rates["acc"])),
            KeyValue(key="beats_hz", value="{:.2f}".format(rates["beats"])),
            KeyValue(key="beats_total", value=str(self._counts["beats"])),
            KeyValue(key="rr_rejected", value=str(self._counts["rr_rejected"])),
            KeyValue(key="frame_errors", value=str(self._counts["frame_errors"])),
            KeyValue(key="ros_clock_delta_s", value=delta_text),
            KeyValue(key="clock_reanchors", value=str(self.anchor.reanchor_count)),
        ]

        msg = DiagnosticArray()
        msg.header.stamp = self._stamp(now_ns)
        msg.header.frame_id = self.p_frame_id
        msg.status = [status]
        self.pub_status.publish(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def main(args=None) -> None:
    rclpy.init(args=args)

    # The BLE stack gets its own event loop on a dedicated thread while rclpy
    # spins the executor on the main thread.
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
    loop_thread.start()

    node = PolarH10Node()
    link = PolarH10Link(node)
    task = asyncio.run_coroutine_threadsafe(link.run(), loop)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C reaches us as either of these depending on whether rclpy's
        # signal handler got there first; both are a normal shutdown.
        node.get_logger().info("Shutting down ...")
    finally:
        loop.call_soon_threadsafe(link.shutdown.set)
        try:
            # Give the link time to stop the ECG/ACC streams; the H10 stays
            # powered on until its battery dies if we skip this.
            task.result(timeout=15.0)
        except Exception as exc:  # noqa: BLE001
            node.get_logger().warn("BLE task did not stop cleanly: {}".format(exc))

        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=3.0)
        loop.close()

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
