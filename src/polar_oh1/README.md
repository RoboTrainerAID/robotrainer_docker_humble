# Polar heart rate sensor drivers

ROS 2 (Humble) drivers for the **Polar H10** chest strap and the **Polar OH1+**
arm strap.

| | `polar_h10_node` | `polar_oh1_node` |
|---|---|---|
| Signals | HR, R-R, ECG, ACC, battery, skin contact, HRV | HR, PPG, PPI, HRV |
| HRV source | ECG-derived R-R | PPG-derived PPI |
| Valid during movement | yes | **no** (see below) |
| Timestamps | hardware, per sample | receive time |

> **Which one to use:** Polar documents that PPI-derived HRV "cannot be
> measured accurately during activity, it should only be used at complete
> rest". For anything involving a moving subject, use the H10.

## Requirements

BLE and the Polar Measurement Data protocol are handled by
[bleakheart](https://github.com/fsmeraldi/bleakheart) — already in the Docker
image, otherwise `pip install bleak bleakheart` (needs Python ≥ 3.10).

These drivers add only what bleakheart does not provide: per-sample
timestamps inside a PMD frame, ROS clock anchoring, HRV, the PPI/PPG payloads
bleakheart has no decoder for, and the ROS interface.

Every topic uses a stock message type, so everything survives a
`ros1_bridge` hop. Recording is left to you.

---

## `polar_h10_node`

```bash
ros2 launch polar_oh1 polar_h10.launch.py
ros2 launch polar_oh1 polar_h10.launch.py address:=A0:9E:1A:E0:BC:97
```

Leave `address` empty to connect to the first advertised `Polar H10`. The
driver logs every sensor it sees during the scan, so run it once to find yours.

### Topics

Prefixed with `topic_prefix` (default `biosensors/polar_h10`). Scalars use
`geometry_msgs/PointStamped` with the value in `point.x` — the standard-message
way to carry a timestamped number.

| Topic | Type | Value | Rate |
|---|---|---|---|
| `heart_rate` | `PointStamped` | bpm | per beat |
| `rr_interval` | `PointStamped` | ms, every beat | per beat |
| `rr_interval_filtered` | `PointStamped` | ms, artifact-free beats only | per beat |
| `ecg` | `PointStamped` | µV | 130 Hz |
| `acc` | `Vector3Stamped` | m/s² (or g, see `acc_in_g`) | 200 Hz |
| `hrv/rmssd` | `PointStamped` | ms | per beat |
| `hrv/sdnn` | `PointStamped` | ms | per beat |
| `hrv/pnn50` | `PointStamped` | % | per beat |
| `hrv/mean_rr` | `PointStamped` | ms | per beat |
| `hrv/mean_hr` | `PointStamped` | bpm | per beat |
| `hrv/quality` | `PointStamped` | 0..1, fraction of beats accepted | per beat |
| `hrv/n_beats` | `PointStamped` | beats behind this value | per beat |
| `hrv/window_sec` | `PointStamped` | seconds this value actually covers | per beat |
| `skin_contact` | `std_msgs/Bool` | electrode contact | 1 Hz |
| `battery` | `std_msgs/Float32` | percent of full charge, 0..100 | 1/60 Hz |
| `status` | `diagnostic_msgs/DiagnosticArray` | rates, clock delta, error counts | 1 Hz |

Heart rate is published per beat rather than at a fixed 1 Hz. One consequence
worth knowing: bleakheart emits a callback only for frames that carry an R-R
interval, so heart rate pauses while the strap has no skin contact. Watch
`skin_contact` and `status` to tell that apart from a dead link.

### HRV latency — what is and is not delayed

**There is no publishing delay.** A new HRV value is emitted the moment an
accepted beat arrives and is stamped with *that beat's* time. If you stop
recording at the end of a trial, the last HRV message in the bag is at most
one R-R interval old (plus ~100 ms of BLE latency).

**What the window does mean** is that each value looks *backward* over
`hrv_window_sec`. With the 30 s default and a 10 s trial, the value published
at the end of the trial is computed over the trial plus the 20 s before it.
That is a window/trial mismatch, not a lag. Three ways to handle it:

1. **Compute HRV offline from `rr_interval` (recommended).** Every R-R
   interval is published with its own beat timestamp, so you can cut exact
   per-trial windows afterwards and re-run with any algorithm you like. This
   is the defensible option for a paper, and it is why the raw topic exists.
2. **Shorten the window** to roughly the trial length. Respect the sizing
   rule: at ~70 bpm you get ~1.2 beats/s, so `hrv_window_sec × 1.2` must
   comfortably exceed `hrv_min_beats` or nothing is published at all. A 10 s
   window yields ~12 beats — ultra-short-term RMSSD is used in the literature
   but is noticeably noisier, and 10 s SDNN is *not* comparable to the
   standard 5-minute SDNN. Label it accordingly.
3. **Keep 30 s for the live trend** and use `hrv/window_sec` and
   `hrv/n_beats` to know exactly what each value covers, then filter on them
   during analysis.

Artifact rejection runs before any of this, because a single missed or doubled
beat is enough to wreck RMSSD: absolute plausibility (`rr_min_ms` ..
`rr_max_ms`) plus relative change against the median of the recent accepted
beats (`rr_max_rel_change`). Successive differences are only formed between
beats that are both accepted **and** adjacent, so a rejected beat breaks the
chain instead of creating a huge fake difference. If `hrv/quality` drops well
below 1.0 the strap contact is poor and the values should not be trusted.

Note that bleakheart rounds R-R to whole milliseconds (Polar transmits
1/1024 s units). This matches how Polar's own tools export R-R and is
negligible for RMSSD.

### Timestamps

Every sample is published individually and stamped with the time it was
**acquired**, not the time its BLE frame arrived:

* ECG and ACC frames carry a hardware timestamp of the frame's *last* sample.
  The inter-sample spacing is measured from consecutive frame timestamps
  (falling back to the nominal rate for the first frame), as the official
  Polar SDK does it.
* Those timestamps are anchored to **`node.get_clock()`** — the same ROS clock
  as every other publisher and as the bridge — via a constant delta captured
  on the first frame of each connection. The BLE latency of that first frame
  is baked in as a constant bias of roughly 100 ms, and the sensor's crystal
  drift accumulates a few tens of milliseconds per hour. Both are fine at the
  precision this setup needs.
* **Multi-subject sessions are handled.** The H10 forgets its own clock about
  a minute after coming off the strap, so passing it to the next subject
  resets its epoch. A fresh bleakheart session is built on every reconnection
  so the offset is recomputed, and a guard re-anchors anything implausible.
  `clock_reanchors` in the `status` topic counts these; a handful over a
  session is normal, a steadily rising number is not.

**BLE frames still arrive in bursts** of 10–20 samples — that is the radio
link and no driver changes it. What is guaranteed is that `header.stamp` is
correct, so the recorded signal is evenly sampled even though bag *receive*
times are bursty. **Sort by `header.stamp` when you analyse**, not by receive
time.

### Bridging to ROS 1

The driver deliberately sticks to long-stable message types whose definitions
are the same on both sides, so the md5 sums match and the bridge carries every
topic: `geometry_msgs/PointStamped`, `geometry_msgs/Vector3Stamped`,
`std_msgs/Float32`, `std_msgs/Bool`, `diagnostic_msgs/DiagnosticArray`.

That is why `battery` is a bare `std_msgs/Float32` and **not**
`sensor_msgs/BatteryState`. `BatteryState`'s definition has changed across
sensor_msgs versions, so a ROS 1 subscriber and the bridge can disagree on its
md5 sum, and the subscriber then refuses the topic outright:

```
[ERROR] Client [...] wants topic /biosensors/polar_h10/battery to have
datatype/md5sum [sensor_msgs/BatteryState/476f837f...], but our version has
[sensor_msgs/BatteryState/4ddae7f0...]. Dropping connection.
```

Nothing is lost — the H10 reports a charge percentage and nothing else, so
`percentage` was the only field of `BatteryState` that was ever populated.

With ECG and ACC both on you are pushing **330 msg/s** of small messages
through the bridge. That is a lot of per-message overhead and the most likely
place to lose samples. If you do not need chest acceleration, set
`publish_acc: false` — it alone is 200 msg/s. Check `ecg_hz` and `acc_hz` in
the `status` topic against 130 and 200 to confirm nothing is being dropped
upstream of the bridge.

#### Subscriber queue depth — do not use `dynamic_bridge`

**A ROS 1 bag recorded through `dynamic_bridge` keeps only about a third of the
ECG.** Measured on this setup with a 10 s recording:

| Measured at | ECG rate | Delivered |
|---|---|---|
| `status/ecg_hz`, i.e. what the driver publishes | 131 Hz | — |
| `ros2 bag record`, ROS 2 side | 125–131 Hz | ~100 % |
| ROS 2 subscriber, QoS depth 100 or more | 133 Hz | ~100 % |
| ROS 2 subscriber, QoS **depth 10** | 45 Hz | **35 %** |
| ROS 1 bag via `dynamic_bridge` | 51 Hz | 39 % |

The cause is burstiness, not throughput. BLE hands over ECG in frames of 10–20
samples, and each sample is published as its own message, so ~15 messages hit
the wire within microseconds. A `KEEP_LAST(10)` subscription overflows
mid-burst and silently discards the remainder — `RELIABLE` does not help,
because keep-last is *allowed* to overwrite. `dynamic_bridge` has no
queue-size option and takes the library default of 10
(`size_t queue_size = 10` in `ros1_bridge/bridge.hpp`), which is exactly the
losing case. This affects `ros2 topic echo` and `rqt` too, and a bag recorded
this way looks like a sensor problem while `ecg_hz` sits at 131.

Use **`parameter_bridge`**, which reads a per-topic `queue_size` (default 100)
and applies it to both its ROS 2 subscription and its ROS 1 publisher. In
`robotrainer_docker_ros1_bridge` this is `bridge_topics.yaml` plus
`./detached_parameter_bridge.sh` — the ECG and ACC entries carry
`queue_size: 500`. The trade-off is that only listed topics cross the bridge,
so anything else you need has to be added to that file.

Recording on the ROS 2 side (`ros2 bag record`) captures everything without
any of this.

### Parameters

See `config/polar_h10.yaml`; every parameter carries a description
(`ros2 param describe /polar_h10_node <name>`).

### Operating notes

* **Always shut the node down cleanly** (Ctrl-C). Polar documents that the H10
  keeps running until its battery is flat if ECG/ACC streaming is not stopped
  by the host. The driver stops both streams on exit.
* The H10 supports two simultaneous BLE connections, so you can run this
  driver and the Polar Flow app together to cross-check readings.
* **Battery level comes from whichever source the host exposes.** Normally it
  is the GATT Battery Level characteristic (`0x2a19`). Where `bluetoothd`'s
  battery plugin claims the Battery Service for itself, that characteristic is
  not re-exported on the GATT D-Bus API and bleak reports *"Characteristic
  00002a19-… was not found!"*; the driver then reads the `Percentage` property
  of `org.bluez.Battery1` on the device object instead and logs which source it
  settled on. To check by hand:

  ```bash
  dbus-send --system --print-reply --dest=org.bluez \
      /org/bluez/hci1/dev_24_AC_AC_1E_C2_03 \
      org.freedesktop.DBus.Properties.GetAll string:org.bluez.Battery1
  ```

  This needs the system bus inside the container — `-v /var/run/dbus:/var/run/dbus`,
  which the run scripts already mount. Everything else in the driver is
  unaffected: the battery level is the only signal BlueZ intercepts.
* Moisten the electrodes. Wash the strap regularly. Worn electrodes give
  unreliable readings even when the strap looks fine.
* Polar warns that motors, LED displays and electrical brakes interfere with
  the link — mount the Bluetooth dongle away from motor controllers and in
  front of the user, not in a rear electronics box. Use the `adapter`
  parameter to pin the driver to a dedicated dongle (`hci1`).
* On a subject change the driver reconnects by itself; no restart needed. Give
  it a warm-up until `skin_contact` is true and `hrv/quality` settles.

---

## `polar_oh1_node`

```bash
ros2 launch polar_oh1 ros2-polar_oh1.launch.py
```

Topics and message types are unchanged from the original version:
`biosensors/polar_oh1/{hr, ppg_ch0..3, ppi, hrv}`
(`hrv` is a `Float32MultiArray` of `[RMSSD, SDNN]` in ms).

Parameters: `Device_Mac_Address`, `adapter`, `publish_ppg`, `publish_ppi`,
`ppg_decimation`, `reconnect_delay`, `ppi_max_error_ms`, `hrv_window_sec`,
`hrv_min_beats`.

### What was fixed

* **Control point responses are checked** (via bleakheart). Start/stop commands
  were previously written and never verified, so a rejected stream produced a
  node that reported "connected" and streamed nothing.
* **PPI validity flags are honoured.** Polar specifies that samples with the
  "not valid" bit set, or without skin contact, must be discarded. Feeding
  them into RMSSD was the direct cause of the unstable HRV. Samples whose
  device-reported error estimate exceeds `ppi_max_error_ms` are dropped too.
* **PPG is no longer silently decimated by 4.** The old code kept every 4th
  sample — 135 Hz became ~34 Hz — under a comment claiming it skipped every
  second sample. Set `ppg_decimation` if you want that back.
* **`rclpy` actually spins.** `asyncio.run()` used to block the main thread so
  the executor never ran; parameters, services and timers were dead.
* **Logging left the BLE callback,** where it stalled the asyncio loop that
  also services D-Bus. Statistics are summarised every 10 s instead.
* Streams are stopped and the sensor disconnected on shutdown; reconnection is
  event-driven instead of a 5 s poll; exceptions in notification callbacks are
  logged instead of being swallowed.

### Known caveat

If the OH1 sends *delta-compressed* PPG, decoding is done by bleakheart, whose
delta unpacking reads the bit stream MSB first while the official Polar SDK
(`BlePMDClient.parseDeltaFrame`) reads it LSB first. The driver logs a warning
once when that path is taken. Uncompressed PPG is decoded here and is
unaffected.

### Remaining hardware limitations

Documented Polar behaviour, not driver faults:

* Enabling PPI switches the sensor to a different algorithm — heart rate then
  updates only **every 5 s**, and the first PPI batch takes about **25 s** to
  arrive. Set `publish_ppi: false` if you only want PPG and a responsive HR.
* PPI-derived HRV is only valid at complete rest.
* OH1 skin-contact detection is unreliable and should not be trusted.

---

## Tests

```bash
colcon test --packages-select polar_oh1
```

`test/test_polar_h10_decoding.py` covers timestamp reconstruction, ROS clock
anchoring including the sensor-epoch reset, HRV and its artifact rejection,
and the PPG/PPI payloads decoded here — all without hardware.
