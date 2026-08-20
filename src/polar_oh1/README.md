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
| `battery` | `sensor_msgs/BatteryState` | `percentage` 0..1 | 1/60 Hz |
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

All message types exist unchanged in ROS 1 (`geometry_msgs/PointStamped`,
`geometry_msgs/Vector3Stamped`, `sensor_msgs/BatteryState`, `std_msgs/Bool`,
`diagnostic_msgs/DiagnosticArray`).

With ECG and ACC both on you are pushing **330 msg/s** of small messages
through the bridge. That is a lot of per-message overhead and the most likely
place to lose samples. If you do not need chest acceleration, set
`publish_acc: false` — it alone is 200 msg/s. Check `ecg_hz` and `acc_hz` in
the `status` topic against 130 and 200 to confirm nothing is being dropped
upstream of the bridge.

### Parameters

See `config/polar_h10.yaml`; every parameter carries a description
(`ros2 param describe /polar_h10_node <name>`).

### Operating notes

* **Always shut the node down cleanly** (Ctrl-C). Polar documents that the H10
  keeps running until its battery is flat if ECG/ACC streaming is not stopped
  by the host. The driver stops both streams on exit.
* The H10 supports two simultaneous BLE connections, so you can run this
  driver and the Polar Flow app together to cross-check readings.
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
