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

# Pure-logic tests for the Polar drivers: the parts that are implemented here
# rather than in bleakheart -- per-sample timestamp reconstruction, ROS clock
# anchoring, HRV, and the PPG/PPI payloads bleakheart does not decode.
# No hardware and no ROS graph needed.

import math

import pytest

from polar_oh1.polar_h10_node import FrameTimebase, HrvAnalyzer, RosTimeAnchor
from polar_oh1.polar_oh1_node import (
    PPI_FLAG_CONTACT_UNSUPPORTED,
    PPI_FLAG_INVALID,
    PPI_FLAG_SKIN_CONTACT,
    PPI_SAMPLE,
    decode_ppg_uncompressed,
)

NOMINAL_ECG_DELTA = round(1e9 / 130)


# --------------------------------------------------------------------------
# Per-sample timestamps
# --------------------------------------------------------------------------

def test_frame_timebase_first_frame_uses_nominal_rate():
    timebase = FrameTimebase(130.0)
    times = timebase.sample_times(1_000_000_000, 10)
    assert len(times) == 10
    # The frame timestamp refers to the LAST sample.
    assert times[-1] == 1_000_000_000
    assert times[1] - times[0] == NOMINAL_ECG_DELTA


def test_frame_timebase_tracks_measured_rate_and_rejects_outliers():
    timebase = FrameTimebase(130.0)
    first = timebase.sample_times(1_000_000_000, 10)
    second = timebase.sample_times(1_076_923_080, 10)
    assert second[0] > first[-1]
    assert abs((second[1] - second[0]) - NOMINAL_ECG_DELTA) < 200

    # A gap far outside tolerance must fall back to the nominal rate.
    third = timebase.sample_times(6_076_923_080, 10)
    assert third[1] - third[0] == NOMINAL_ECG_DELTA


def test_frame_timebase_reset_clears_history():
    timebase = FrameTimebase(130.0)
    timebase.sample_times(1_000_000_000, 10)
    timebase.reset()
    times = timebase.sample_times(9_000_000_000, 10)
    assert times[1] - times[0] == NOMINAL_ECG_DELTA


# --------------------------------------------------------------------------
# ROS clock anchoring
# --------------------------------------------------------------------------

def test_anchor_is_identity_when_ros_and_system_clocks_agree():
    anchor = RosTimeAnchor()
    assert anchor.to_ros(1_000_000_000, 1_000_000_000) == 1_000_000_000
    assert anchor.delta_ns == 0
    # A later frame keeps the same delta, so sensor-side spacing is preserved.
    assert anchor.to_ros(1_500_000_000, 1_500_000_100) == 1_500_000_000


def test_anchor_applies_constant_ros_offset():
    anchor = RosTimeAnchor()
    anchor.to_ros(1_000_000_000, 4_000_000_000)  # ROS clock 3 s ahead
    assert anchor.delta_ns == 3_000_000_000
    assert anchor.to_ros(2_000_000_000, 5_000_000_000) == 5_000_000_000


def test_anchor_reanchors_when_the_sensor_loses_its_epoch():
    # The H10 resets its clock about a minute after leaving the strap, which
    # is exactly what happens when the strap is handed to the next subject.
    anchor = RosTimeAnchor(max_error_ns=5_000_000_000)
    anchor.to_ros(1_000_000_000, 1_000_000_000)
    mapped = anchor.to_ros(500_000_000_000_000, 2_000_000_000)
    assert anchor.reanchor_count == 1
    assert mapped == 2_000_000_000


# --------------------------------------------------------------------------
# HRV
# --------------------------------------------------------------------------

def _feed(analyzer, intervals):
    now, last = 0, None
    for interval in intervals:
        now += int(interval * 1e6)
        result = analyzer.add_beat(now, float(interval))
        if result is not None:
            last = result
    return last


def test_hrv_matches_reference_rmssd():
    analyzer = HrvAnalyzer(60.0, 5, 300.0, 2000.0, 0.25)
    intervals = [800, 820, 790, 810, 800, 830, 795, 805]
    result = _feed(analyzer, intervals)

    diffs = [intervals[i + 1] - intervals[i] for i in range(len(intervals) - 1)]
    expected = math.sqrt(sum(d * d for d in diffs) / len(diffs))

    assert result is not None
    assert result['rmssd'] == pytest.approx(expected)
    assert result['quality'] == pytest.approx(1.0)
    assert 70.0 < result['mean_hr'] < 78.0


def test_hrv_rejects_dropped_beat_artifact():
    analyzer = HrvAnalyzer(60.0, 5, 300.0, 2000.0, 0.25)
    # The 1600 ms interval is a missed beat; without rejection it would
    # dominate RMSSD completely.
    result = _feed(analyzer, [800, 800, 800, 800, 800, 1600, 800, 800, 800])
    assert result is not None
    assert result['rmssd'] < 1.0
    assert result['quality'] < 1.0


def test_hrv_reports_per_beat_acceptance():
    analyzer = HrvAnalyzer(60.0, 5, 300.0, 2000.0, 0.25)
    analyzer.add_beat(0, 800.0)
    assert analyzer.last_accepted
    analyzer.add_beat(100_000_000, 5000.0)  # outside rr_max_ms
    assert not analyzer.last_accepted
    analyzer.add_beat(200_000_000, 810.0)
    assert analyzer.last_accepted


def test_hrv_reports_window_provenance():
    analyzer = HrvAnalyzer(60.0, 5, 300.0, 2000.0, 0.25)
    result = _feed(analyzer, [800] * 12)
    assert result is not None
    assert result['n_beats'] == 12.0
    # 12 beats of 800 ms span 8.8 s between the first and the last.
    assert result['window_sec'] == pytest.approx(8.8, abs=0.01)


def test_hrv_window_drops_beats_that_fell_out_of_it():
    # A 10 s window over 800 ms beats can hold at most 13 of them.
    analyzer = HrvAnalyzer(10.0, 5, 300.0, 2000.0, 0.25)
    result = _feed(analyzer, [800] * 40)
    assert result is not None
    assert result['n_beats'] <= 13
    assert result['window_sec'] <= 10.0


# --------------------------------------------------------------------------
# Payloads bleakheart does not decode
# --------------------------------------------------------------------------

def test_ppg_uncompressed_is_24bit_signed_little_endian():
    payload = b''.join(
        v.to_bytes(3, 'little', signed=True) for v in (-517445, 10, 20, 30))
    assert decode_ppg_uncompressed(payload) == [[-517445, 10, 20, 30]]


def test_ppg_uncompressed_rejects_bad_length():
    with pytest.raises(ValueError):
        decode_ppg_uncompressed(b'\x00' * 11)


def _ppi(hr, interval, error, flags):
    return bytes([hr]) + interval.to_bytes(2, 'little') \
        + error.to_bytes(2, 'little') + bytes([flags])


def test_ppi_sample_layout():
    hr, interval, error, flags = PPI_SAMPLE.unpack(
        _ppi(75, 800, 5, PPI_FLAG_SKIN_CONTACT))
    assert (hr, interval, error) == (75, 800, 5)
    assert not flags & PPI_FLAG_INVALID
    assert flags & PPI_FLAG_SKIN_CONTACT


def test_ppi_flags_identify_unusable_samples():
    # Polar: discard when the "not valid" bit is set, or when contact
    # detection is supported but reports no skin contact.
    _, _, _, invalid = PPI_SAMPLE.unpack(_ppi(75, 1500, 5, PPI_FLAG_INVALID))
    assert invalid & PPI_FLAG_INVALID

    _, _, _, no_contact = PPI_SAMPLE.unpack(_ppi(75, 810, 5, 0x00))
    assert not no_contact & PPI_FLAG_CONTACT_UNSUPPORTED
    assert not no_contact & PPI_FLAG_SKIN_CONTACT
