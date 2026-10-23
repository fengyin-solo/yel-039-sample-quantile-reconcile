#!/usr/bin/env python3
"""Recover hidden channel assignments from domain fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path


TOP_KEYS = {"survey_id", "limits", "profiles", "readings"}
LIMIT_KEYS = {"max_slots", "residual_ceiling"}
PROFILE_KEYS = {"sample_id", "fingerprint"}
READING_KEYS = {'sample_n', 'trim_loss', 'scaling', 'sample_sq', 'slot', 'sample_sum'}
OUTPUT_KEYS = {"survey_id", "status", "sample_assignments", "total_residual", "accepted", "diagnostic"}
ASSIGNMENT_KEYS = {"slot", "sample_id", "residual"}
LEDGER_KEYS = ['survey_id', 'input_sha256', 'result_sha256', 'total_residual']
LEDGER_ORDER = ['survey_id', 'input_sha256', 'result_sha256', 'total_residual']


class Refusal(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def visible_identifier(value, minimum=2, maximum=40):
    return (
        isinstance(value, str)
        and minimum <= len(value) <= maximum
        and value == value.strip()
        and all(ord(char) >= 32 and ord(char) != 127 for char in value)
    )


def pairs_hook(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refusal("统计拒收: BAD_SAMPLE")
        result[key] = value
    return result


def compute_features(reading):
    sample_sum = reading["sample_sum"]
    sample_sq = reading["sample_sq"]
    sample_n = reading["sample_n"]
    trim_loss = reading["trim_loss"]
    scaling = reading["scaling"]
    count = sample_n + 2.0 * trim_loss
    mean = sample_sum / count
    variance = sample_sq / count - mean * mean
    features = [mean / scaling, math.sqrt(max(variance, 0.0)) / scaling, trim_loss / (count + 1.0)]
    if not all(finite(value) and 0.0 < value <= 1_000_000_000.0 for value in features):
        raise Refusal("统计拒收: BAD_SAMPLE")
    return tuple(round(float(value), 12) for value in features)


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        raise Refusal("统计拒收: BAD_SAMPLE")
    case_id = payload["survey_id"]
    if not visible_identifier(case_id):
        raise Refusal("统计拒收: BAD_SAMPLE")
    limits = payload["limits"]
    if not isinstance(limits, dict) or set(limits) != LIMIT_KEYS:
        raise Refusal("统计拒收: BAD_SAMPLE")
    max_slots = limits["max_slots"]
    ceiling = limits["residual_ceiling"]
    if isinstance(max_slots, bool) or not isinstance(max_slots, int) or not 3 <= max_slots <= 12:
        raise Refusal("统计拒收: BAD_SAMPLE")
    if not finite(ceiling) or not 0.0 < ceiling <= 1.0:
        raise Refusal("统计拒收: BAD_SAMPLE")
    profiles = payload["profiles"]
    readings = payload["readings"]
    if not isinstance(profiles, list) or not isinstance(readings, list):
        raise Refusal("统计拒收: BAD_SAMPLE")
    if not 3 <= len(profiles) <= max_slots or len(readings) != len(profiles):
        raise Refusal("统计拒收: BAD_SAMPLE")
    clean_profiles = []
    seen_profiles = set()
    for record in profiles:
        if not isinstance(record, dict) or set(record) != PROFILE_KEYS:
            raise Refusal("统计拒收: BAD_SAMPLE")
        identifier = record["sample_id"]
        if not visible_identifier(identifier, 2, 32) or identifier in seen_profiles:
            raise Refusal("统计拒收: BAD_SAMPLE")
        seen_profiles.add(identifier)
        fingerprint = record["fingerprint"]
        if not isinstance(fingerprint, list) or len(fingerprint) != 3:
            raise Refusal("统计拒收: BAD_SAMPLE")
        if not all(finite(value) and 0.0 < value <= 1_000_000_000.0 for value in fingerprint):
            raise Refusal("统计拒收: BAD_SAMPLE")
        clean_profiles.append((identifier, tuple(float(value) for value in fingerprint)))
    clean_readings = []
    seen_slots = set()
    for record in readings:
        if not isinstance(record, dict) or set(record) != READING_KEYS:
            raise Refusal("统计拒收: BAD_SAMPLE")
        slot = record["slot"]
        if not visible_identifier(slot, 2, 32) or slot in seen_slots:
            raise Refusal("统计拒收: BAD_SAMPLE")
        seen_slots.add(slot)
        if not all(finite(record[name]) and record[name] > 0.0 for name in ['sample_sum', 'sample_sq', 'sample_n', 'trim_loss', 'scaling']):
            raise Refusal("统计拒收: BAD_SAMPLE")
        clean_readings.append((slot, compute_features(record)))
    clean_readings.sort(key=lambda item: item[0])
    return case_id, ceiling, clean_profiles, clean_readings


def normalized_distance(left, right):
    total = 0.0
    for a, b in zip(left, right):
        scale = abs(a) + abs(b) + 1.0e-12
        total += ((a - b) / scale) ** 2
    return total


def assign(profiles, readings):
    distances = [
        [normalized_distance(reading[1], profile[1]) for profile in profiles]
        for reading in readings
    ]
    count = len(profiles)

    @lru_cache(maxsize=None)
    def best(index, used):
        if index == count:
            return (0.0, ())
        choice = None
        for profile_index in range(count):
            if used & (1 << profile_index):
                continue
            tail_cost, tail_sequence = best(index + 1, used | (1 << profile_index))
            candidate = (
                distances[index][profile_index] + tail_cost,
                (profile_index,) + tail_sequence,
            )
            if choice is None:
                choice = candidate
                continue
            if candidate[0] < choice[0] - 1.0e-12:
                choice = candidate
            elif abs(candidate[0] - choice[0]) <= 1.0e-12:
                candidate_names = tuple(profiles[item][0] for item in candidate[1])
                choice_names = tuple(profiles[item][0] for item in choice[1])
                if candidate_names < choice_names:
                    choice = candidate
        return choice

    total, sequence = best(0, 0)
    rows = []
    for index, profile_index in enumerate(sequence):
        rows.append(
            {
                "slot": readings[index][0],
                "sample_id": profiles[profile_index][0],
                "residual": distances[index][profile_index],
            }
        )
    return total, rows


def stage_bytes(path, payload):
    if not path.parent.is_dir() or path.is_dir() or not os.access(path.parent, os.W_OK):
        raise Refusal("保存失败: SAMPLE_IO")
    descriptor, temporary = tempfile.mkstemp(prefix=".sample-", dir=path.parent)
    stage = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        stage.unlink(missing_ok=True)
        raise Refusal("保存失败: SAMPLE_IO")
    return stage


def atomic_write(path, payload):
    stage = stage_bytes(path, payload)
    try:
        os.replace(stage, path)
    except OSError:
        stage.unlink(missing_ok=True)
        raise Refusal("保存失败: SAMPLE_IO")


def read_ledger(path):
    if not path.exists():
        return b"", {}
    try:
        raw = path.read_bytes()
    except OSError:
        raise Refusal("复核簿拒收: BAD_SAMPLE_LEDGER")
    if raw and not raw.endswith(b"\n"):
        raise Refusal("复核簿拒收: BAD_SAMPLE_LEDGER")
    entries = {}
    for line in raw.splitlines():
        try:
            item = json.loads(line, object_pairs_hook=pairs_hook)
        except (ValueError, UnicodeError, Refusal):
            raise Refusal("复核簿拒收: BAD_SAMPLE_LEDGER")
        if not isinstance(item, dict) or list(item) != LEDGER_ORDER:
            raise Refusal("复核簿拒收: BAD_SAMPLE_LEDGER")
        if (
            not visible_identifier(item["survey_id"])
            or not isinstance(item["input_sha256"], str)
            or not isinstance(item["result_sha256"], str)
            or not finite(item["total_residual"])
        ):
            raise Refusal("复核簿拒收: BAD_SAMPLE_LEDGER")
        if len(item["input_sha256"]) != 64 or len(item["result_sha256"]) != 64:
            raise Refusal("复核簿拒收: BAD_SAMPLE_LEDGER")
        previous = entries.get(item["result_sha256"])
        if previous is not None and previous != item:
            raise Refusal("复核簿拒收: BAD_SAMPLE_LEDGER")
        entries[item["result_sha256"]] = item
    return raw, entries


def publish(source_bytes, result_bytes, target, ledger):
    if ledger is not None and ledger.resolve() == target.resolve():
        raise Refusal("复核簿拒收: BAD_SAMPLE_LEDGER")
    old_result = target.read_bytes() if target.is_file() else None
    old_ledger = None
    entry = None
    if ledger is not None:
        old_ledger, entries = read_ledger(ledger)
        result_sha = hashlib.sha256(result_bytes).hexdigest()
        entry = {
            "survey_id": json.loads(result_bytes)["survey_id"],
            "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "result_sha256": result_sha,
            "total_residual": json.loads(result_bytes)["total_residual"],
        }
        if result_sha in entries:
            if entries[result_sha] != entry:
                raise Refusal("复核簿拒收: BAD_SAMPLE_LEDGER")
            entry = None
    result_stage = None
    ledger_stage = None
    try:
        result_stage = stage_bytes(target, result_bytes)
        if ledger is not None and entry is not None:
            ledger_bytes = old_ledger + (
                json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            ledger_stage = stage_bytes(ledger, ledger_bytes)
        os.replace(result_stage, target)
        result_stage = None
        if ledger_stage is not None:
            os.replace(ledger_stage, ledger)
            ledger_stage = None
    except OSError:
        if result_stage is not None:
            result_stage.unlink(missing_ok=True)
        if ledger_stage is not None:
            ledger_stage.unlink(missing_ok=True)
        if old_result is None:
            target.unlink(missing_ok=True)
        else:
            atomic_write(target, old_result)
        if ledger is not None and old_ledger is not None:
            atomic_write(ledger, old_ledger)
        raise Refusal("保存失败: SAMPLE_IO")


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("统计拒收: BAD_SAMPLE\n")
        return 1
    source, target = Path(argv[1]), Path(argv[2])
    ledger = Path(argv[3]) if len(argv) == 4 else None
    try:
        source_bytes = source.read_bytes()
        payload = json.loads(source_bytes.decode("utf-8"), object_pairs_hook=pairs_hook)
        case_id, ceiling, profiles, readings = validate(payload)
        total, rows = assign(profiles, readings)
        accepted = total <= ceiling + 1.0e-12
        result = {
            "survey_id": case_id,
            "status": "samples_recovered" if accepted else "quarantined",
            "sample_assignments": rows,
            "total_residual": total,
            "accepted": accepted,
            "diagnostic": "fingerprints_within_ceiling" if accepted else "fingerprints_outside_ceiling",
        }
        result_bytes = (json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        publish(source_bytes, result_bytes, target, ledger)
        return 0
    except Refusal as exc:
        sys.stderr.write(exc.message + "\n")
        return 1
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        sys.stderr.write("统计拒收: BAD_SAMPLE\n")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
