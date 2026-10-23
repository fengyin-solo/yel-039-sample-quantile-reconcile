"""Acceptance tests for hidden-channel fingerprint reconciliation."""

import copy
import json
import os
import subprocess
import sys

import pytest


ENTRY = os.environ.get("SAMPLE_QUANTILE_ENTRY", "/app/sample_quantile_reconcile.py")
PROFILES = [{'sample_id': 'SAMPLE-A', 'fingerprint': [12.4, 3.15, 0.034]},
 {'sample_id': 'SAMPLE-B', 'fingerprint': [18.7, 4.28, 0.061]},
 {'sample_id': 'SAMPLE-C', 'fingerprint': [26.1, 2.74, 0.022]},
 {'sample_id': 'SAMPLE-D', 'fingerprint': [33.5, 6.09, 0.078]},
 {'sample_id': 'SAMPLE-E', 'fingerprint': [13.223, 3.37175, 0.05321]},
 {'sample_id': 'SAMPLE-F', 'fingerprint': [21.165, 4.8704, 0.10293]},
 {'sample_id': 'SAMPLE-G', 'fingerprint': [31.2405, 3.3253, 0.07729]},
 {'sample_id': 'SAMPLE-H', 'fingerprint': [42.278, 7.7414, 0.16628]},
 {'sample_id': 'SAMPLE-I', 'fingerprint': [16.515, 4.25875, 0.13005]},
 {'sample_id': 'SAMPLE-J', 'fingerprint': [26.095, 6.0512, 0.18679]}]
READINGS = [{'slot': 'CH-3',
  'sample_sum': 14.209442060085838,
  'sample_sq': 187.56750000000002,
  'sample_n': 1.0,
  'trim_loss': 0.07296137339055794,
  'scaling': 1.0},
 {'slot': 'CH-10',
  'sample_sum': 23.896810933940774,
  'sample_sq': 470.2795271070615,
  'sample_n': 1.0,
  'trim_loss': 0.13895216400911162,
  'scaling': 1.0},
 {'slot': 'CH-6',
  'sample_sum': 28.502510460251045,
  'sample_sq': 752.1141991631799,
  'sample_n': 1.0,
  'trim_loss': 0.04602510460251046,
  'scaling': 1.0},
 {'slot': 'CH-13',
  'sample_sum': 45.883886255924175,
  'sample_sq': 1587.9085824644549,
  'sample_n': 1.0,
  'trim_loss': 0.1848341232227488,
  'scaling': 1.0},
 {'slot': 'CH-9',
  'sample_sum': 16.37255943508136,
  'sample_sq': 230.5709385063355,
  'sample_n': 1.0,
  'trim_loss': 0.11909398151256742,
  'scaling': 1.0},
 {'slot': 'CH-5',
  'sample_sum': 32.13794406527816,
  'sample_sq': 716.218372825947,
  'sample_n': 1.0,
  'trim_loss': 0.2592238144407787,
  'scaling': 1.0},
 {'slot': 'CH-12',
  'sample_sum': 42.664777850062684,
  'sample_sq': 1347.97025215793,
  'sample_n': 1.0,
  'trim_loss': 0.18284403018617962,
  'scaling': 1.0},
 {'slot': 'CH-8',
  'sample_sum': 84.4090430300851,
  'sample_sq': 3688.295756914745,
  'sample_n': 1.0,
  'trim_loss': 0.49826201606136883,
  'scaling': 1.0},
 {'slot': 'CH-4',
  'sample_sum': 28.12616772536829,
  'sample_sq': 495.3921214845334,
  'sample_n': 1.0,
  'trim_loss': 0.35153399107987565,
  'scaling': 1.0},
 {'slot': 'CH-11',
  'sample_sum': 57.219708981194735,
  'sample_sq': 1573.440136121221,
  'sample_n': 1.0,
  'trim_loss': 0.5963730404520929,
  'scaling': 1.0}]
EXPECTED_ROWS = [{'slot': 'CH-10', 'sample_id': 'SAMPLE-B', 'residual': 0.0},
 {'slot': 'CH-11', 'sample_id': 'SAMPLE-J', 'residual': 0.0},
 {'slot': 'CH-12', 'sample_id': 'SAMPLE-G', 'residual': 0.0},
 {'slot': 'CH-13', 'sample_id': 'SAMPLE-D', 'residual': 0.0},
 {'slot': 'CH-3', 'sample_id': 'SAMPLE-A', 'residual': 0.0},
 {'slot': 'CH-4', 'sample_id': 'SAMPLE-I', 'residual': 0.0},
 {'slot': 'CH-5', 'sample_id': 'SAMPLE-F', 'residual': 0.0},
 {'slot': 'CH-6', 'sample_id': 'SAMPLE-C', 'residual': 0.0},
 {'slot': 'CH-8', 'sample_id': 'SAMPLE-H', 'residual': 0.0},
 {'slot': 'CH-9', 'sample_id': 'SAMPLE-E', 'residual': 0.0}]
OUTPUT_KEYS = ['survey_id', 'status', 'sample_assignments', 'total_residual', 'accepted', 'diagnostic']
LEDGER_KEYS = ['survey_id', 'input_sha256', 'result_sha256', 'total_residual']


def base_payload():
    return {
        "survey_id": "SAMPLE-STD-7",
        "limits": {"max_slots": 12, "residual_ceiling": 0.02},
        "profiles": copy.deepcopy(PROFILES),
        "readings": copy.deepcopy(READINGS),
    }


def invoke(tmp_path, payload=None, ledger=None, raw=None, output_name="result.json"):
    source = tmp_path / "readings.json"
    target = tmp_path / output_name
    if raw is None:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    source.write_bytes(raw.encode("utf-8"))
    command = [sys.executable, ENTRY, str(source), str(target)]
    if ledger is not None:
        command.append(str(ledger))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return completed, source, target


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_writes_aligned_output_quietly(tmp_path):
    """题面：「盲样报告落盘成功时，标准输出和标准错误均为空。」"""
    result, _, output = invoke(tmp_path, base_payload())
    assert result.returncode == 0
    assert result.stdout == "" and result.stderr == ""
    assert output.is_file()


def test_global_assignment_is_minimum_residual(tmp_path):
    """题面：「样本指纹和档案按全组最小总差形成不重复配对。」"""
    result, _, output = invoke(tmp_path, base_payload())
    document = read(output)
    assert result.returncode == 0
    assert [(row["slot"], row["sample_id"]) for row in document["sample_assignments"]] == [(row["slot"], row["sample_id"]) for row in EXPECTED_ROWS]
    assert [row["residual"] for row in document["sample_assignments"]] == pytest.approx([row["residual"] for row in EXPECTED_ROWS], abs=1e-12)


def test_assignment_rows_have_exact_fields(tmp_path):
    """题面：「sample_assignments 对象的成员是 slot、sample_id、residual。」"""
    result, _, output = invoke(tmp_path, base_payload())
    document = read(output)
    assert result.returncode == 0
    assert all(set(row) == {"slot", "sample_id", "residual"} for row in document["sample_assignments"])


def test_output_has_exact_top_level_keys(tmp_path):
    """题面：「最外层对象成员为 survey_id、status、sample_assignments、total_residual、accepted、diagnostic。」"""
    result, _, output = invoke(tmp_path, base_payload())
    document = read(output)
    assert result.returncode == 0
    assert list(document) == OUTPUT_KEYS


def test_output_is_compact_utf8_with_single_lf(tmp_path):
    """题面：「报告写成紧凑 UTF-8 JSON，并以单个换行结束。」"""
    result, _, output = invoke(tmp_path, base_payload())
    raw = output.read_bytes()
    assert result.returncode == 0
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b": " not in raw and b", " not in raw


def test_total_residual_matches_assignment(tmp_path):
    """题面：「全组 total_residual 等于每个 residual 的算术和。」"""
    result, _, output = invoke(tmp_path, base_payload())
    document = read(output)
    total = sum(row["residual"] for row in document["sample_assignments"])
    assert result.returncode == 0
    assert document["total_residual"] == pytest.approx(total, abs=1e-12)
    assert document["total_residual"] == pytest.approx(0.0, abs=1e-10)


def test_ceiling_accepts_inside_case(tmp_path):
    """题面：「总差小于等于 residual_ceiling 时，accepted 为 true。」"""
    payload = base_payload()
    payload["limits"]["residual_ceiling"] = 1.0
    payload["profiles"][0]["fingerprint"] = [value * 1.04 for value in payload["profiles"][0]["fingerprint"]]
    result, _, output = invoke(tmp_path, payload)
    document = read(output)
    assert result.returncode == 0 and document["accepted"] is True
    assert document["status"] == "samples_recovered"


def test_ceiling_quarantines_first_outside_value(tmp_path):
    """题面：「总差大于 residual_ceiling 时，status 写入 quarantined，报告照常保留。」"""
    payload = base_payload()
    payload["limits"]["residual_ceiling"] = 0.0001
    payload["profiles"][0]["fingerprint"] = [value * 1.04 for value in payload["profiles"][0]["fingerprint"]]
    result, _, output = invoke(tmp_path, payload)
    document = read(output)
    assert result.returncode == 0 and document["accepted"] is False
    assert document["status"] == "quarantined"


def test_missing_top_level_key_is_refused(tmp_path):
    """题面：「根对象缺输入项时退出 1，并输出 统计拒收: BAD_SAMPLE。」"""
    payload = base_payload()
    payload.pop("limits")
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "统计拒收: BAD_SAMPLE\n" and not output.exists()


def test_extra_top_level_key_is_refused(tmp_path):
    """题面：「根对象带额外输入项时退出 1，并输出 统计拒收: BAD_SAMPLE。」"""
    payload = base_payload()
    payload["memo"] = "shift"
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "统计拒收: BAD_SAMPLE\n" and not output.exists()


def test_duplicate_top_level_key_is_refused(tmp_path):
    """题面：「JSON 出现重名键时，按损坏输入输出 统计拒收: BAD_SAMPLE。」"""
    payload = base_payload()
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw = raw.replace('"survey_id":"', '"survey_id":"duplicate", "survey_id":"', 1) + "\n"
    result, _, output = invoke(tmp_path, raw=raw)
    assert result.returncode == 1 and result.stderr == "统计拒收: BAD_SAMPLE\n" and not output.exists()


def test_boolean_numeric_is_refused(tmp_path):
    """题面：「数值位置出现布尔值时，输出 统计拒收: BAD_SAMPLE。」"""
    payload = base_payload()
    payload["readings"][0]["sample_sum"] = True
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "统计拒收: BAD_SAMPLE\n" and not output.exists()


def test_profile_fingerprint_length_is_refused(tmp_path):
    """题面：「三元 fingerprint 缺分量、含零或非有限数时，输出 统计拒收: BAD_SAMPLE。」"""
    payload = base_payload()
    payload["profiles"][0]["fingerprint"].pop()
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "统计拒收: BAD_SAMPLE\n" and not output.exists()


def test_duplicate_slot_is_refused(tmp_path):
    """题面：「两个样本占用同一 slot 时，输出 统计拒收: BAD_SAMPLE。」"""
    payload = base_payload()
    payload["readings"][1]["slot"] = payload["readings"][0]["slot"]
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "统计拒收: BAD_SAMPLE\n" and not output.exists()
    minimum = base_payload()
    minimum["survey_id"] = "AB"
    minimum_result, _, minimum_output = invoke(tmp_path, minimum, output_name="minimum.json")
    assert minimum_result.returncode == 0 and read(minimum_output)["survey_id"] == "AB"
    spaced = base_payload()
    spaced["survey_id"] = "A B"
    spaced_result, _, spaced_output = invoke(tmp_path, spaced, output_name="spaced.json")
    assert spaced_result.returncode == 0 and read(spaced_output)["survey_id"] == "A B"
    control = base_payload()
    control["survey_id"] = "A\tB"
    control_result, _, control_output = invoke(tmp_path, control, output_name="control.json")
    assert control_result.returncode == 1 and control_result.stderr == "统计拒收: BAD_SAMPLE\n" and not control_output.exists()


def test_ledger_has_exact_shape_and_is_idempotent(tmp_path):
    """题面：「同一输入和同一结果再次提交时，留痕簿保持原字节。」"""
    ledger = tmp_path / "trail.jsonl"
    result, _, output = invoke(tmp_path, payload=base_payload(), ledger=ledger)
    first = ledger.read_bytes()
    second, _, _ = invoke(tmp_path, payload=base_payload(), ledger=ledger)
    item = json.loads(first)
    assert result.returncode == 0 and second.returncode == 0
    assert list(item) == LEDGER_KEYS and first.endswith(b"\n")
    assert ledger.read_bytes() == first and second.stdout == "" and second.stderr == ""


def test_different_result_appends_one_ledger_line(tmp_path):
    """题面：「新的有效结果出现后，留痕簿按时间先后续写一行。」"""
    ledger = tmp_path / "trail.jsonl"
    first, _, _ = invoke(tmp_path, payload=base_payload(), ledger=ledger)
    changed = base_payload()
    changed["survey_id"] = "SAMPLE-STD-8"
    second, _, _ = invoke(tmp_path, payload=changed, ledger=ledger)
    assert first.returncode == 0 and second.returncode == 0
    assert len(ledger.read_bytes().splitlines()) == 2
    assert ledger.read_bytes().endswith(b"\n")


def test_ledger_path_equal_to_output_is_refused(tmp_path):
    """题面：「留痕位置与结果位置完全相同时，输出 复核簿拒收: BAD_SAMPLE_LEDGER。」"""
    payload = base_payload()
    source = tmp_path / "readings.json"
    target = tmp_path / "same.json"
    source.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    result = subprocess.run([sys.executable, ENTRY, str(source), str(target), str(target)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and result.stderr == "复核簿拒收: BAD_SAMPLE_LEDGER\n" and not target.exists()


def test_io_failure_preserves_existing_target(tmp_path):
    """题面：「保存结果失败时，输出 保存失败: SAMPLE_IO，旧文件不被替换。」"""
    target = tmp_path / "occupied"
    target.mkdir()
    source = tmp_path / "readings.json"
    source.write_text(json.dumps(base_payload(), separators=(",", ":")), encoding="utf-8")
    result = subprocess.run([sys.executable, ENTRY, str(source), str(target)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and result.stderr == "保存失败: SAMPLE_IO\n"
    assert target.exists() and target.is_dir() and not any(target.iterdir()) and "traceback" not in result.stderr.lower()


def test_wrong_argument_count_is_refused(tmp_path):
    """题面：「参数数量不符合规定时，输出 统计拒收: BAD_SAMPLE 并返回 1。」"""
    target = tmp_path / "result.json"
    result = subprocess.run([sys.executable, ENTRY], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and result.stderr == "统计拒收: BAD_SAMPLE\n" and not target.exists()
