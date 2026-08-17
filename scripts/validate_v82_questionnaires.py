#!/usr/bin/env python3
"""Statically validate v82 standalone questionnaire forms and embedded audio."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v82_questionnaires import FORBIDDEN_PRIVATE_TOKENS, FORMS


CONFIG_PATTERN = re.compile(r"window\.V82_CONFIG=(.*?);window\.V82_AUDIO=", re.DOTALL)
AUDIO_PATTERN = re.compile(r"window\.V82_AUDIO=(.*?);</script>", re.DOTALL)
SCRIPT_PATTERN = re.compile(r"<script>(.*?)</script>", re.DOTALL | re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_payloads(html: str) -> tuple[dict, dict]:
    config_match = CONFIG_PATTERN.search(html)
    audio_match = AUDIO_PATTERN.search(html)
    if not config_match or not audio_match:
        raise ValueError("questionnaire payload markers are missing")
    return json.loads(config_match.group(1)), json.loads(audio_match.group(1))


def node_syntax_check(html: str, node: str) -> None:
    scripts = SCRIPT_PATTERN.findall(html)
    if len(scripts) != 2:
        raise ValueError(f"expected exactly two inline scripts, found {len(scripts)}")
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as handle:
        handle.write("\n".join(scripts))
        handle.flush()
        completed = subprocess.run(
            [node, "--check", handle.name], text=True, capture_output=True
        )
    if completed.returncode:
        raise ValueError(f"JavaScript syntax check failed: {completed.stderr}")


def validate(args: argparse.Namespace) -> dict:
    audio_rows = read_csv(args.audio_manifest)
    expected_audio = {row["stimulus_id"]: row["audio_sha256"] for row in audio_rows}
    by_form = {}
    for form in FORMS:
        path = args.questionnaire_dir / f"MoonMuse-v82-{form}-{args.mode}.html"
        html = path.read_text(encoding="utf-8")
        config, audio = parse_payloads(html)
        public_rows = read_csv(args.study_dir / "forms_public" / f"{form}.csv")
        expected_ids = {row["stimulus_id"] for row in public_rows}
        if config.get("form") != form or set(config.get("stimulus_ids", [])) != expected_ids:
            raise ValueError(f"form/config stimulus mismatch: {form}")
        if set(audio) != expected_ids:
            raise ValueError(f"form/audio stimulus mismatch: {form}")
        if config.get("formalStudyData") != (args.mode == "launch"):
            raise ValueError(f"formalStudyData mismatch: {form}")
        hashes = {}
        for stimulus_id, data_url in audio.items():
            prefix = "data:audio/mpeg;base64,"
            if not data_url.startswith(prefix):
                raise ValueError(f"invalid audio URL: {form}/{stimulus_id}")
            payload = base64.b64decode(data_url[len(prefix):], validate=True)
            digest = sha256_bytes(payload)
            if digest != expected_audio[stimulus_id]:
                raise ValueError(f"embedded audio hash mismatch: {form}/{stimulus_id}")
            hashes[stimulus_id] = digest
        lowered = html.lower()
        leaked = [token for token in FORBIDDEN_PRIVATE_TOKENS if token.lower() in lowered]
        if leaked:
            raise ValueError(f"private mapping leak in {form}: {leaked}")
        required_markers = (
            "localStorage",
            "played_fraction",
            "played_fraction<.8",
            "moonmuse-v82-listener-export-v1",
            "不会自动上传",
        )
        missing = [marker for marker in required_markers if marker not in html]
        if missing:
            raise ValueError(f"functional markers missing in {form}: {missing}")
        node_syntax_check(html, args.node)
        by_form[form] = {
            "html": str(path.resolve()),
            "html_sha256": sha256_bytes(path.read_bytes()),
            "bytes": path.stat().st_size,
            "stimuli": len(audio),
            "embedded_audio_hashes_match": True,
            "javascript_syntax_pass": True,
            "private_mapping_leak": False,
        }
    report = {
        "schema": "moonmuse-v82-questionnaire-static-validation-v1",
        "status": f"{args.mode}_static_validation_passed",
        "formalStudyData": args.mode == "launch",
        "forms": by_form,
        "cross_form_stimulus_overlap": len(
            set(json.loads(CONFIG_PATTERN.search((args.questionnaire_dir / f"MoonMuse-v82-F1-{args.mode}.html").read_text(encoding="utf-8")).group(1))["stimulus_ids"])
            & set(json.loads(CONFIG_PATTERN.search((args.questionnaire_dir / f"MoonMuse-v82-F2-{args.mode}.html").read_text(encoding="utf-8")).group(1))["stimulus_ids"])
        ),
        "remaining_gate": (
            "Browser interaction spot-check and ethics approval/exemption remain required."
            if args.mode == "prelaunch"
            else "Issue forms under the frozen private participant-code allocation."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questionnaire-dir", required=True, type=Path)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--audio-manifest", required=True, type=Path)
    parser.add_argument("--mode", choices=("prelaunch", "launch"), default="prelaunch")
    parser.add_argument("--node", default="node")
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    report = validate(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
