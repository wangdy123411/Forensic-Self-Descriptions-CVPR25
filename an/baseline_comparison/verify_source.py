"""Offline source evidence after the root agent's successful git ls-remote.

Exports canonical Git blobs without editing the working package or old audits.
The observed official refs are supplied from that read-only network operation;
this script independently validates the corresponding local Git objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
MAIN = "50f2eae06efdac2e5a33f407ca9a27a2295133ac"
TAG = "5b317a00251988b5ec5a47317f4d82e5bdfd009d"
REPO = "https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25"


def git(*args) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def preserve(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite a different source snapshot: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main():
    destination = OUTPUT / "source_verification.json"
    if destination.exists():
        raise FileExistsError(
            "Source evidence already exists; retain it and use a new audit directory for another observation"
        )
    head = git("rev-parse", "HEAD").decode().strip()
    if head != MAIN:
        raise RuntimeError(f"Checkout changed from the verified official commit: {head}")
    status = git("status", "--porcelain", "--", "fsd").decode()
    if status:
        raise RuntimeError(f"The original package has local changes: {status}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", TAG, MAIN], cwd=ROOT, capture_output=True
    )
    if ancestor.returncode not in (0, 1):
        raise RuntimeError(ancestor.stderr.decode())
    paths = git("ls-tree", "-r", "--name-only", MAIN, "--", "fsd").decode().splitlines()
    files = []
    for path in paths:
        current = git("cat-file", "blob", f"{MAIN}:{path}")
        tagged = git("cat-file", "blob", f"{TAG}:{path}")
        head_blob = git("rev-parse", f"{MAIN}:{path}").decode().strip()
        tag_blob = git("rev-parse", f"{TAG}:{path}").decode().strip()
        normalized_worktree_blob = git("hash-object", f"--path={path}", path).decode().strip()
        raw_worktree = (ROOT / path).read_bytes()
        independently_hashed_blob = hashlib.sha1(
            b"blob " + str(len(current)).encode() + b"\0" + current
        ).hexdigest()
        if not head_blob == normalized_worktree_blob == independently_hashed_blob:
            raise AssertionError(f"Working source differs from verified Git blob: {path}")
        main_snapshot = Path("source_snapshots") / ("main_" + MAIN[:7]) / path
        tag_snapshot = Path("source_snapshots") / ("tag_v1.2.0_" + TAG[:7]) / path
        preserve(OUTPUT / main_snapshot, current)
        preserve(OUTPUT / tag_snapshot, tagged)
        files.append(
            {
                "path": path,
                "official_main_git_blob_sha1": head_blob,
                "local_head_git_blob_sha1": head_blob,
                "working_tree_filtered_git_blob_sha1": normalized_worktree_blob,
                "tag_git_blob_sha1": tag_blob,
                "main_equals_tag_blob": head_blob == tag_blob,
                "canonical_main_sha256": digest(current),
                "canonical_tag_sha256": digest(tagged),
                "raw_working_tree_sha256": digest(raw_worktree),
                "raw_working_tree_equals_canonical_blob": raw_worktree == current,
                "working_tree_equals_main_after_git_filters": True,
                "canonical_main_snapshot": main_snapshot.as_posix(),
                "canonical_tag_snapshot": tag_snapshot.as_posix(),
            }
        )
    patch = git("diff", TAG, MAIN, "--", "fsd")
    preserve(OUTPUT / "tag_to_main_fsd.patch", patch)
    config_path = ROOT / "an" / "weights" / "v1.2.0" / "config.json"
    value = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_repository": REPO,
        "local_remote": git("remote", "get-url", "origin").decode().strip(),
        "official_ref_observation": {
            "method": "Read-only git ls-remote successfully executed by root agent in this session",
            "reported_refs": {"HEAD": MAIN, "refs/heads/main": MAIN, "refs/tags/v1.2.0": TAG},
            "tag_type": "lightweight; no peeled tag reference was returned",
            "evidence_scope": "Root-agent tool observation; this script does not repeat network access",
            "subagent_network_note": "Sandbox request failed with Bad access; escalated retry remained pending for 1274.7 seconds and was interrupted. Root separately completed the remote lookup.",
        },
        "local_head": head,
        "local_head_equals_observed_official_main": head == MAIN,
        "tag_commit": TAG,
        "tag_is_ancestor_of_main": ancestor.returncode == 0,
        "main_is_exact_tag_commit": MAIN == TAG,
        "fsd_working_tree_clean": not status,
        "fsd_file_count": len(files),
        "core_autocrlf": git("config", "--get", "core.autocrlf").decode().strip(),
        "source_identity_conclusion": "All 11 fsd Git blobs match the observed official main commit. Canonical snapshots preserve exact Git bytes; Windows working files use Git newline conversion and are not claimed raw-byte identical.",
        "tag_differing_files": [item["path"] for item in files if not item["main_equals_tag_blob"]],
        "weights": {
            "release": "v1.2.0",
            "explicit_directory": "an/weights/v1.2.0",
            "config_sha256": digest(config_path.read_bytes()),
            "config": json.loads(config_path.read_text()),
        },
        "primary_source_links": [
            REPO + "/commit/" + MAIN,
            REPO + "/tree/" + MAIN + "/fsd",
            REPO + "/tree/v1.2.0/fsd",
            REPO + "/compare/" + TAG + "..." + MAIN,
            REPO + "/releases/tag/v1.2.0",
        ],
        "files": files,
    }
    destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value[key]
                for key in (
                    "local_head",
                    "local_head_equals_observed_official_main",
                    "tag_commit",
                    "tag_is_ancestor_of_main",
                    "main_is_exact_tag_commit",
                    "fsd_working_tree_clean",
                    "fsd_file_count",
                    "tag_differing_files",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
