from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

CONTROL = Path(__file__).resolve().parents[1] / ".cursor/skills/verify-embedforge/bin/control-embed"

REQUIRED_FOR_HELP = (
    "bash",
    "dirname",
    "uname",
    "id",
    "awk",
    "cut",
    "true",
    "false",
    "cat",
    "printf",
)


def _link_commands(bindir: Path, names: tuple[str, ...]) -> None:
    bindir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = _which(name)
        if src is None:
            continue
        dest = bindir / name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src)


def _which(name: str) -> Path | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _run_control(bindir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = str(bindir)
    bash = bindir / "bash"
    return subprocess.run(
        [str(bash), str(CONTROL), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_help_succeeds_when_getent_is_absent(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    _link_commands(bindir, REQUIRED_FOR_HELP)
    assert not (bindir / "getent").exists()
    result = _run_control(bindir, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage: control-embed" in result.stdout


def test_help_does_not_call_getent_when_uname_is_darwin(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    _link_commands(bindir, REQUIRED_FOR_HELP)
    marker = tmp_path / "getent-called"
    getent = bindir / "getent"
    getent.write_text(f"#!/bin/sh\nprintf x > '{marker}'\nexit 127\n")
    getent.chmod(getent.stat().st_mode | stat.S_IXUSR)
    uname = bindir / "uname"
    if uname.exists() or uname.is_symlink():
        uname.unlink()
    uname.write_text("#!/bin/sh\nprintf '%s\\n' Darwin\n")
    uname.chmod(uname.stat().st_mode | stat.S_IXUSR)

    result = _run_control(bindir, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage: control-embed" in result.stdout
    assert not marker.exists()


def test_smoke_is_documented_as_a_command(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    _link_commands(bindir, REQUIRED_FOR_HELP)
    result = _run_control(bindir, "--help")
    assert result.returncode == 0, result.stderr
    assert "smoke" in result.stdout


def test_canonical_paths_collapse_slashes(tmp_path: Path) -> None:
    target = tmp_path / "T" / "embedforge-verify-slash-tmp" / "cache" / "embedforge"
    target.mkdir(parents=True)
    doubled = f"{tmp_path}/T//embedforge-verify-slash-tmp/cache/embedforge"
    left = os.path.realpath(os.path.normpath(doubled))
    right = os.path.realpath(os.path.normpath(str(target)))
    assert left == right
    assert "//" not in left


def _control(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(CONTROL), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_doctor_accepts_tmpdir_with_trailing_slash(tmp_path: Path) -> None:
    tmpdir = tmp_path / "T"
    tmpdir.mkdir()
    env = os.environ.copy()
    env["TMPDIR"] = f"{tmpdir}/"
    env["CONTROL_EMBED_EVIDENCE"] = str(tmp_path / "evidence")
    env["CONTROL_EMBED_STATE"] = str(tmp_path / "state.env")
    env.pop("CONTROL_EMBED_SCRATCH", None)
    launch = subprocess.run(
        [str(CONTROL), "launch", "--run-id", "slash-tmp"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    doctor = subprocess.run(
        [str(CONTROL), "doctor"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    cleanup = subprocess.run(
        [str(CONTROL), "cleanup"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert launch.returncode == 0, launch.stdout + launch.stderr
    assert doctor.returncode == 0, doctor.stdout + doctor.stderr
    assert "doctor: OK" in doctor.stdout
    assert cleanup.returncode == 0, cleanup.stdout + cleanup.stderr


def test_transcript_reuses_shell_quoted_cmd(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["CONTROL_EMBED_STATE_DIR"] = str(tmp_path / "state")
    env["CONTROL_EMBED_RUN_ID"] = "quote-run"
    env["CONTROL_EMBED_EVIDENCE"] = str(tmp_path / "ev")
    env["CONTROL_EMBED_SCRATCH"] = str(tmp_path / "scratch")
    env.pop("CONTROL_EMBED_STATE", None)
    assert _control(env, "launch", "--run-id", "quote-run").returncode == 0
    result = _control(env, "cli", "--", "set", "output_column", "my embedding")
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = tmp_path / "ev"
    cmd = next(evidence.glob("*.cmd.txt")).read_text()
    transcript = next(evidence.glob("*.transcript.txt")).read_text()
    first = transcript.splitlines()[0]
    assert first == f"$ {cmd.rstrip(chr(10))}"
    assert "output_column my embedding" not in first
    _control(env, "cleanup")


def test_two_runs_keep_config_evidence_and_cleanup_separate(tmp_path: Path) -> None:
    base = os.environ.copy()
    base["CONTROL_EMBED_STATE_DIR"] = str(tmp_path / "state")
    base.pop("CONTROL_EMBED_STATE", None)
    env_a = base.copy()
    env_a["CONTROL_EMBED_RUN_ID"] = "run-a"
    env_a["CONTROL_EMBED_EVIDENCE"] = str(tmp_path / "ev-a")
    env_a["CONTROL_EMBED_SCRATCH"] = str(tmp_path / "scratch-a")
    env_b = base.copy()
    env_b["CONTROL_EMBED_RUN_ID"] = "run-b"
    env_b["CONTROL_EMBED_EVIDENCE"] = str(tmp_path / "ev-b")
    env_b["CONTROL_EMBED_SCRATCH"] = str(tmp_path / "scratch-b")

    assert _control(env_a, "launch", "--run-id", "run-a").returncode == 0
    set_a = _control(env_a, "cli", "--", "set", "output_column", "alpha-from-a")
    assert set_a.returncode == 0, set_a.stdout + set_a.stderr
    assert _control(env_b, "launch", "--run-id", "run-b").returncode == 0

    get_a = _control(env_a, "cli", "--", "get", "output_column")
    get_b = _control(env_b, "cli", "--", "get", "output_column")
    assert get_a.returncode == 0, get_a.stdout + get_a.stderr
    assert get_b.returncode == 0, get_b.stdout + get_b.stderr

    a_out = "\n".join(path.read_text() for path in (tmp_path / "ev-a").glob("*.stdout.txt"))
    b_out = "\n".join(path.read_text() for path in (tmp_path / "ev-b").glob("*.stdout.txt"))
    assert "alpha-from-a" in a_out
    assert "alpha-from-a" not in b_out
    assert any(path.name.startswith("01-") for path in (tmp_path / "ev-b").glob("*.stdout.txt"))

    assert (tmp_path / "scratch-a").is_dir()
    assert (tmp_path / "scratch-b").is_dir()
    assert _control(env_a, "cleanup").returncode == 0
    assert not (tmp_path / "scratch-a").exists()
    assert (tmp_path / "scratch-b").is_dir()
    get_b_again = _control(env_b, "cli", "--", "get", "output_column")
    assert get_b_again.returncode == 0, get_b_again.stdout + get_b_again.stderr
    assert _control(env_b, "cleanup").returncode == 0
    assert not (tmp_path / "scratch-b").exists()
