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
