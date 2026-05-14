from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_MODELS = [
    Path("models/yolov8s_pothole.onnx"),
    Path("models/depth_anything_v2_vits.onnx"),
]


def main() -> None:
    checks = {
        "python": check_python(),
        "venv_launcher": check_venv_launcher(),
        "opencv": check_import("cv2", version_attr="__version__"),
        "numpy": check_import("numpy", version_attr="__version__"),
        "ultralytics": check_import("ultralytics", version_attr="__version__"),
        "onnxruntime": check_onnxruntime(),
        "models": check_models(),
        "requirements": check_requirements(),
    }
    print(json.dumps(checks, indent=2))

    failed = [
        name
        for name, result in checks.items()
        if isinstance(result, dict) and result.get("ok") is False
    ]
    if failed:
        raise SystemExit(f"Runtime checks failed: {', '.join(failed)}")


def check_python() -> Dict[str, Any]:
    return {
        "ok": True,
        "executable": sys.executable,
        "version": sys.version.replace("\n", " "),
    }


def check_venv_launcher() -> Dict[str, Any]:
    launcher = Path("myenv/Scripts/python.exe")
    if not launcher.exists():
        return {
            "ok": True,
            "path": str(launcher),
            "warning": "venv python launcher not found; create a fresh venv before running heavy demos",
        }

    completed = subprocess.run(
        [str(launcher), "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return {
            "ok": True,
            "path": str(launcher),
            "returncode": completed.returncode,
            "warning": (
                "venv python launcher exists but is not runnable. This usually means the "
                "venv was copied from another machine or its base Python moved; recreate "
                "myenv and reinstall requirements.txt."
            ),
            "output": output,
        }
    return {
        "ok": True,
        "path": str(launcher),
        "returncode": completed.returncode,
        "output": output,
    }


def check_models() -> Dict[str, Any]:
    missing = [str(path) for path in REQUIRED_MODELS if not path.exists()]
    return {
        "ok": not missing,
        "required": [str(path) for path in REQUIRED_MODELS],
        "missing": missing,
    }


def check_import(module_name: str, version_attr: Optional[str] = None) -> Dict[str, Any]:
    try:
        module = __import__(module_name)
    except ImportError as exc:
        return {
            "ok": False,
            "error": str(exc),
        }

    result: Dict[str, Any] = {"ok": True}
    if version_attr:
        result["version"] = getattr(module, version_attr, None)
    return result


def check_onnxruntime() -> Dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return {
            "ok": False,
            "error": str(exc),
        }

    providers = list(ort.get_available_providers())
    gpu_providers = [
        provider
        for provider in providers
        if provider != "CPUExecutionProvider"
    ]
    return {
        "ok": "CPUExecutionProvider" in providers,
        "providers": providers,
        "cpu_provider_available": "CPUExecutionProvider" in providers,
        "gpu_providers_visible": gpu_providers,
    }


def check_requirements() -> Dict[str, Any]:
    path = Path("requirements.txt")
    if not path.exists():
        return {"ok": False, "error": "requirements.txt not found"}

    text = _read_text(path)
    lines: List[str] = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    gpu_packages = [
        line
        for line in lines
        if line.lower().startswith("onnxruntime-gpu")
    ]
    return {
        "ok": not gpu_packages,
        "onnxruntime_cpu": any(line.lower().startswith("onnxruntime==") for line in lines),
        "gpu_packages": gpu_packages,
    }


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return path.read_text()


if __name__ == "__main__":
    main()
