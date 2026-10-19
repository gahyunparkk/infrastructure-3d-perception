# Architecture

Perception stage: camera/video → KIT 2.5D ONNX detector → geometry and calibrated projection → tracking → structured states `{id, x, y, yaw}`.

Communication stage: UDP transmission → CARLA server → spawn/update/remove a vehicle actor keyed by ID.

In this repository, `src.communication` is a runnable **protocol integration demo** with synthetic states. `src.integration` retains historical integration scripts and does not include their external team dependencies.

The wire format here mirrors the recorded whitespace-separated `id x y yaw` format. `yaw` in CARLA `Rotation(yaw=...)` expects degrees; coordinate-frame alignment and units must be verified against the sender before real-world use. This example does not supply real camera calibration or a valid CARLA map transform.
