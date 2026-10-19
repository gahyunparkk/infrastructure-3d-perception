# Infrastructure 3D Perception — Team Capstone Project

Roadside-camera perception, real-time 2.5D vehicle detection integration, object tracking, bird's-eye-view (BEV) visualization, UDP transfer, and CARLA digital-twin integration, developed in a **2025 team capstone project**. This repository publishes the team integration code and a small runnable network-protocol demonstration. It is **not a claim that the complete camera-to-CARLA system runs on any PC without the external model and hardware**.

## Reference research and attribution

The original 2.5D object detector was developed by the researchers of **[2.5D Object Detection for Intelligent Roadside Infrastructure](https://arxiv.org/abs/2507.03564)**.

- **Paper:** [arXiv:2507.03564](https://arxiv.org/abs/2507.03564) ([full text](https://arxiv.org/html/2507.03564))
- **Official implementation and model:** [KIT 2.5D Object Detection GitLab](https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection)

The object detection model and upstream inference implementation are **not originally developed by this capstone team**. We used them as external research components and built integration, tracking, mapping, visualization, latency measurement, and simulator communication around them. **The pretrained `model_v2_half.onnx` file is not redistributed in this repository.** Public access to an upstream repository alone does not establish third-party redistribution rights. See [third-party notes](docs/third-party.md).

## Architecture

```text
Roadside RGB camera / recorded video
  → KIT 2.5D ONNX inference (external dependency)
  → pixel-to-world projection / camera calibration
  → SORT-based object tracking
  → BEV visualization and latency profiling
  → vehicle states (id, x, y, yaw) via UDP
  → CARLA vehicle representation
```

This is a team project; collaborators can be added via GitHub and tracked through commits.

## Quick start: UDP transport demonstration

Works with **Python 3.10+**, without ONNX, camera hardware or CARLA. From the repository root, use two terminals.

Terminal A:

```bash
python -m src.communication.receiver --bind 127.0.0.1 --port 5001
```

Terminal B:

```bash
python -m src.communication.sender_demo --frames 20
```

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

To use a running CARLA simulator with the demo, install the matching CARLA Python API and follow `python -m src.communication.receiver --help` for `--carla` and simulator host/port settings. CARLA integration is **not hardware-verified in this packaged distribution**.

## Original model download (not bundled)

Install Git LFS, then obtain the detector **directly from KIT**:

```bash
git lfs install
git clone https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection.git
cd 2.5d-object-detection
git lfs pull
```

Confirm `checkpoints/model_v2_half.onnx` exists in the KIT checkout and follow the [upstream README](https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection) for their tested inference prerequisites. On the team-side, the historical live pipeline used DepthAI and camera-specific calibration, plus the external detector; copying its original code into this repository is **not required** to study the integration work. For full reconstruction and remaining dependencies see [integration notes](docs/integration.md).

## Repository layout

```text
src/communication/       runnable UDP sender / receiver / optional CARLA bridge
src/team_pipeline/       team integration, tracking, BEV and calibration versions
docs/                    architecture, environment, integration, upstream attribution
tests/                   protocol unit tests
scripts/                 quick demo helper
checkpoints/             model download directions (no weights)
samples/                 sample-asset directions (no large recordings)
```

## Scope

The UDP demonstration is reproducible without external dependencies. Historical team pipeline scripts are provided for inspection and for integration in a compatible research environment; several versions reflect intermediate development and can require the upstream inference implementation, particular calibration settings, DepthAI camera, compatible GPU/ONNX runtime, and CARLA. No fully automated end-to-end test has been run against that hardware stack here.
