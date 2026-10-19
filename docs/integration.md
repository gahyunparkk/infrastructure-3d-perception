# Integration notes

The 2025 team scripts are stored in `src/team_pipeline/`, including historical versions. They are preserved for review and are not uniformly verified end-to-end.

## Full detector setup

Clone the original [KIT 2.5D Object Detection](https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection) repository separately and use its documented Git LFS workflow to acquire `checkpoints/model_v2_half.onnx`. Do not commit upstream model weights to this repository. Use the upstream inference code in its original repository to validate compatibility before integrating with the team pipeline.

The latest archived `obstacles_live_latency_v2.py` imports `detection_inference_onnx_v2`, and the BEV/tracking modules listed in the team source directory. `detection_inference_onnx_v2` was excluded here pending verification of redistribution rights; copy or adapt it only under the applicable permissions. Scripts may also require DepthAI hardware, camera-specific homography configuration and CARLA.

For a verified immediately runnable component without KIT dependencies, use the stdlib UDP demo documented in README. Port numbers in historical source files differ from the UDP demo; set up network endpoints consistently before running full integration. Historical LAN addresses have been replaced with loopback addresses for privacy.
