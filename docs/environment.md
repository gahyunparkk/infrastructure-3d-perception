# Environment reproduction work

Team research also included CoBEV and other roadside-model candidates. Recorded challenges covered compatibility across CUDA/PyTorch, MMCV, MMDetection3D, spconv and custom CUDA extensions. CoBEV environment construction was recorded as completed during the semester; no unverified Dockerfile is presented here as a tested final build.

The packaged network demo has **no PyTorch/ONNX requirement**. CARLA integration requires the CARLA Python API installed separately; the full perception pipeline requires upstream/runtime dependencies.
