# 개발 환경 및 환경 재현

## 1. 개발 환경 개요

본 프로젝트에서는 KIT 2.5D 객체 인식 모델을 기반으로 차량 검출, 추적, BEV 시각화 및 CARLA 연동을 진행했습니다.

| 항목 | 기준 또는 확인된 정보 |
|---|---|
| 운영체제 | Ubuntu 20.04 LTS — 팀 환경 구축 기록 |
| CARLA | 0.9.13 — 당시 **설치 예정**으로 기록, 최종 실행 버전 미확인 |
| KIT 객체 인식용 Python | 3.10 — KIT 기준 환경 |
| KIT 객체 인식용 CUDA | 12.8 — KIT 기준 환경 |
| cuDNN | `cudnn-cuda-12` — KIT 기준 환경 |
| 추론 환경 | ONNX Runtime GPU / FP16 ONNX 모델 |
| 영상 처리 및 좌표 변환 | OpenCV, NumPy 및 관련 라이브러리 |
| 카메라 입력 | DepthAI 카메라 또는 저장 영상 |

KIT의 기준 버전이 팀 PC에 설치된 **실제 버전과 정확히 일치하는지는 별도 검증되지 않았습니다.** KIT 원본 `requirements.txt`에는 개별 패키지 버전이 일괄 고정되어 있지 않으므로 임의의 버전 번호를 추가하지 않았습니다.

- [KIT 원본 코드 및 설치 안내](https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection)
- [Ubuntu·CARLA 관련 팀 회의 기록](https://app.notion.com/p/30ef6ee804a68188b880c93d7eda0162)

## 2. 추가 연구 환경

팀 프로젝트 과정에서는 CoBEV 등 도로변 인지 모델도 검토했습니다. 관련 기록에는 CUDA / PyTorch, MMCV, MMDetection3D, spconv 및 사용자 정의 CUDA 확장 모듈 간 **버전 호환성 문제**가 포함되어 있습니다.

CoBEV 환경 구축은 학기 중 완료한 것으로 기록되어 있지만, 해당 개발 환경을 그대로 재현할 수 있는 **검증된 Dockerfile**은 이 저장소에 포함되어 있지 않습니다.

## 3. 실행 범위

- **UDP 통신 데모:** 별도 PyTorch, ONNX 또는 CARLA 설치 없이 실행할 수 있습니다.
- **CARLA 연동 데모:** 실행할 CARLA 버전에 맞는 Python API를 별도로 설치해야 합니다.
- **전체 객체 인식 파이프라인:** KIT 모델·추론 코드, DepthAI 또는 영상 입력, 카메라별 캘리브레이션, 호환되는 런타임 등이 필요합니다.

실행 명령은 메인 [README](../README.md), 통합 환경 주의사항은 [통합 안내](integration.md)를 참고하세요.
