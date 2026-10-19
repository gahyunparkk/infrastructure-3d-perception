# Infrastructure 3D Perception

**2025 캡스톤디자인 팀 프로젝트 | 도로변 카메라 기반 2.5D 객체 인식 및 자율주행 인프라 시각화**

도로변 카메라 영상에서 차량을 인식하고, 객체의 위치와 방향을 추정해 추적한 뒤, BEV(Bird's-Eye View) 지도와 CARLA 시뮬레이터에 전달하는 인프라 인지 시스템을 구현한 팀 프로젝트입니다.

본 저장소에는 팀의 **통합 코드와 UDP 통신 데모**가 포함되어 있습니다. 실제 카메라부터 CARLA까지 모든 기능을 실행하려면 외부 객체 인식 모델과 해당 하드웨어·실행 환경이 추가로 필요합니다.

## 주요 기능

- **2.5D 객체 인식 연동:** 도로변 RGB 카메라 또는 저장된 영상에서 차량 검출
- **좌표 변환 및 캘리브레이션:** 이미지 좌표를 월드 좌표로 변환하고 차량의 방향(yaw) 계산
- **SORT 기반 객체 추적:** 프레임 간 차량 ID 유지 및 위치 추적
- **BEV 시각화:** 추적 결과를 탑다운 뷰 지도에 표현
- **실시간 처리 및 지연시간 분석:** 영상 수신·추론·후처리 단계의 처리 시간 측정
- **UDP–CARLA 연동:** 차량의 ID·위치·방향을 네트워크로 전달하여 시뮬레이터에서 표현

## 시스템 구성

```text
도로변 RGB 카메라 / 저장된 영상
  ↓
2.5D 객체 인식 (KIT ONNX 모델, 외부 구성요소)
  ↓
카메라 캘리브레이션 및 픽셀 → 월드 좌표 변환
  ↓
SORT 기반 객체 추적
  ↓
BEV 시각화 및 지연시간 측정
  ↓
UDP 통신 (id, x, y, yaw)
  ↓
CARLA 시뮬레이터 차량 시각화
```

## 개발 환경

2025년 프로젝트 진행 당시의 회의록 및 저장된 코드를 바탕으로 정리한 환경입니다. **실제 사용 기록**과 **당시 설치 계획**을 구분했으며, 확인되지 않은 세부 버전은 임의로 기입하지 않았습니다.

| 구분 | 개발 환경 / 사용 기술 | 확인 상태 |
|---|---|---|
| 운영체제 | **Ubuntu 20.04 LTS** | 2025.08.08 회의록의 대여 노트북 환경 구성 기록 |
| 시뮬레이터 | **CARLA 0.9.13** | 동일 회의록에 명시된 **설치 예정 버전** (최종 실행 버전 별도 확인 필요) |
| 프로그래밍 언어 | Python | 당시 전체 파이프라인의 정확한 세부 버전 미확인 |
| 객체 인식 | KIT 2.5D Object Detection, ONNX (`model_v2_half.onnx`) | 코드 및 실행 명령에서 확인 |
| 카메라 및 입력 | DepthAI 기반 RGB 카메라 / 저장된 영상 | 팀 실시간 처리 코드에서 확인 |
| 객체 추적 및 시각화 | SORT, OpenCV, BEV | 팀 코드에서 확인 |
| 추론 및 연산 | ONNX Runtime, PyTorch, NumPy | 팀/원본 코드에서 사용 확인; 정확한 패키지 버전 미확인 |
| 네트워크 통신 | UDP | 팀 CARLA 연동 코드에서 확인 |
| CARLA Python API | 설치된 CARLA 버전과 호환되는 패키지 필요 | 당시 사용된 wheel의 정확한 버전 미확인 |

> **버전 기록 참고:** [2025.08.08 주간회의](https://app.notion.com/p/30ef6ee804a68188b880c93d7eda0162)에는 Ubuntu 20.04 환경 구축과 CARLA 0.9.13 설치 계획이 기록되어 있습니다. 최종 확정된 CARLA·Python·CUDA 및 주요 라이브러리 버전은 당시 실행 PC의 기록이나 패키지 목록으로 재확인해야 합니다.
>
> 아래의 **Python 3.10 이상** 조건은 이 저장소의 **독립 UDP 데모** 실행 기준이며, 2025년 전체 통합 시스템에서 실제 사용한 Python 버전이라는 뜻은 아닙니다.

## 실행 방법

### 1. UDP 통신 데모

**Python 3.10 이상** 환경에서 ONNX 모델, 카메라 및 CARLA 없이 통신 흐름을 확인할 수 있습니다. 저장소 루트 경로에서 터미널 두 개를 실행합니다.

**터미널 A — 수신 프로그램**

```bash
python -m src.communication.receiver --bind 127.0.0.1 --port 5001
```

**터미널 B — 송신 프로그램**

```bash
python -m src.communication.sender_demo --frames 20
```

**단위 테스트**

```bash
python -m unittest discover -s tests -v
```

CARLA와 연결하려면 해당 CARLA 버전의 Python API를 설치하고 다음 명령으로 제공되는 옵션을 확인하세요.

```bash
python -m src.communication.receiver --help
```

`--carla` 및 시뮬레이터 호스트·포트 옵션을 사용할 수 있습니다. 단, 패키지의 CARLA 연동 기능은 실제 시뮬레이터 환경에서 별도로 검증해야 합니다.

### 2. 2.5D 객체 인식 모델 준비

객체 인식에 사용하는 **`model_v2_half.onnx` 가중치는 이 저장소에 포함하지 않습니다.** 모델과 원본 추론 구현은 KIT의 공식 GitLab에서 확인할 수 있습니다.

```bash
git lfs install
git clone https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection.git
cd 2.5d-object-detection
git lfs pull
```

공식 저장소의 `checkpoints/model_v2_half.onnx` 파일을 확인한 후, [KIT 원본 README](https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection)에 따라 모델 추론 환경을 구성하세요.

팀의 실시간 처리 파이프라인에는 해당 모델 외에도 DepthAI 카메라, 카메라별 캘리브레이션 설정, 호환되는 ONNX Runtime 및 CARLA 환경 등이 필요할 수 있습니다. 자세한 내용은 [통합 실행 안내](docs/integration.md)와 [환경 설정 안내](docs/environment.md)를 참고하세요.

> **실행 범위 안내:** 위 UDP 데모는 독립적으로 실행할 수 있지만, 저장소의 기존 팀 통합 코드는 당시 개발 환경을 바탕으로 보존된 버전도 포함합니다. 전체 카메라 → 인식 → 추적 → BEV → CARLA 과정을 새 환경에서 자동으로 재현하는 테스트는 완료되지 않았습니다.

## 프로젝트 디렉터리 구조

```text
src/
  communication/    UDP 송신·수신 및 선택적 CARLA 연동 데모
  team_pipeline/    팀 통합 코드, SORT, BEV, 캘리브레이션 관련 버전
docs/               시스템 구성, 실행 환경, 통합 과정 및 외부 출처
tests/              UDP 프로토콜 단위 테스트
scripts/            데모 실행 보조 스크립트
checkpoints/        외부 모델 다운로드 안내 (가중치 미포함)
samples/            예제 데이터 준비 안내
```

## 참고 논문 및 외부 코드 출처

본 프로젝트는 KIT 연구진이 공개한 **2.5D 객체 인식 연구**를 활용해 객체 추적, 좌표 변환, BEV 시각화, 지연시간 분석 및 CARLA 통신을 통합한 팀 프로젝트입니다.

- **참고 논문:** [2.5D Object Detection for Intelligent Roadside Infrastructure (arXiv:2507.03564)](https://arxiv.org/abs/2507.03564)
- **논문 전문:** [arXiv HTML](https://arxiv.org/html/2507.03564)
- **공식 구현 및 모델:** [KIT 2.5D Object Detection — GitLab](https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection)
- **외부 구성요소 관련 안내:** [docs/third-party.md](docs/third-party.md)

**2.5D 객체 검출 모델과 원본 추론 코드는 본 캡스톤 팀이 개발한 결과물이 아닙니다.** 해당 연구의 원저작자를 명시하고 있으며, 사전 학습된 ONNX 가중치는 이 저장소에서 직접 재배포하지 않습니다. 공개 저장소에서 코드를 내려받을 수 있다는 사실만으로 모든 재배포 권한이 허용되는 것은 아니므로, 원본 구성요소를 사용할 때는 원저작자의 안내를 확인하세요.
