# 통합 실행 안내

2025년 캡스톤 팀의 통합 스크립트는 `src/team_pipeline/`에 보존되어 있습니다. 개발 과정의 여러 버전이 포함되어 있으며, **모든 버전의 전체 파이프라인 실행이 검증된 것은 아닙니다.**

## 1. KIT 객체 인식 모델 준비

2.5D 객체 인식 모델 및 원본 추론 코드는 [KIT 공식 GitLab](https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection)에서 별도로 내려받습니다.

```bash
git lfs install
git clone https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection.git
cd 2.5d-object-detection
git lfs pull
```

`checkpoints/model_v2_half.onnx`가 준비되었는지 확인하고, 먼저 KIT 원본 저장소에서 추론 코드가 정상 작동하는지 검증하는 것을 권장합니다. **원본 모델 가중치는 이 저장소에 커밋하지 않습니다.**

## 2. 팀 통합 코드 관련 주의사항

보존된 `obstacles_live_latency_v2.py`는 `detection_inference_onnx_v2`와 팀의 BEV·추적 모듈을 불러옵니다.

`detection_inference_onnx_v2`는 원본 코드의 재배포 권한을 확인하지 못해 현재 저장소에서 제외했습니다. 원본 모듈을 복사하거나 수정할 때는 해당 저작권 및 사용 조건을 확인해야 합니다.

전체 통합 스크립트 실행에는 다음이 추가로 필요할 수 있습니다.

- DepthAI 지원 카메라 및 관련 환경
- 카메라별 호모그래피(homography)·캘리브레이션 설정
- KIT 모델과 호환되는 추론 런타임
- CARLA 시뮬레이터 및 해당 버전의 Python API

## 3. UDP 데모와 포트 설정

KIT 모델 없이 즉시 실행해 볼 수 있는 항목은 [메인 README의 UDP 통신 데모](../README.md#실행-방법)입니다.

기존 팀 소스의 UDP 포트와 독립 데모의 기본 포트는 다를 수 있으므로, 전체 연동 시에는 송신 측과 수신 측의 **IP 주소 및 포트를 동일하게 맞춰야 합니다.**

공개 저장소에는 과거 내부 네트워크의 IP 주소 대신 루프백 주소(`127.0.0.1`)를 사용하도록 정리했습니다.

## 4. 실행 검증 범위

UDP 프로토콜 데모는 별도로 검증되었지만, 카메라 → KIT 객체 인식 → 추적 → BEV → CARLA의 **전체 구성을 새로운 PC에서 재현하는 통합 테스트는 완료되지 않았습니다.**
