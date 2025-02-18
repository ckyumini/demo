# KoreanSTRExplainer

한글 Scene Text Recognition(STR) 모델의 예측을 설명하기 위한 라이브러리입니다.

## 주요 기능

- 전역적/지역적/결합된 설명 생성
- 자모 단위 분석
- 획순 기반 분석
- GPU 가속 지원
- 실시간 진행 상황 모니터링

## 설치 방법

pip install torch captum scikit-image numpy matplotlib

## 사용 예시

(사용 예시에 대한 예제 코드를 여기에 추가하세요)

## 주요 컴포넌트

- ModelWrapper
- KoreanSTRExplainer

## 주요 메서드

### 전역적 설명
- _compute_global_attribution: 이미지 전체에 대한 attribution 계산
- _process_global_attr: 전역적 설명 후처리

### 지역적 설명
- get_jamo_attribution: 자모 단위 분석
- get_stroke_attribution: 획순 기반 분석
- _compute_local_attribution: 문자별 지역적 설명 계산

### 결합 설명
- _combine_explanations: 전역적/지역적 설명 통합
- _get_context_importance: 문맥 기반 중요도 계산

## GPU 최적화

- 자동 배치 크기 조정
- 메모리 캐싱
- 혼합 정밀도 연산
- 동적 메모리 관리

## 로깅 시스템 및 시각화

- 전역적 설명 히트맵
- 지역적 설명 히트맵
- 결합된 설명 히트맵

## 주의사항

- GPU 메모리 관리: 대량 처리 시 메모리 모니터링 필요 및 주기적인 캐시 정리 권장
- 성능 최적화: 배치 크기 적절히 조정하고 캐시 시스템을 활용할 것
- 에러 처리: 모든 주요 연산에 예외 처리를 구현하고 로그 파일을 확인할 것

## 라이선스

MIT License

## 참조

- Captum
- PyTorch
- scikit-image
