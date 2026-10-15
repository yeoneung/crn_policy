# Automatica 원고 — finite-power 인증을 포함한 12쪽 원고

**Control Sensitivity and the Cost of Certifying Small Policy Improvements**  
Kyunghyun Lee · Yeoneung Kim (교신저자)

사용자가 요청한 **2단 12쪽**을 목표로 SCL 8쪽판의 설명을 확장한 독립 원고입니다.
참고문헌과 선언문을 포함해 실제 빌드 결과가 12쪽입니다. 글꼴 크기·여백·줄 간격을
임의로 줄이지 않았습니다. 기존 `scl_submission`과 `MCS_submission`은 수정하지 않았습니다.

| 파일 | 내용 |
|---|---|
| `main.pdf` | 전체 원고 PDF, 12쪽 |
| `main.tex` | 초록·본문·부록의 모든 증명·실험·선언문을 담은 유일한 TeX 파일 |
| `ref.bib` | 본문에서 인용한 문헌 22개 |
| `autart.cls` | Automatica 전용 2단 클래스, 원본 그대로 유지 |
| `elsarticle-harv.bst` | 저자–연도 인용과 알파벳 순서 참고문헌 |
| `certification_mechanism.pdf` | 동결된 실험 기록에서 재생성한 벡터 그림 |
| `output/Automatica_manuscript.pdf` | 업로드용 원고 사본 |
| `output/Automatica_source.zip` | 다른 폴더에서 독립 재빌드한 제출 소스 |
| `output/validation.json` | 쪽수·교차 참조·소스 ZIP 재빌드 확인 |
| `evidence/` | 그림의 원자료와 개별 실험 기록 대조 결과 |
| `audit_matrix_certificate.py` | 새 finite-power 정리의 수치 스트레스 검사 |
| `certification/finite_power.py` | calibration·block 인증·return support·고정 표본 validation 구현 |
| `examples/nonnormal_certificate.py` | 새 분석 예제의 독립 calibration + fresh validation 실행 |
| `examples/finite_power_study.py` | 추가 nonnormal 실험과 모델 기반 참조 방법 비교·재현 |
| `certification/model_reference.py` | matrix·noise uncertainty를 반영한 horizon-three 모델 기반 gain 인증 |
| `evidence/finite_power_study/` | 추가 실험의 최종 protocol·reset 데이터·return banks·결과 |
| `CODE_README.md` | 최종 코드 패키지 설치·API·가정·재현 안내 |
| `reproduce.py` | 전체 패키지의 단일 검증·재현 진입점 |
| `Automatica_code.zip` | 현재 원고·새 구현·기존 코드·원자료를 포함한 최종 독립 실행 패키지 |
| `Automatica_code.zip.sha256` | 최종 코드 ZIP의 SHA-256 확인값 |
| `reproducibility/crn_policy/` | 기존 동결 코드·실험 자료의 독립 사본 |

## 스토리라인과 확장 내용

1. 작은 제어기 개선은 왜 학습 이후에도 검증 비용을 요구하는가.
2. 같은 양의 관측을 보아도 ordinary range만으로 검증할 때 드문 손실을 배제하지 못하는 이유.
3. baseline closed-loop sensitivity와 강제 action displacement가 paired-return support를 제공하는 방법.
4. support와 gain이 함께 줄어들 때의 검증 비용, 확률적 sharpness 예제, actuator geometry.
5. 한 단계 수축을 요구하지 않는 finite-power certificate와 그 획득 비용을 포함한 총비용 정리.
6. regional escape allowance를 확보하는 비용 때문에 절감이 사라질 수 있는 경우.
7. 같은 정책과 관측에서 정보만 바꾼 실험, calibration 비용을 포함한 실험, 기계 시스템 사례.

SCL판보다 동기·문헌상 위치·정리 사이의 연결을 풀어 썼습니다. 이후 가독성을 위해
서론의 반복을 줄이고, 2절의 통계 설명을 세 소절로 정리해 제어 민감도 정리에 더 빨리
도달하도록 했습니다. 4절은 인증 방법 → 총비용 정리 → 비용이 일정하게 유지되는
이유의 순서로 읽힙니다. 이후 요청에 따라 총비용 정리를 finite-power 조건으로
일반화했습니다. 기존의 반복적인 정보·비용 요약표는 새 제어 결과와 분석 예제에
지면을 배분하기 위해 제거했습니다. 기존 수치 실험과 두 결과표는 유지합니다.

최신 흐름 검토에서는 첫 두 문단에 작은 개선의 검증 비용이라는 질문과 인증 획득
비용을 포함하는 핵심 답을 배치했습니다. 총비용 정리 앞에 고정된 matrix accuracy가
충분한 이유를 먼저 설명하고, 한 단계 수축 특수 경우의 계산은 부록 A.2로 옮겼습니다.
정리 8개와 증명 8개의 내용, 가정, 문헌 및 실험 수치는 유지했습니다.
해당 변경의 대조 기록은 `evidence/flow_review.json`에 있습니다.

그 뒤 문헌·비교 방법·새 정리의 수치 연결을 보완했습니다. 데이터 효율적 policy
optimization, direct/indirect control, stability-constrained learning 및 robust
certificate 학습 문헌 4개를 추가해 학습 비용과 학습 후 채택 검증 비용을 구분했습니다.
기존 결과표는 보존하고 24회의 독립 calibration/validation 반복을 별도로 추가했습니다.
같은 비선형 nonnormal 시스템에서 ordinary, finite-horizon support, finite-power support,
동일 calibration 정보로 직접 gain을 인증하는 모델 기반 참조 방법을 비교합니다.
새 Table 3과 Appendix C에 방법·결과·모델 기반 참조의 유도를 포함했습니다.

세 변위에서 block 방법은 각각 112 pairs와 총 1,018 transitions를 사용합니다.
짧은 horizon에서는 finite-horizon 방법이 442 transitions, 예제에 특화된 모델 기반
방법이 346 transitions로 더 저렴합니다. 이러한 결과도 그대로 보고하며, 모든 모델
기반 방법에 대한 우위나 실제 비용 상수의 최적성을 주장하지 않습니다.
기존 동결 실험의 seed·코드·결과는 변경하지 않았습니다. 정리·증명·기존 결과표와
새 비교 실험의 검토 기록은 `evidence/strengthening_review.json`에 있습니다.

## 한 단계 수축 조건을 완화한 결과

- **Corollary 5:** 데이터로 인증한 `M_hat`의 유한 개 matrix powers에서 작은 gain을
  확인하면 모든 시간에 유효한 state-difference envelope를 얻습니다. 알려진 actuator
  image도 유지할 수 있습니다.
- **Theorem 7:** matrix error의 목표 정확도와 robust block margin을 고정하면,
  replicated-reset 수와 validation pair 수가 모두 `d`에 무관합니다. 명시적 reset
  수와 전체 확률 계산은 부록 A.2에 있습니다.
- **선형 경우:** 모든 Schur-stable `M`은 적당한 유한 block 길이와 양의 정확도
  목표를 허용합니다. `||M||_2 > 1`인 경우도 포함합니다. 비선형 residual이 있으면
  잔차와 matrix uncertainty를 포함하는 별도 양의 robust margin이 필요합니다.
- **Horizon:** 할인율이 1보다 작고 gain margin이 공통으로 유지되면 pair 수의 상한도
  유한 horizon 전체에 걸쳐 동일하게 잡을 수 있습니다. 실제 transitions는 여전히
  `2 N_c + 2 H N_v`입니다.
- **분석 예제:** `M=[[1/2,1],[0,1/2]]`, residual `0.001 tanh(x)`에서
  `||M||_2=1.2071`로 한 단계 기준은 실패합니다. `K=4`, matrix accuracy 목표 `0.02`,
  failure allowance `0.05`에서는 `173` replicated resets, 즉 `346` transitions가
  충분합니다. `S_*=4.1781`, `eta_*=0.3196`이며, 실제 bounded-reward 제어 문제에서
  양의 선형 gain margin도 증명했습니다.

유한 horizon의 `O(d)` bound 자체는 기존 matrix recursion으로도 얻을 수 있습니다.
새 결과는 **시간 전체에서 유효한 상한, transient amplification과 matrix uncertainty의
명시적 여유, 그 인증 획득 비용**을 연결합니다. small-gain 원리 자체는 새 결과로
주장하지 않으며 Doyle–Francis–Tannenbaum의 고전 문헌을 추가했습니다.

통계 하한의 짧은 증명, 핵심 민감도 정리, matrix envelope 및 regional 결과는 본문에
있습니다. 긴 calibration 계산은 같은 `main.tex` 안의 부록으로 옮겼습니다.

- 부록 A.1: baseline calibration의 정확한 pivot과 증명.
- 부록 A.2: matrix accuracy 목표를 포함한 명시적 reset 수와 새 총비용 정리의 전체 증명.
- 부록 A.3: 한 단계 수축이 실패하는 비선형 stochastic 예제에서의 gain margin 증명.
- 부록 A.4: unknown Gaussian covariance에 대한 rowwise certificate와 증명.
- 부록 B: 특정 betting stake의 power 보장.
- 부록 C: 동일 calibration 데이터에 기반한 horizon-three 모델 기반 gain 인증.

정리의 가정·오류확률·power·총비용 보장은 본문에 남겼고, 부록을 직접 참조하게 했습니다.
외부 보충 원고에 의존하지 않습니다. `needspace`는 증명 시작이나 정리의 조건이
부자연스럽게 끊기는 것을 줄이는 데 사용하며, 클래스의 글꼴·여백은 바꾸지 않습니다.

기여는 새로운 CRN 기법이나 집중 부등식이 아니라, **제어 민감도로 paired support를
정당화하고 그 정보의 획득 비용을 함께 분석하는 연결**로 기술합니다.
`O(1)`은 고정된 신뢰수준·matrix margin·양의 gain/displacement 비율 아래에서 `d`에
대한 결론입니다. 고정 할인율 아래의 uniform pair budget과 실제 horizon별 transition
비용을 구별했습니다. 하한과 상한이 서로 다른 정당화된 정보 클래스에 관한 결과임을
명시했습니다. 지역적 결과에서도 `mu - 2q`의 양의 여유를 유지합니다.

## 템플릿 출처와 분량 기준

Automatica 전용 `autart` 클래스와 저자–연도 참고문헌 형식을 사용했습니다.
[Overleaf의 Automatica 템플릿](https://www.overleaf.com/latex/templates/template-for-automatica-ifac-journal/wxstjkyjmkmp)은
출처를 Elsevier 저자 가이드로 명시하고 `autart.cls` 사용법을 제공합니다.
직접적인 과거 publisher/autsubmit 클래스 다운로드는 현재 접근되지 않아,
같은 클래스의 공개 저장소 사본을 내려받아 수정 없이 포함했습니다. 클래스 내부의
Elsevier 저작권·배포 라이선스를 보존했습니다. `elsarticle-harv.bst`는 기존에 내려받은
공식 CTAN `elsarticle` 패키지에서 가져왔습니다. URL·SHA-256·클래스 버전은
`template/download_manifest.json`에 있습니다.

**12쪽은 이번 작업에서 사용자가 지정한 작성 목표입니다.**
현재 [Automatica 공식 Guide for Authors](https://www.sciencedirect.com/journal/automatica/publish/guide-for-authors)는
HTTP 403으로 열리지 않아 최신 page limit 전체를 확인했다고 주장하지 않습니다.
오래된 대학 도서관 미러의 10쪽 안내와 최신 규정을 혼동하지 않았으며, 현재 버전을
공식 가이드 전체 및 투고 시스템 필수 항목까지 검증한 제출본으로 표시하지 않습니다.

## 빌드

pdfLaTeX·BibTeX와 Python `pypdf`가 필요합니다.

```sh
python build.py
```

빌드는 임시 폴더에서 수행합니다. ZIP을 별도 폴더에 풀어 재빌드한 뒤 PDF의 페이지별
텍스트가 일치하는지도 확인합니다. 필요하면 TeX 실행 파일 폴더를 `AUTOMATICA_TEXBIN`으로
지정할 수 있습니다. 소스 ZIP의 구성은 평면 구조이며 원고, 문헌, 클래스, BST, 그림,
생성된 BBL의 6개 파일뿐입니다. 원고 빌드에 상위 폴더나 실험 archive가 필요하지 않습니다.

소스 ZIP만 받은 경우:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

## 수치와 재현

최종 코드 패키지에서는 다음으로 검증·재현합니다. Python 3.11, CPU 환경이며,
원자료 ZIP도 포함해 Python 의존성 설치 이후에는 네트워크가 필요하지 않습니다.

```sh
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py quick
python reproduce.py all
```

`quick`은 새 구현의 16개 테스트와 기존 15개 테스트, 정리의 스트레스 검사 및 실제
calibration → block 인증 → fresh paired-return validation 예제를 실행합니다.
`all`은 추가 실험 replay·독립 audit, 기존 모든 replay 그룹과 원고 표·그림 재집계까지 포함합니다. 결과·로그는
새 `runs/` 하위 폴더에 저장되며, 원자료와 배포 입력 파일은 보존합니다.
단일 분석 예제는 정리의 보수적 충분 예산을, 추가 실험은 원고의 sequential mixture를
사용합니다. 이 둘은 서로 다른 검증 절차이며 pair 수를 혼동하지 않습니다.
자세한 API와 가정은 `CODE_README.md`에 있습니다.

배포 ZIP과 SHA-256은 이 폴더 바로 아래에 있으며,
`python package_code.py`로 다시 생성할 수 있습니다.
아래는 각 작업을 직접 실행하는 기존 진입점입니다.

```sh
python make_figure.py
python audit_evidence.py
python audit_matrix_certificate.py
cd reproducibility/crn_policy
python reproduce.py verify
```

그림 재생성과 대조에는 NumPy·Matplotlib가 필요합니다. 그림은 기존의
`v2_coupling_sensitivity/summary.json`에서 재집계했고, 두 실험 결과표는 원래의
개별 plant 기록에서 채택 수·사용 pairs·gain·SE·transitions를 재계산해 대조했습니다.
기존 동결 실험을 재학습하거나 새 rollout으로 대체하지 않았고 seed를 재선택하지도
않았습니다. 이 실험의 수치 감사 기록은 `evidence/`에 있습니다.
독립 성능평가 bank는 채택 판단에 사용하지 않는 연구용 평가 자료이며, 표의 사용
transitions는 표에 명시한 학습·calibration·validation의 합계입니다.

기존 수치 실험과 별도로 `audit_matrix_certificate.py`는 새 정리의 검산을 위해
비가환 matrix perturbation, 임의의 residual 방향을 넣은 상태 궤적, Gaussian reset
calibration 1,000회 및 예제의 time-two gain 항등식을 점검합니다. 결과는
`evidence/matrix_certificate_audit.json`에 있습니다. 이 수치 진단은 증명을 대체하지
않으며, 기존 동결 실험의 seed·자료·결과와 섞지 않았습니다. 이 진단에는 SciPy도 필요합니다.

저자·소속·연구비·CRediT·데이터 가용성은 기존 원고에서 유지했습니다. AI 선언은 GPT만
명시하는 두 문장입니다. 이 패키지 생성은 실제 저널 투고를 수행하지 않습니다.
