# EMBwithLLM: Improve text embeding with efficient LLM

LLM을 텍스트 임베딩 성능 개선에 도입한 연구로 비용적인 측면에서 저비용의 임베딩 모델과 작은 파라미터를 가진 LLM 모델을 사용하여 임베딩 성능을 개선함으로써 저비용으로 고효율 실현한다.

논문은 [여기](#)서 확인할 수 있습니다.

## Quick Links


- [EMBwithLLM: Improve text embeding with efficient LLM](#embwithllm-improve-text-embeding-with-efficient-llm)
    - [Quick Links](#quick-links)
    - [Embedding Model](#embedding-model)
    - [LLM Model](#llm-model)
    - [Datasets](#datasets)
    - [Installation](#installation)
    - [Getting Started](#getting-started)
    - [Result](#result)

## Embedding Model


임베딩 모델로는 파라미터 수는 110M으로 대규모 모델보다 적은 편에 속하며, 상당히 작은 크기에도 불구하고 다양한 NPL 작업 및 코드 검색 등 다양한 도메인에서 높은 성능을 보인`gte-large` 를 사용하였습니다.

 [https://huggingface.co/thenlper/gte-large](https://huggingface.co/thenlper/gte-large)

## LLM Model


LLM모델은 사이즈가 작아 활용성이 높고 임베딩 미세조정의 성능을 떨어뜨리지 않는 합리적인 모델로 Mistral기반에서 파인튜닝 된 모델인 `Starling-RM-7B-alpha` 를 사용하였습니다.

[https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha)

## Datasets

데이터셋으로는 Bank77, FewNerd, StackEx, MTOP(D) 총 4가지 데이터셋을 사용했습니다.

| Task | Name | #clusters | #data(small) |
| --- | --- | --- | --- |
| Intent | Bank77 | 77 | 3,080 |
| Type | FewNerd | 58 | 3,789 |
| Topic | StackEx | 121 | 4,156 |
| Domain | MTOP(D) | 11 | 4,386 |

## Installation

```bash
pip install -r requirements.txt

```

환경세팅은 다음과 같이 진행할 수 있으며 `run.ipynb` 에도 포함되어 있습니다. 

## Getting Started


`run.ipynb` 에서 실행할 수 있으며 노트북은 다음과 같이 구성됩니다.

### 1. 오리지널 임베딩

```bash
cd perspective/2_finetune
bash scripts/get_embedding_gte.sh
```

gte를 사용한 임베딩의 성능 지표는 `measures` 에 저장됩니다.

### 2. 샘플 삼중항

```bash
cd perspective/1_predict_triplet
bash scripts/triplet_sampling.sh

```

샘플링된 삼중항은 `perspective/1_predict_triplet/sampled_triplet_results`에 생성됩니다.

### 3. 삼중항 예측

```bash
cd perspective/1_predict_triplet
bash scripts/predict_triplet.sh

```

예측된 삼중항은 `perspective/1_predict_triplet/predicted_triplet_results`에 생성됩니다.

### 4. 삼중항 변환

```bash
cd perspective/2_finetune
bash scripts/convert_triplet.sh
bash scripts/convert_triplet_self.sh

```

변환된 삼중항은 `perspective/2_finetune/converted_triplet_results`에 생성됩니다.

### 5. 미세조정

```bash
cd perspective/2_finetune
bash scripts/finetune_gte.sh
```

미세조정된 모델은 `perspective/2_finetune/checkpoints`에 생성됩니다.

### 6. 미세조정 후 임베딩

```bash
cd perspective/2_finetune
bash scripts/get_finetuned_embedding_gte.sh
```

checkpoints로 전환 후 미세조정된 모델로 임베딩을 진행합니다. 이 임베딩 성능 지표는 `after_measures` 에 저장됩니다. 

## Result

실험 전 성능은 `measure` , 실험 후 성능은 `after_measures` 에서 확인 가능하며 다음과 같습니다. 

|데이터셋| Bank77   | FewNerd   | StackEx   | MTOP(D)   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 방법 | Acc | NMI | ACC | NMI | ACC | NMI | ACC | NMI |
| E5 | 59.90 | 77.71 | 25.49 | 40.62 | 37.31 | 58.59 | 91.23 | 57.23 |
| E5-GPT3.5 | 69.09 | 83.17 | 28.52 | 44.45 | 43.01 | 63.81 | 89.18 | 56.32 |
| GTE | 65.52 | 82.00 | 31.08 | 46.50 | 40.61 | 62.42 | 91.68 | 58.44 |
| GTE-Starling | 68.21 | 83.10 | 31.31 | 47.39 | 41.77 | 63.23 | 92.08 | 54.99 |
