# Avian Intelligence

A production-oriented bird species identification application built with:

- FastAPI
- PyTorch
- torchvision
- EfficientNet-B3
- CUB-200-2011
- HTML/CSS/JavaScript
- pytest
- Docker

## Current behaviour

The application automatically uses the trained EfficientNet-B3 checkpoint when:

`models/best_model_finetuned.pth`

exists.

Until that checkpoint exists, the API falls back to a deterministic mock classifier so the entire backend/frontend can be developed and tested immediately.

## Windows local setup

```powershell
cd C:\Users\rohan\bird-identifier
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Open:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health/

## Tests

```powershell
pytest tests/ -v
```

## Dataset

The training pipeline expects CUB-200-2011 in:

```text
CUB_200_2011/
```

Then:

```powershell
python -m ml.prepare_cub
```

The prepared dataset becomes:

```text
data/birds/train/
data/birds/test/
```

## Training

Training is intended for a GPU runtime such as Google Colab:

```powershell
python -m ml.train
```

The trained checkpoint is:

```text
models/best_model_finetuned.pth
```

## Evaluation

After training:

```powershell
python -m ml.evaluate
```

This produces:

```text
models/evaluation_results.json
```

Record the actual measured top-1 and top-5 accuracy. Do not use example numbers from the textbook as your own results.

## API

POST:

```text
/api/v1/predict
```

Multipart field:

```text
image
```

Optional backend:

```text
auto
pytorch
mock
```

`auto` uses PyTorch when the checkpoint exists and mock otherwise.

## Docker

```powershell
docker build -t avian-intelligence .
docker run -p 8000:8000 --env-file .env avian-intelligence
```

The trained `.pth` file is intentionally ignored by Git. Keep model artifacts out of source control unless you explicitly choose an artifact-storage strategy.

## Study later

The implementation follows the progression of the Avian Intelligence textbook:

1. FastAPI foundation
2. Computer vision foundations
3. CNN architectures
4. Transfer learning
5. CUB-200-2011 dataset
6. Training
7. Evaluation
8. PyTorch/FastAPI integration
9. Testing
10. Frontend
11. Deployment

The point of this bundle is to get the system assembled first. We can then return to the chapters and reverse-engineer every line until you can explain the entire project in an interview.
