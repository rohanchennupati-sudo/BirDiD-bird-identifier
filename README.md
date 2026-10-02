# BirDiD 🐦
### *Teaching a machine to look up.*

I have spent a considerable portion of my childhood and youth looking at birds. Sometimes the bird is obvious. Sometimes you get three seconds of wings, a silhouette, a flash of colour, and then it is gone.

So I thought to myself 'what if I could teach a machine to do the looking with me,' and **BirDiD** is my attempt to do just that.

It is an end-to-end bird species identification system built around a trained **EfficientNet-B3** image classifier, wrapped in a **FastAPI** backend and exposed through a simple web interface.

The project takes a bird photograph, runs it through the trained model, and returns the species it thinks it is looking at along with a probability and its top five predictions.

It is part computer vision experiment, part backend application, and part excuse to spend even more time looking at birds. I would like to one day build the same for an exhaustive dataset of indian birds and would also like to help build the aforementioned dataset.

---

## What does it actually do?

Give BirDiD an image:
```text
        🖼️
         │
         ▼
   Image validation
         │
         ▼
  Image preprocessing
         │
         ▼
   EfficientNet-B3
         │
         ▼
    200 species
         │
         ▼
  Top predictions
```

The API accepts JPEG, PNG, and WebP images. When the trained checkpoint is available, the application uses the real PyTorch classifier. If not, it falls back to a mock classifier that returns a fixed demo answer, so the API and frontend can be developed and tested without the model. Every response includes `backend_used` (`pytorch` or `mock`) so a demo answer is never mistaken for a real one, and the mock fallback is disabled when `ENVIRONMENT=production`.

The application also supports explicitly selecting:
- `auto` — use the trained model when available, otherwise use the mock
- `pytorch` — require the trained model
- `mock` — use the mock classifier (fixed demo answer)

---

## The model

The classifier is based on **EfficientNet-B3** and was trained using the **CUB-200-2011** dataset.

CUB-200-2011 contains:
- **200 bird species**
- **11,788 images**
- **5,994 training images**
- **5,794 test images**

The model was trained using transfer learning in two stages:
1. Train the new classification head while the backbone is frozen.
2. Unfreeze the network and fine-tune the entire model with a lower learning rate.

The final model was evaluated on the official CUB test set.

### Results

| Metric | Result |
|---|---:|
| Test images | 5,794 |
| Species | 200 |
| Top-1 accuracy | **80.32%** |
| Top-5 accuracy | **96.34%** |

Top-5 accuracy is particularly useful for this problem because often even humans let alone machines don't always get the species right on their first guess, especially when two species are visually very similar. Using the Top-5 accuracy if the correct species is somewhere among its five strongest predictions, the system has still narrowed the field considerably.

The complete evaluation output is available in:
```text
results/evaluation_results.json
```


> **Note on these results:** this checkpoint was trained before I fixed a bug in `ml/dataset.py`. The training and validation splits shared one `ImageFolder`, so setting the validation transform also switched off augmentation for training. The model above was therefore trained **without** data augmentation. The loader now builds separate datasets for each split (covered by `tests/test_dataset.py`), and retraining with augmentation is the next step.

---

# Try it yourself

## 1. Clone the repository
```powershell
git clone https://github.com/rohanchennupati-sudo/BirDiD-bird-identifier.git
cd BirDiD-bird-identifier
```

## 2. Create a virtual environment
On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install the dependencies
```powershell
pip install -r requirements.txt
```

> **Note:** `requirements.txt` installs CPU-only PyTorch from PyTorch's package index (the file includes the `--extra-index-url` line). To train on a GPU machine such as Colab, use `requirements-training.txt` instead.

## 4. Get the trained model
The trained checkpoint is intentionally **not stored in the Git repository**.

Download `best_model_finetuned.pth` from the
[**v1.0.0 release**](https://github.com/rohanchennupati-sudo/BirDiD-bird-identifier/releases/tag/v1.0.0).

Place it here:
```text
models/
└── best_model_finetuned.pth
```
The application will detect the checkpoint automatically.

## 5. Start the server
```powershell
python -m uvicorn app.main:app --reload
```

Then open:
- **Web application:** http://127.0.0.1:8000
- **API documentation:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health/
Upload a photograph and see what the model thinks it is.

---

# Running the tests

The project includes tests for both the classifier service and the API.

Run:
```powershell
pytest tests/ -v
```

Current test suite: 13 tests (the real-model test is skipped if the checkpoint hasn't been downloaded).

The tests cover:
- mock and real classifier output (top 5 sorted, confidence band)
- health endpoint and request IDs
- successful predictions for JPEG and PNG
- rejection of unsupported types (400), corrupt images (400), oversized files (413) and invalid backend values (422)
- `backend=pytorch` without a checkpoint (503)
- the data loaders: training images are augmented, validation and test images are not, and the splits don't overlap

---

# The API
The main prediction endpoint is:

```text
POST /api/v1/predict
```

It accepts a multipart form upload:
```text
image=<your image>
```

and optionally:
```text
backend=auto
backend=pytorch
backend=mock
```

`auto` uses PyTorch when the checkpoint exists and mock otherwise. A successful response contains the predicted species, confidence level, probability, request ID, top-five predictions, and `backend_used`.

Interactive API documentation is available automatically through FastAPI at:
```text
/docs
```

---

# Training the model

The repository also contains the complete training pipeline.
The original CUB-200-2011 dataset should be placed at:

```text
CUB_200_2011/
```

Prepare the dataset:
```powershell
python -m ml.prepare_cub
```

This creates:
```text
data/
└── birds/
    ├── train/
    └── test/
```

Training is designed for a GPU environment such as Google Colab:
```powershell
python -m ml.train
```

The best checkpoint is saved as:
```text
models/best_model_finetuned.pth
```

Evaluation can then be run with:
```powershell
python -m ml.evaluate
```

which produces:
```text
results/evaluation_results.json
```

For the full Colab workflow, see:
```text
COLAB_TRAINING.md
```

---

# How the pieces fit together

One of the main goals of this project was to keep the machine-learning model separate from the application that serves it.
```text
                         ┌──────────────────┐
                         │   Web Interface  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI      │
                         │   /api/v1/...    │
                         └────────┬─────────┘
                                  │
                         image + request
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Prediction Route │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Classifier       │
                         │ Service          │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  EfficientNet    │
                         │      -B3         │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ 200 bird species │
                         └──────────────────┘
```

This separation means the API does not need to know how EfficientNet works, only how to ask the classifier for a prediction.
That distinction will be useful as I scale the project in future updates.

---

# Project structure
```text
BirDiD-bird-identifier/
│
├── app/
│   ├── middleware/
│   │   └── rate_limiter.py
│   ├── models/
│   │   └── schemas.py
│   ├── routers/
│   │   ├── health.py
│   │   └── predict.py
│   ├── services/
│   │   ├── mock_classifier.py
│   │   └── pytorch_classifier.py
│   ├── config.py
│   └── main.py
│
├── ml/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── prepare_cub.py
│   └── train.py
│
├── results/
│   ├── evaluation_results.json
│   └── training_summary.md
│
├── scripts/
│   ├── run_dev.ps1
│   └── test.ps1
│
├── static/
│   └── index.html
│
├── tests/
│   ├── conftest.py
│   ├── test_classifier.py
│   ├── test_dataset.py
│   └── test_predict.py
│
├── .env.example
├── COLAB_TRAINING.md
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── LICENSE
├── requirements.txt
├── requirements-training.txt
└── testconfig.py
```

The dataset, environment files, Python virtual environment, and trained checkpoint are deliberately excluded from normal Git history.

---

# Docker

The application can also be run in a container.
Build:

```powershell
docker build -t birdid .
```

Run:
```powershell
docker run -p 8000:8000 --env-file .env birdid
```

The trained model still needs to be supplied separately.

---

# A few honest limitations

BirDiD currently recognizes the **200 species represented in CUB-200-2011** and thus it is not a general-purpose bird detector. If you give it a species it has never been trained on, the model can still produce one of its 200 classes and does not currently have a reliable mechanism for saying: > "I don't know."

Similarly, `not_a_bird` is currently part of the API response structure rather than a separately trained bird/non-bird detection model.

The current API also does not populate scientific names from an external taxonomy service.

These are places for the project to grow in the updates to come.

---

# Where this could go

Some of the directions I'd like to explore are:
- a proper bird/non-bird rejection model
- a larger and more diverse bird dataset
- scientific-name and taxonomy integration
- mobile-friendly inference
- model optimization for deployment
- better handling of visually similar species

---

## License

MIT License. See [LICENSE](LICENSE).
