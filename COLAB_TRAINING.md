# GPU training

The local application does not need the dataset to run in mock mode.

For the real 200-species classifier, use Google Colab with a GPU.

## 1. Upload the project

Zip these folders/files:

```text
app/
ml/
requirements.txt
```

Upload the zip to Colab and extract it.

## 2. Install training dependencies

```python
!pip install -q torch torchvision
```

## 3. Download CUB-200-2011

```bash
!wget -q https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
!tar -xzf CUB_200_2011.tgz
```

## 4. Prepare the dataset

```bash
!python -m ml.prepare_cub
```

## 5. Verify GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 6. Train

```bash
!python -m ml.train
```

## 7. Download

Download:

```text
models/best_model_finetuned.pth
```

Place that file into the local project's:

```text
models/
```

Then restart the FastAPI server.

The API will automatically switch from mock to PyTorch inference.
