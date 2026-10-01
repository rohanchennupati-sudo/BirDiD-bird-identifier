# Training summary

Training was run on a Google Colab Tesla T4.

## Phase 1

- 10 epochs
- EfficientNet-B3 backbone frozen
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Best validation accuracy: `59.1%`

## Phase 2

- 25 epochs completed
- Entire network fine-tuned
- Learning rate: `5e-5`
- Weight decay: `1e-4`
- Early stopping patience: 5
- Best validation accuracy: `79.56%`

## Held-out test set

- Test images: `5,794`
- Top-1 accuracy: `80.32%`
- Top-5 accuracy: `96.34%`
