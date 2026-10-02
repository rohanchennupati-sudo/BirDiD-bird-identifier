# Training summary

## v1.1 (current model)

Trained on a Google Colab Tesla T4 with the fixed data loader, so training images are augmented (random crop, horizontal flip, ±15° rotation, colour jitter) and validation/test images are not.

### Phase 1: classifier head only

- Up to 10 epochs, EfficientNet-B3 backbone frozen
- Adam, learning rate `1e-3`, weight decay `1e-4`, cosine annealing
- Ran all 10 epochs (about 88 s each)
- Best validation accuracy: `60.3%` (epoch 7)

### Phase 2: whole network fine-tuned

- Up to 25 epochs, Adam, learning rate `5e-5`, weight decay `1e-4`, cosine annealing
- Early stopping with patience 5: stopped after epoch 12 (about 108 s each)
- Best validation accuracy: `79.53%` (epoch 7), saved as `best_model_finetuned.pth`
- At the last epoch, training loss was 0.23 against validation loss 0.75, so the model still overfits

### Held-out test set

- Test images: `5,794`
- Top-1 accuracy: `79.36%`
- Top-5 accuracy: `95.93%`
- Weakest classes: Common Tern, Herring Gull, Western Wood Pewee (33.3% each)
- Most common confusion: Acadian Flycatcher → Least Flycatcher (10 times)

## v1.0 (previous model)

Same settings, but trained before the data-loader bug was fixed: the training and validation splits shared one dataset object, so augmentation was switched off for training too.

| | v1.0 (no augmentation) | v1.1 (augmentation) |
|---|---:|---:|
| Phase 1 best validation accuracy | 59.1% | 60.3% |
| Phase 2 best validation accuracy | 79.56% | 79.53% |
| Test top-1 | 80.32% | 79.36% |
| Test top-5 | 96.34% | 95.93% |
| Weakest class | Common Tern, 13.3% | Common Tern, 33.3% |

## What this shows

- Turning augmentation on didn't change validation accuracy (79.56% vs 79.53%). The 1-point test difference comes from a single run each, with no fixed seed for weight initialisation or shuffling, and a test set of 5,794 images (standard error about 0.5 points), so it isn't a clear difference.
- The errors are spread more evenly: the weakest class improved from 13.3% to 33.3%.
- v1.1 is published because it is what the current code produces, so the results above can be reproduced.
- Next experiment: colour is one of the main cues for telling these species apart, so the hue and saturation jitter may hurt. Retrain with only cropping, flipping and rotation, and compare on validation accuracy.
