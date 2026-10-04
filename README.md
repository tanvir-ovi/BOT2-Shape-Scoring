<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">Automated Scoring of BOT-2 Shape Drawings under Child-Grouped Cross-Validation</h2>

<p align="center">
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-green.svg'></a>
</p>

<br/>

</div>

This repository scores the eight shape-drawing items of the Bruininks-Oseretsky Test of Motor Proficiency, Second
Edition (BOT-2) with three paradigms: rules that apply the BOT-2 rubric to OpenCV measurements, machine-learning
classifiers on 64 OpenCV geometric features, and deep networks. Every scorer is evaluated with the same five-fold
cross-validation, in which all drawings of a child stay in one fold. The proposed scorer, DPE, averages the score
probabilities of three networks pretrained on different image sources: supervised natural images at 512 pixels,
self-supervised natural images, and self-supervised scanned documents.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Scorers</h2>

| Paradigm | Scorer | Input and pretraining |
|---|---|---|
| Deep learning, proposed | **DPE**, mean of the score probabilities of its three members | |
| Deep learning, member | ConvNeXt V2-T | 512 px; FCMAE, then ImageNet-22k and ImageNet-1k |
| Deep learning, member | DINOv2 ViT-L/14 with registers | 336 px; self-supervised, LVD-142M |
| Deep learning, member | DiT-B | 224 px; self-supervised, 42 million scanned document pages |
| Deep learning, baseline | ResNet-50 | 384 px; ImageNet-1k |
| Deep learning, baseline | MobileNetV3-Large | 384 px; ImageNet-1k |
| Deep learning, baseline | EfficientNetV2-S | 384 px; ImageNet-21k, then ImageNet-1k |
| Deep learning, baseline | ViT-B/16 | 224 px; ImageNet-21k, then ImageNet-1k |
| Deep learning, baseline | CViT, Xception and ViT-B/16 features concatenated | 299 and 224 px; ImageNet |
| Deep learning, control | BEiT-B | 224 px; ImageNet-22k |
| Deep learning, control | BEiT-B | 224 px; ImageNet-22k, then QuickDraw sketches |
| Deep learning, control | ConvNeXt V2-T | 224 px; as the member |
| Machine learning | Gradient boosting, random forest, SVM with an RBF kernel, logistic regression | 64 OpenCV features |
| Rule-based | OpenCV rubric rules | OpenCV measurements |
| Reference | Most frequent training score of each item | |

**Networks.** All networks share one head (layer normalisation and one linear score head per item) and one recipe:
cross-entropy with label smoothing of 0.05, natural sampling, AdamW with linear warm-up and cosine decay, an
exponential moving average of the weights, and early stopping on validation accuracy. A run whose validation predictions
collapse to the most frequent score of every item is restarted at half the learning rate, at most twice. The predicted
score is the most probable valid score of the item; no threshold or prior correction is tuned. `models.recipe()` lists
every setting.

**OpenCV features.** The pencil stroke is separated from the scanned page, and 64 measurements are taken from it:
size and position, enclosed regions and overlap, symmetry, contour shape, angles and corners, points, crossings,
closure (overshoot tails, gaps between free stroke ends) and stroke quality. `features.table()` describes each one.

**Machine-learning classifiers.** One model per item, fitted on the training part of each fold with hyperparameters
fixed in advance. Each predicts the most probable score.

**Rubric rules.** Each criterion of each item (basic shape, closure, edges, orientation, overlap, overall size) is
judged by one OpenCV measurement against a cut-off. The item score is the number of criteria passed, or 0 when the
basic shape fails, as the rubric prescribes. Each cut-off is set on the training drawings of the fold, where it agrees
best (Cohen's kappa) with the examiner's mark for that criterion, read from the file name.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Protocol</h2>

- 4,296 drawings of 664 children inferred from the image numbering. Byte-identical files with conflicting scores are
  excluded.
- Five folds, stratified by item and score and grouped by inferred child. In each fold, the test part is the held-out
  fold; one eighth of the remaining children form the validation part and the rest the training part.
- The training part fits every scorer. The validation part only selects the checkpoint of a network and, for the
  comparison of paradigms, the network and the classifier with the best validation accuracy in that fold. The test part
  is scored once, after training.
- Every drawing is therefore scored once, by scorers that never saw its child. All reported numbers are these
  out-of-fold test predictions. One training seed (2026) is used for every network.
- Accuracy is the mean of the eight item accuracies. Agreement with the expert is reported as Cohen's kappa and
  quadratic weighted kappa on the full score scale of each item, with the Landis and Koch bands.
- Paired comparisons use the same drawings for both scorers. Intervals come from 10,000 bootstrap resamples of
  children, and p values from a child-level sign-flip permutation test. The Holm correction is applied within each
  family of comparisons (paradigms, the proposed ensemble against every scorer, the ensemble without one member,
  pretraining source and input size), and the equivalence margin is one accuracy point.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Environment setup</h2>

The notebook runs on Google Colab with an A100 runtime and installs what Colab lacks:

```bash
pip install -U timm "transformers>=4.56" safetensors
```

For a local run:

```bash
pip install -r requirements.txt
```

The exact package versions, the GPU and the SHA-256 checksums of the code are written to `results/logs` at the
start of every run.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Running</h2>

1. Put `Shapes.zip` in a Google Drive folder, and copy this repository into a subfolder named `final` next to it.
   `DRIVE_DIR` in the setup cell is the folder that holds `Shapes.zip`.
2. Open `BOT2_5Fold_CV.ipynb` in Colab with an A100 runtime and run all cells. The 55 network-fold units take about
   six hours. The training cell prints the training loss, training accuracy, validation accuracy and validation QWK of
   every epoch, and the test accuracy of every fold.
3. If the runtime stops, run all cells again: finished units are skipped, and their stored epoch logs are printed
   in their place. The runtime is released after the last cell and after any error.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Outputs</h2>

| Folder | Content |
|---|---|
| `results/data` | index of the drawings with their fold, excluded files, score and criterion tables, OpenCV features, input cache |
| `results/runs` | per scorer and fold: validation and test probabilities; for networks also test embeddings, the epoch log and the weights of the ensemble members |
| `results/predictions` | out-of-fold predicted score and probabilities of every scorer for every drawing |
| `results/metrics` | summary, agreement, accuracy per item and per fold, metrics per item, confusion matrices, rule cut-offs and criterion agreement, training summary, paradigm selection, precision-recall data, embedding separation, weight verification |
| `results/statistics` | paired comparisons with intervals, permutation and McNemar p values, Holm-adjusted p values and verdicts |
| `results/embeddings` | out-of-fold embeddings of every network and t-SNE coordinates |
| `results/figures` | figures as vector PDF at IEEE column or text width |
| `results/logs` | run log, software versions and code checksums |
| `results/manifest_sha256.csv` | SHA-256 checksum of every result file |

`verify.check_weights` rebuilds every saved network from its weight file, predicts its test fold again and compares
the probabilities with the stored ones. Linear and convolution weights are stored in bfloat16, the precision in which
they were used on the GPU, so an A100 reproduces the stored predictions.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Data availability</h2>

The drawings are not distributed with this repository.
