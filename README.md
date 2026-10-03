<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">Automated Scoring of BOT-2 Shape Drawings under Child-Grouped Cross-Validation</h2>

<p align="center">
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-green.svg'></a>
</p>

<br/>

</div>

This repository scores the eight shape-drawing items of the Bruininks-Oseretsky Test of Motor Proficiency, Second
Edition (BOT-2) with image networks and feature-based scorers, and evaluates every scorer with the same five-fold
cross-validation, in which all drawings of a child stay in one fold. The proposed scorer, DPE, averages the score
probabilities of three backbones pretrained on different image sources: supervised natural images at 512 pixels,
self-supervised natural images, and self-supervised scanned documents.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Scorers</h2>

| Role | Scorer | Pretraining | Input |
|---|---|---|---|
| Proposed | **DPE**, mean of the score probabilities of the three members | | |
| Member | ConvNeXt V2-T | FCMAE, then ImageNet-22k and ImageNet-1k | 512 px |
| Member | DINOv2 ViT-L/14 with registers | self-supervised, LVD-142M | 336 px |
| Member | DiT-B | self-supervised, 42M scanned document pages | 224 px |
| Baseline | ResNet-50 | ImageNet-1k | 384 px |
| Baseline | EfficientNetV2-S | ImageNet-21k and ImageNet-1k | 384 px |
| Baseline | ViT-B/16 | ImageNet-21k and ImageNet-1k | 224 px |
| Baseline | CViT, Xception and ViT-B/16 features concatenated | ImageNet | 299 and 224 px |
| Control | BEiT-B | ImageNet-22k | 224 px |
| Control | BEiT-B | ImageNet-22k, then QuickDraw sketches | 224 px |
| Control | ConvNeXt V2-T | as the member | 224 px |
| Feature-based | Most frequent training score of each item | | |
| Feature-based | Gradient boosting on 64 geometric descriptors | | |

All networks share one head (layer normalisation and one linear score head per item) and one recipe: cross-entropy
with label smoothing of 0.05, natural sampling, AdamW with linear warm-up and cosine decay, an exponential moving
average of the weights, and early stopping on validation accuracy. A run whose validation predictions collapse to the
most frequent score is restarted at half the learning rate. The predicted score is the most probable valid score of
the item, and no threshold or prior correction is tuned.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Protocol</h2>

- 4,296 drawings of 664 children inferred from the image numbering. Byte-identical files with conflicting scores are
  excluded.
- Five folds, stratified by item and score and grouped by inferred child. Within each fold, one eighth of the training
  children form the validation set, which is used only to select the checkpoint.
- One training seed (2026). Every network uses the same folds, seed and schedule.
- Each drawing is scored once, by the models of the fold that held it out.
- Paired comparisons use the same drawings for both scorers. Intervals come from 10,000 bootstrap resamples of
  children, and p values from a child-level sign-flip permutation test. The Holm correction is applied within each
  family of comparisons, and the equivalence margin is one accuracy point.


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
2. Open `BOT2_5Fold_CV.ipynb` in Colab with an A100 runtime and run all cells. `DRIVE_DIR` in the setup cell is the
   folder that holds `Shapes.zip`.
3. One runtime trains the 50 network-fold units in about five hours. To use three runtimes at once, open the notebook
   in three tabs with `N_WORKERS = 3` and `WORKER = 0`, `1` and `2`. Worker 0 waits for the others, takes over the
   units of a worker that stops, and runs the analysis.
4. Finished units are skipped when the notebook is run again, so an interrupted run resumes where it stopped. The
   runtime is released at the end of the run and after any error.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Outputs</h2>

| Folder | Content |
|---|---|
| `results/data` | index of the drawings with their fold, excluded files, input cache, geometric descriptors |
| `results/runs` | per network and fold: validation and test probabilities, test embeddings, training history, weights |
| `results/predictions` | out-of-fold predicted score and probabilities of every scorer for every drawing |
| `results/metrics` | summary, accuracy per item and per fold, precision, recall and F1 per item, confusion matrices, precision-recall data, embedding separation, weight verification |
| `results/statistics` | paired comparisons with intervals, permutation and McNemar p values, Holm-adjusted p values and verdicts |
| `results/embeddings` | out-of-fold embeddings of every network and t-SNE coordinates |
| `results/figures` | figures as vector PDF at IEEE column or text width |
| `results/logs` | run logs, software versions and code checksums |
| `results/manifest_sha256.csv` | SHA-256 checksum of every result file |

`verify.check_weights` rebuilds every saved network from its weight file, predicts its test fold again and compares
the probabilities with the stored ones. Linear and convolution weights are stored in bfloat16, the precision in which
they were used on the GPU, so an A100 reproduces the stored predictions.


<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Data availability</h2>

The drawings are not distributed with this repository.
