# Small-object detection in aerial imagery (MAVREC)

Final-year project, IIIT Kottayam. We study what actually improves object detection on drone footage, where most
objects are tiny, using the [MAVREC](https://mavrec.github.io/) aerial dataset (10 classes, 2704×1520 frames,
~80 objects per image).

Every experiment follows a fixed, pre-registered plan: screen detector families, pick a resolution, measure data
scaling, test tiling and ground-view pretraining, then combine only the changes that helped.

## Final result

**YOLOv8m, COCO-pretrained, trained on 2000 aerial frames at 1280 px, full-frame inference.**

| Split | mAP50-95 | mAP50 | AP small | Recall |
|---|---|---|---|---|
| Val  | 70.8 | 92.7 | 43.5 | – |
| Test | 68.6 | 91.2 | 43.8 | 88.6 |

Runs at about 30 FPS on a Colab T4 (25.9 M parameters). Recall on the test set by object size: tiny 70%,
small 85%, medium 95%, large 98%.

### How we got there (test set, cumulative)

| Step | mAP50-95 | AP small |
|---|---|---|
| A. YOLOv7, 500 frames, 640 px (Review 2 baseline) | 21.9 | 6.8 |
| B. + resolution 1280 | 35.5 | 25.7 |
| C. + architecture: YOLOv8m | 59.2 | 34.4 |
| D. + data: 2000 frames | **68.6** | **43.8** |

## What we found

- **Detector choice** (1000 frames, 960 px, val mAP50-95): YOLOv8m 61.2, YOLO11m 59.2, Faster R-CNN v2 55.5,
  RT-DETR-L 52.9, Faster R-CNN v1 46.8, YOLOv7 45.8, Deformable DETR 19.6.
- **Resolution matters most for small objects.** YOLOv8m val mAP50-95: 640 px 50.6, 960 px 61.2, 1280 px 66.9
  (AP small 16.1 → 29.5 → 38.4).
- **More data keeps helping.** 500 / 1000 / 2000 frames: 63.1 / 66.9 / 70.8 val mAP50-95, roughly +3.9 per doubling.
- **Tiling was not worth it.** Training and inferring on 640-px tiles raised AP small by 8.4 but cost 9.0 mAP
  overall and ran at ~3 FPS, so it was dropped.
- **Ground-view pretraining did not help.** Pretraining on MAVREC's ground-level images before aerial fine-tuning
  scored 2.4 mAP lower than going straight from COCO, and 5.5 lower than spending the same compute on longer
  aerial training. A model trained only on ground views scores 8.0 mAP on aerial val (Deformable DETR: 1.5).
- **Transfer to VisDrone is weak.** The final model scores 6.0 mAP50-95 on VisDrone zero-shot. Fine-tuning on 1000
  VisDrone images gives 28.1 from COCO weights and 26.4 from our MAVREC weights.

### Extra experiments (after the main plan)

- **Test-time augmentation** (notebook 11): Ultralytics' built-in TTA lowered test mAP to 67.5. Flip + upscaled
  (1280 + 1600 px) TTA reached 73.1 on val (+2.3) at ~143 ms per image; not yet evaluated on test.
- **Deformable DETR with ground pretraining** (notebook 07): ground stage complete; aerial fine-tune incomplete.
- **P2 detection head** (notebook 11): not trained.

### Limitations

- All runs use a single seed (42), so differences of about 1 mAP should not be over-read.
- MAVREC's official val and test splits come from the same 12 video sequences as training, often only a few
  frames apart, so absolute scores are likely optimistic compared with unseen locations.

## Repository layout

```
notebooks/   00–11 Colab notebooks, run in order (00 prepares data and writes the shared library)
src/         mavrec_lib.py (shared code used by every notebook), run_full.py (headless runner)
results/
  tables/      every results table as CSV + Markdown (master_results.csv has all experiments)
  figures/     plots and qualitative examples used in the report
  records/     one JSON per experiment: config, training time, all metrics
  decisions.json  each design decision, the rule used and the numbers behind it
```

Model weights, predictions and the dataset are not included because of their size.

## Running it

The notebooks are written for Google Colab (T4 GPU) with Google Drive mounted.

1. Download MAVREC from the [official site](https://mavrec.github.io/) and place it in your Drive under `MyDrive/MAVREC/`.
2. Run `00_Setup_and_Data_Preparation` once. It builds the train/val/test splits, image caches and
   `mavrec_lib.py`.
3. Run notebooks `01`–`10` in order. Each one skips experiments that already have a record and resumes training
   from the last checkpoint if Colab disconnects.

## Acknowledgements

MAVREC dataset: Dutta et al., *Multiview Aerial Visual RECognition (MAVREC): Can Multi-view Improve Aerial Visual
Perception?*, CVPR 2024. VisDrone dataset for the transfer experiments. Detectors from Ultralytics, torchvision and
Hugging Face Transformers.
