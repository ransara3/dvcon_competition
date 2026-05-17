# DVCon India 2026 – Stage 2A Submission
## Task-Driven Object Detection using OWL-ViT v2

---

## Quick Start — Run on Your Own Image

```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Run on a single image with all 14 tasks
python src/main.py \
  --image path/to/your/photo.jpg \
  --queries-file data/queries.json \
  --output results/my_result.jpg
```

---

## The 14 COCO-Tasks (Official)

| # | Task | What the detector looks for |
|---|------|-----------------------------|
| 1 | step on something | stool, chair, box, ladder |
| 2 | sit comfortably | chair, sofa, bench, couch |
| 3 | place flowers | vase, cup, bottle, bowl |
| 4 | get potatoes out of fire | fork, tongs, kitchen utensil |
| 5 | water plant | bottle, cup, bowl, jug |
| 6 | get lemon | spoon or fork |
| 7 | dig hole | shovel, spade, fork |
| 8 | open beer | knife, fork, bottle opener |
| 9 | open parcel | knife or scissors |
| 10 | serve wine | wine glass or drinking glass |
| 11 | pour sugar | spoon or teaspoon |
| 12 | smear butter | butter knife or spatula |
| 13 | extinguish fire | bottle, bucket, bowl, cup |
| 14 | pound carpet | bat, racket, broom, stick |

Full task descriptions (original COCO-Tasks paper wording) are in `data/tasks_full.json`.

---

## Folder Structure

```
DVCon2026_Stage2A_Submission/
├── src/
│   ├── detector.py       ← Core OWL-ViT v2 detection engine
│   ├── main.py           ← CLI: run on any image
│   ├── evaluate.py       ← Batch evaluation over a dataset
│   ├── visualize.py      ← Draw bounding boxes on images
│   ├── demo.py           ← Matplotlib interactive demo
│   └── __init__.py
├── data/
│   ├── queries.json      ← 14 task queries (edit to customise)
│   └── tasks_full.json   ← Full COCO-Tasks descriptions + typical objects
├── results/
│   ├── summary.txt              ← Per-task detection statistics
│   ├── evaluation_report.json   ← Full JSON results
│   ├── report_grid.jpg          ← 4×4 grid of annotated images
│   ├── demo_video.mp4           ← Demo video (all 20 images)
│   └── annotated/               ← 20 annotated JPGs + JSON sidecars
│       ├── coco_*.jpg
│       └── coco_*.json
├── report/
│   ├── DVCon2026_Stage2A_Report_FINAL.md
│   └── report_grid.jpg
├── run_all.py            ← One-click: evaluate → report → video
├── requirements.txt
└── README_SUBMISSION.md  ← This file
```

---

## Model

- **Model**: `google/owlv2-base-patch16-ensemble` (OWL-ViT v2, ~620 MB)
- **Source**: HuggingFace Transformers — auto-downloaded on first run
- **Approach**: Zero-shot open-vocabulary detection — no fine-tuning required

## Requirements

```
torch>=2.0.0
torchvision
transformers>=4.37.0
Pillow
opencv-python
matplotlib
tqdm
numpy
```

---

## Reference

Task definitions from:  
*"What Object Should I Use? - Task Driven Object Detection"*  
Sawatzky et al., CVPR 2019 — https://arxiv.org/abs/1904.03000  
Dataset: https://coco-tasks.github.io/
