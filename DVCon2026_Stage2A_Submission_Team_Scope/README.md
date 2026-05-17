# DVCon India 2026 – Design Contest Stage 2A
## Open-Vocabulary Object Detection via Text Prompts

---

### Project Overview

This application detects objects in images based on **natural-language text queries**.  
It uses **OWL-ViT v2** (Vision Transformer for Open-World Localization), a zero-shot
object detection model from Google, running entirely on **CPU** for inference.

Given a query such as `"a person wearing a helmet"`, the model outputs bounding boxes
drawn around matching objects in the input image(s).

---

### Folder Structure

```
stage2a/
├── src/
│   ├── main.py         ← CLI entry point (single-image & batch)
│   ├── detector.py     ← OWL-ViT v2 detector class
│   ├── visualize.py    ← Bounding-box drawing & image saving
│   ├── evaluate.py     ← Batch evaluation over all 14 queries
│   └── demo.py         ← Interactive demo with matplotlib display
├── data/
│   └── queries.json    ← ⚠ Replace with actual contest queries!
├── dataset/
│   └── images/         ← Put contest dataset images here
├── results/            ← All output images and JSON reports
├── report/             ← Two-page submission report
├── requirements.txt
└── README.md
```

---

### Setup

#### 1. Install Python dependencies

```bash
cd stage2a
pip install -r requirements.txt
```

> **Note:** The first run downloads the OWL-ViT v2 model (~700 MB) from HuggingFace
> and caches it automatically. An internet connection is required on first run only.

#### 2. Add the dataset

Place the contest-provided images inside `dataset/images/`.

#### 3. Update the queries

Edit `data/queries.json` to match the **exact 14 prompts** provided by the contest
organisers.  The file must be a JSON array:

```json
[
  {"id": 1, "query": "your first query here"},
  {"id": 2, "query": "your second query here"},
  ...
]
```

---

### Usage

#### Single image

```bash
python src/main.py \
  --image dataset/images/img001.jpg \
  --queries "a car on the road" "a traffic light" \
  --output results/img001_result.jpg
```

#### Batch mode (all images, queries from file)

```bash
python src/main.py \
  --folder dataset/images/ \
  --queries-file data/queries.json \
  --output-dir results/batch/
```

#### Full evaluation (all 14 queries × all images)

```bash
python src/evaluate.py \
  --dataset dataset/images/ \
  --queries data/queries.json \
  --output-dir results/evaluation/
```

This generates:
- `results/evaluation/annotated/*.jpg` – annotated images
- `results/evaluation/evaluation_report.json` – full per-query statistics
- `results/evaluation/summary.txt` – human-readable table

#### Interactive demo

```bash
python src/demo.py --image dataset/images/img001.jpg
```

---

### Model Details

| Property       | Value                                  |
|----------------|----------------------------------------|
| Model          | OWL-ViT v2 (owlv2-base-patch16-ensemble) |
| Source         | HuggingFace Transformers               |
| Task           | Zero-shot open-vocabulary detection    |
| Backend        | PyTorch (CPU inference)                |
| Input          | Any RGB image + list of text queries   |
| Output         | Bounding boxes with confidence scores  |

**Why OWL-ViT v2?**  
- No fine-tuning required — works directly with natural-language queries  
- Runs on CPU with no CUDA dependency  
- State-of-the-art open-vocabulary detection benchmark results  
- Simple HuggingFace API, easy to verify and reproduce  

---

### Adjusting Detection Sensitivity

Use `--threshold` to control how many detections are returned:

| Threshold | Effect                                           |
|-----------|--------------------------------------------------|
| 0.05      | Very permissive — more detections, more false positives |
| 0.10      | **Default** — balanced precision/recall          |
| 0.20      | Conservative — fewer but higher-confidence boxes |

---

### Submission Checklist

- [ ] Source code (`src/`)  
- [ ] Two-page report (`report/DVCon2026_Stage2A_Report.pdf`)  
- [ ] Results snapshots (`results/evaluation/annotated/`)  
- [ ] Short demo video  
