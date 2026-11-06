# LOD: live object detection (YOLOv8)

**This is a November 2026 rebuild** of work I did as a remote Machine Learning Trainee at Technofly (May–Jul 2025). The original code isn't available, so I'm rewriting the same approach from scratch. Nothing here is Technofly's code, and none of its results are Technofly's numbers.

## What works now
`detect.py` opens the laptop webcam, runs a pretrained YOLOv8n model on every frame, and draws a box around each detected object with its label and confidence score (for example `person 0.8770`). Press `q` to quit.

The model is pretrained: Ultralytics' YOLOv8n, trained on the COCO dataset (80 everyday classes). This repo does inference only so far; it doesn't train anything yet.

Tested on 5 Nov 2026 on Ubuntu with a Lenovo LOQ laptop's built-in webcam.

## Run it
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python detect.py        # first run downloads yolov8n.pt (~6.5 MB)
```

## Still to build
- [ ] Tkinter GUI: start/stop, the webcam feed inside the window, a confidence-threshold slider
- [ ] Scrape and clean a labelled people-vs-objects image set, then train an SVM classifier with a confidence score
- [ ] Annotate a small set of my own images and fine-tune YOLO on them
- [ ] Dockerfile and a GitHub Actions workflow that runs the tests

## How this was built
I typed the code myself. Claude (an AI assistant) taught each step and reviewed my attempts.
