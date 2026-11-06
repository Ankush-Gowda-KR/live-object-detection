# LOD: live object detection (YOLOv8)

**This is a November 2026 rebuild** of work I did as a remote Machine Learning Trainee at Technofly (May–Jul 2025). The original code isn't available, so I'm rewriting the same approach from scratch. Nothing here is Technofly's code, and none of its results are Technofly's numbers.

## What works now
`detect.py` opens the laptop webcam, runs a pretrained YOLOv8n model on every frame, and draws a box around each detected object with its label and confidence score (for example `person 0.8770`). Press `q` to quit.

The model is pretrained: Ultralytics' YOLOv8n, trained on the COCO dataset (80 everyday classes). This repo does inference only so far; it doesn't train anything yet.

`gui.py` does the same inside a Tkinter window, with a slider for the confidence cutoff (default 0.55). YOLO drops every box below the slider's value, live. At 0.05 it draws junk, like a "wine glass" at 0.08 on a T-shirt print; at 0.95 even a clear person at 0.87 disappears.

Tested on Ubuntu with a Lenovo LOQ laptop's built-in webcam: `detect.py` on 5 Nov 2026, `gui.py` on 6 Nov 2026.

## Run it
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python detect.py        # first run downloads yolov8n.pt (~6.5 MB)
python gui.py           # the windowed version with the confidence slider
```

## Still to build
- [x] Tkinter window with the live webcam feed and a confidence-cutoff slider (6 Nov)
- [ ] Start/Stop buttons in the window
- [ ] Scrape and clean a labelled people-vs-objects image set, then train an SVM classifier with a confidence score
- [ ] Annotate a small set of my own images and fine-tune YOLO on them
- [ ] Dockerfile and a GitHub Actions workflow that runs the tests

## How this was built
I typed the code myself. Claude (an AI assistant) taught each step, reviewed my attempts, and fixed the indentation in my first `gui.py` edit.
