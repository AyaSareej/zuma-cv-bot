# Zuma CV Bot (Computer Vision Only)

A Windows + Python + OpenCV project skeleton for an auto-play Zuma assignment using **classical computer vision only** (no ML/DL).

## Team Structure (who owns what)
- **Member 1** → `src/capture/` (screen capture + auto ROI/window detection)
- **Member 2** → `src/frog/` (frog detection + shoot origin)
- **Member 3** → `src/balls/` (ball detection + color classification)
- **Member 4** → `src/path/` (path model + ball ordering along the curve)
- **Member 5** → `src/decision/` (decision engine + shot simulation + scoring)
- **Integration / runs** → `src/main/`

## Repository Structure
assets/
screenshots/ # test images (png/jpg)
templates/ # optional templates (frog, UI elements)
docs/
methodology.md
interview_notes.md
src/
capture/
frog/
balls/
path/
decision/
main/


## Setup (Windows)
### 1) Create & activate venv
```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
