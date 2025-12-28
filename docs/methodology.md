# Methodology (Zuma CV Bot — No ML)

## 1) Problem Summary
We need a Python system that observes the Zuma game in real time, detects key objects (frog + balls), decides an optimal shot direction using rule-based strategies, and executes the shot via mouse control. The solution must rely on classical computer vision (no ML/DL).

## 2) System Pipeline (High Level)
1. **Screen Capture + Game ROI Detection**
2. **Frog Detection (shoot origin)**
3. **Ball Detection + Color Classification**
4. **Path Modeling + Ordering (near end vs far end)**
5. **Decision Engine (scoring + chain reaction simulation)**
6. **Angle/Direction Computation**
7. **Mouse Click Execution**
8. **Monitoring + Re-detection fallback**

## 3) Module Responsibilities (Team Work)
- **Member 1 (capture/)**: capture frames and detect the game window/ROI automatically. Output: `roi_frame`, `roi_bbox`.
- **Member 2 Aya (frog/)**: detect frog and compute `frog_center` (stable across frames). Optional: detect current ball color.
- **Member 3 (balls/)**: detect balls (centers/radii) and classify their colors in HSV, with temporal filtering to reduce false detections.
- **Member 4 (path/)**: build a path model and compute each ball’s progress `s` along the track (danger estimation and ordering).
- **Member 5 (decision/)**: rule-based decision engine that evaluates candidate targets, simulates removals/chain reactions, and selects the best shot.

## 4) Key CV Techniques Used
- **HSV thresholding** for robust color separation under varying brightness.
- **Morphological operations** (open/close) to remove noise and fill gaps.
- **Contour analysis / Hough circles** for shape-based detection.
- **Temporal consistency** (simple tracking / frame-to-frame filtering) to avoid transient effects.
- **Geometric reasoning** for ordering balls on a curved path.

## 5) Strategy / Scoring (Decision Engine)
We evaluate multiple candidate targets and assign a score based on:
- **Points**: prefer shots that create a 3+ match, larger removals, and potential chain reactions.
- **Safety**: prioritize reducing danger when the chain is close to the end.
- **Ease**: prefer shots with simpler geometry (less likely to miss / less occlusion).
- **Risk reduction**: avoid inserting into mixed-color zones that reduce match probability.

The selected target is the one with maximum score, subject to a minimum confidence threshold (do not shoot if uncertain).

## 6) Performance Considerations
- Process only the **ROI** (not the full screen).
- Avoid full re-detection every frame: track stable objects (frog center) when possible.
- Re-detect ROI/frog when confidence drops.
- Keep debug overlays optional to maintain FPS.

## 7) Testing Plan
- Use a small dataset of screen recordings and screenshots to tune thresholds.
- Validate:
  - frog center stability
  - ball detection correctness
  - color classification reliability
  - decision logic correctness on known scenarios
- Then move to live mode with logging (FPS, latency, shot decisions).
