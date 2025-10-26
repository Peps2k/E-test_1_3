# Endoscopy + 3D Gallbladder Overlay (with Mouse Controls & Recording)

This project overlays a 3D gallbladder model (`.obj`) on top of an endoscopy video.  
You can interact with the 3D model (rotate, move, zoom) using your mouse and record or take screenshots directly from the GUI.

## Requirements

```bash
python -m venv .venv

.venv/scripts/activate
```
-Install dependencies:
```bash
pip install requirements.txt
```

## Run

Run the main script:

python app_mouse_record.py --video video001_kort.mp4 --obj galleblære_segmentert.obj


Optional parameters:
- `--tris N` → limit number of triangles (default `8000` for speed)
- `--alpha X` → transparency of overlay (default `0.35`)

## Mouse Controls
| Action | Mouse |
|--------|--------|
| Rotate | Left-drag |
| Roll | Shift + Left-drag |
| Move | Right-drag |
| Zoom in/out | Mouse wheel |

## Keyboard Controls

| Key | Action |
|-----|--------|
| **M** | Start/stop recording (`overlay_record.mp4`) |
| **P** | Save screenshot of overlay (saved to `screens/frame_XXXXXX.png`) |
| **O** | Toggle wireframe/fill mode |
| **R / F** | Scale up / down |
| **W / S** | Move up / down |
| **A / D** | Move left / right |
| **Z / X** | Move closer / farther |
| **Q / E** | Rotate yaw |
| **T / G** | Rotate pitch |
| **C / V** | Rotate roll |
| **1 / 2 / 3** | Triangle caps (≈ 4k / 8k / 16k tris) |
| **Space** | Pause/resume video |
| **Esc** | Quit program |

## Outputs

- **Screenshots:**  
  Saved automatically to the `screens/` folder, including the overlaid 3D model.

- **Recordings:**  
  Saved as `overlay_record.mp4` in the same directory.
