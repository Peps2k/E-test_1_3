"""
- Mouse: Left-drag rotate, Shift+Left roll, Right-drag move, Wheel zoom
- Press 'O' to toggle Fill/Wire
- Press 'M' (or 'm') to start/stop recording to overlay_record.mp4
- Press 'P' to save a screenshot of the *overlayed* frame to ./screens/
- Press Esc to quit

Run:
  python app.py --video video001_kort.mp4 --obj galleblære_segmentert.obj 
"""
import argparse, math, os, cv2, numpy as np
from dataclasses import dataclass

def load_obj(path):
    verts, faces = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'): continue
            p = line.strip().split()
            if not p: continue
            if p[0]=='v' and len(p)>=4:
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif p[0]=='f' and len(p)>=4:
                def idx(tok): return int(tok.split('/')[0]) - 1
                ids = [idx(t) for t in p[1:]]
                for i in range(1, len(ids)-1):
                    faces.append([ids[0], ids[i], ids[i+1]])
    V = np.array(verts, np.float32)
    F = np.array(faces, np.int32)
    if V.size:
        c = V.mean(axis=0, keepdims=True)
        V = V - c
        s = np.max(np.linalg.norm(V, axis=1))
        if s>0: V /= s
    return V, F

def euler_to_matrix(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cr, -sr, 0],[sr, cr, 0],[0,0,1]], np.float32)
    Rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]], np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], np.float32)
    return Ry @ Rx @ Rz

def project(P, f, cx, cy, z_near=0.1):
    Z = np.clip(P[:,2], z_near, None)
    x = (f * P[:,0] / Z) + cx
    y = (f * P[:,1] / Z) + cy
    return np.stack([x,y], axis=1)

def overlay_mesh_fast(frame, V, F, pose, mode_fill=True, alpha=0.35, max_tris=8000):
    h, w = frame.shape[:2]
    yaw,pitch,roll,sx,sy,sz,tx,ty,tz,f = pose
    R = euler_to_matrix(yaw,pitch,roll).astype(np.float32)
    S = np.diag([sx,sy,sz]).astype(np.float32)
    T = np.array([tx,ty,tz], np.float32)
    PV = (V @ (S @ R).T) + T
    pts = project(PV, f=f, cx=w/2, cy=h/2)
    out = frame.copy()

    #triangle count
    if max_tris is not None and len(F) > max_tris:
        step = max(1, len(F)//max_tris)
        useF = F[::step]
    else:
        useF = F

    if mode_fill:
        overlay = np.zeros_like(out)
        depths = PV[useF].mean(axis=1)[:,2]
        order = np.argsort(-depths)
        tris = useF[order]
        for tri_idx in tris:
            tri = pts[tri_idx].astype(np.int32)
            cv2.fillConvexPoly(overlay, tri, (0,255,0))
        out = cv2.addWeighted(out, 1.0, overlay, alpha, 0.0)
    else:
        for tri_idx in useF:
            tri = pts[tri_idx].astype(np.int32)
            cv2.polylines(out, [tri], True, (0,255,0), 1, cv2.LINE_AA)
    return out

@dataclass
class State:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    sz: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 3.0
    f: float  = 800.0
    fill: bool = False
    paused: bool = False
    max_tris: int = 8000

@dataclass
class Mouse:
    lx: int = -1
    ly: int = -1
    left: bool = False
    right: bool = False
    shift: bool = False
    h: int = 0
    w: int = 0

def make_mouse_cb(state: State, mouse: Mouse):
    rot_sens = 0.01
    mov_sens = 0.003
    roll_sens = 0.01
    zoom_step = 0.15

    def cb(event, x, y, flags, userdata):
        mouse.shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse.left = True; mouse.lx, mouse.ly = x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse.right = True; mouse.lx, mouse.ly = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            mouse.left = False
        elif event == cv2.EVENT_RBUTTONUP:
            mouse.right = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse.lx>=0 and mouse.ly>=0:
                dx = x - mouse.lx
                dy = y - mouse.ly
                if mouse.left and not mouse.shift:
                    state.yaw   += rot_sens * dx
                    state.pitch += rot_sens * dy
                elif mouse.left and mouse.shift:
                    state.roll  += roll_sens * dx
                elif mouse.right:
                    state.tx += mov_sens * dx
                    state.ty += mov_sens * dy
                mouse.lx, mouse.ly = x, y
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Fallback: getMouseWheelDelta
            delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, 'getMouseWheelDelta') else (1 if flags>0 else -1)
            state.tz = max(0.2, state.tz - zoom_step * (1 if delta>0 else -1))
    return cb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--obj', required=True)
    ap.add_argument('--alpha', type=float, default=0.35)
    ap.add_argument('--tris', type=int, default=8000)
    args = ap.parse_args()

    V, F = load_obj(args.obj)
    if V.size==0 or F.size==0:
        print('Failed to load mesh.')
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print('Failed to open video.')
        return

    # Recording setup
    rec = False
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st = State(max_tris=args.tris)
    os.makedirs('screens', exist_ok=True)

    win = 'Endoscopy+OBJ Overlay (mouse+record)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Prime first frame for mouse size and loop
    ok, frame = cap.read()
    if not ok:
        print('Could not read first frame.')
        return
    mouse = Mouse(h=frame.shape[0], w=frame.shape[1])
    cv2.setMouseCallback(win, make_mouse_cb(st, mouse))
    frame_idx = 1

    while True:
        if not st.paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

        pose = (st.yaw, st.pitch, st.roll, st.sx, st.sy, st.sz, st.tx, st.ty, st.tz, st.f)
        out = overlay_mesh_fast(frame, V, F, pose, mode_fill=st.fill, alpha=args.alpha, max_tris=st.max_tris)

        # HUD + REC indicator
        hud1 = f'MOUSE L: rotate | Shift+L: roll | R: move | Wheel: zoom | {"FILL" if st.fill else "WIRE"} | tris<={st.max_tris}'
        cv2.putText(out, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        if rec:
            cv2.putText(out, 'REC \u25CF', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

        # Write to file if recording
        if rec and writer is not None:
            writer.write(out)

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF

        # Handle both 'm' and 'M' on Windows
        if key in (ord('m'), ord('M')):
            if not rec:
                writer = cv2.VideoWriter('overlay_record.mp4', fourcc, fps, (w, h))
                rec = True
                print('REC ON -> overlay_record.mp4')
            else:
                if writer is not None:
                    writer.release()
                    writer = None
                rec = False
                print('REC OFF')
        elif key == 27:   # ESC
            break
        elif key == ord(' '): st.paused = not st.paused
        elif key == ord('r'): st.sx*=1.05; st.sy*=1.05; st.sz*=1.05
        elif key == ord('f'): st.sx/=1.05; st.sy/=1.05; st.sz/=1.05
        elif key == ord('w'): st.ty -= 0.03
        elif key == ord('s'): st.ty += 0.03
        elif key == ord('a'): st.tx -= 0.03
        elif key == ord('d'): st.tx += 0.03
        elif key == ord('z'): st.tz = max(0.2, st.tz - 0.1)
        elif key == ord('x'): st.tz += 0.1
        elif key == ord('q'): st.yaw -= 0.03
        elif key == ord('e'): st.yaw += 0.03
        elif key == ord('t'): st.pitch -= 0.03
        elif key == ord('g'): st.pitch += 0.03
        elif key == ord('c'): st.roll -= 0.03
        elif key == ord('v'): st.roll += 0.03
        elif key == ord('o'): st.fill = not st.fill
        elif key == ord('1'): st.max_tris = 4000
        elif key == ord('2'): st.max_tris = 8000
        elif key == ord('3'): st.max_tris = 16000
        elif key == ord('p'):
            fn = os.path.join('screens', f'frame_{frame_idx:06d}.png')
            cv2.imwrite(fn, out) 
            print('Saved screenshot ->', fn)

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
