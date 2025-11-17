import os
import sys
import glob
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np

# Optional: if you want OpenCV centroids fast (not required strictly)
try:
    import cv2
    HAS_CV = True
except Exception:
    HAS_CV = False

# ---------------- YOLO (Ultralytics) ----------------
try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not installed. Run: pip install ultralytics")
    raise

# ================== USER SETTINGS ====================
# Default Test root with city subfolders inside:
DEFAULT_TEST_ROOT = r"E:\ECP DATASET\Train\Images"

# Which YOLOv8 model to use (you can switch to 'yolov8s.pt' for better accuracy)
YOLO_WEIGHTS = "yolov8m.pt"

# Class filter: COCO 'person' = 0
PERSON_CLASS_ID = 0

# Max display size (GUI will scale the image to fit but draw boxes correctly)
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720

# Save annotated outputs next to the original? (toggle in GUI too)
DEFAULT_SAVE_DRAWN = False

# =====================================================


def human_readable(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.1f}k" if n >= 1e3 else str(n))


class YoloViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv8 Pedestrian Viewer")
        self.geometry(f"{MAX_DISPLAY_W+320}x{MAX_DISPLAY_H+120}")  # room for controls
        self.minsize(900, 600)

        # State
        self.test_root = DEFAULT_TEST_ROOT
        self.cities = []
        self.images = []
        self.city_var = tk.StringVar()
        self.idx = 0
        self.model = None
        self.image_panel = None
        self.tkimg_cache = None  # keep reference
        self.last_result = None

        # Controls state
        self.conf_var = tk.DoubleVar(value=0.35)
        self.iou_var = tk.DoubleVar(value=0.50)
        self.save_drawn_var = tk.BooleanVar(value=DEFAULT_SAVE_DRAWN)
        self.show_centroid_var = tk.BooleanVar(value=True)
        self.model_path_var = tk.StringVar(value=YOLO_WEIGHTS)

        # Build UI
        self._build_ui()

        # Try loading model
        self._load_model(self.model_path_var.get())

        # Populate cities and load first
        self._load_test_root(self.test_root)

        # Keyboard shortcuts
        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<space>", lambda e: self.run_infer_current())

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # Folder picker
        ttk.Label(top, text="Test root:").grid(row=0, column=0, sticky="w")
        self.root_entry = ttk.Entry(top, width=60)
        self.root_entry.insert(0, self.test_root)
        self.root_entry.grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Browse", command=self.browse_root).grid(row=0, column=2, padx=4)

        # Model picker
        ttk.Label(top, text="YOLO weights:").grid(row=1, column=0, sticky="w")
        self.model_entry = ttk.Entry(top, width=60, textvariable=self.model_path_var)
        self.model_entry.grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Load Model", command=lambda: self._load_model(self.model_entry.get())).grid(row=1, column=2, padx=4)

        # City dropdown
        ttk.Label(top, text="City:").grid(row=2, column=0, sticky="w")
        self.city_combo = ttk.Combobox(top, textvariable=self.city_var, state="readonly", width=30)
        self.city_combo.grid(row=2, column=1, sticky="w", padx=4)
        self.city_combo.bind("<<ComboboxSelected>>", lambda e: self.city_changed())

        # Sliders
        sliders = ttk.Frame(top)
        sliders.grid(row=0, column=3, rowspan=3, sticky="nsw", padx=8)

        ttk.Label(sliders, text="conf").grid(row=0, column=0, sticky="w")
        conf_scale = ttk.Scale(sliders, from_=0.05, to=0.90, variable=self.conf_var, orient=tk.HORIZONTAL, length=150, command=lambda e: self.update_status())
        conf_scale.grid(row=0, column=1, padx=6)

        ttk.Label(sliders, text="IoU").grid(row=1, column=0, sticky="w")
        iou_scale = ttk.Scale(sliders, from_=0.10, to=0.90, variable=self.iou_var, orient=tk.HORIZONTAL, length=150, command=lambda e: self.update_status())
        iou_scale.grid(row=1, column=1, padx=6)

        # Toggles
        toggles = ttk.Frame(top)
        toggles.grid(row=0, column=4, rowspan=3, sticky="nsw", padx=8)
        ttk.Checkbutton(toggles, text="Save drawn", variable=self.save_drawn_var).pack(anchor="w")
        ttk.Checkbutton(toggles, text="Show centroid", variable=self.show_centroid_var).pack(anchor="w")

        # Navigation + actions
        nav = ttk.Frame(self)
        nav.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        ttk.Button(nav, text="Prev (←)", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Next (→)", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Infer (Space)", command=self.run_infer_current).pack(side=tk.LEFT, padx=8)
        ttk.Button(nav, text="Refresh List", command=self.refresh_images).pack(side=tk.LEFT, padx=2)

        # Canvas area
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", width=MAX_DISPLAY_W, height=MAX_DISPLAY_H, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

    # --------------- Data / Model ----------------
    def browse_root(self):
        path = filedialog.askdirectory(initialdir=self.test_root, title="Select Test Root (folder with city subfolders)")
        if path:
            self._load_test_root(path)

    def _load_test_root(self, path):
        if not os.path.isdir(path):
            messagebox.showerror("Error", f"Folder not found:\n{path}")
            return
        self.test_root = path
        self.root_entry.delete(0, tk.END)
        self.root_entry.insert(0, path)
        # find cities (directories)
        self.cities = [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
        self.city_combo["values"] = self.cities
        if self.cities:
            # auto-select 'amsterdam' if present
            default_city = "amsterdam" if "amsterdam" in self.cities else self.cities[0]
            self.city_var.set(default_city)
            self.city_changed()
        else:
            self.images = []
            self.city_var.set("")
            self.clear_canvas()
        self.update_status()

    def city_changed(self):
        city = self.city_var.get()
        if not city:
            return
        city_dir = os.path.join(self.test_root, city)
        exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")
        self.images = sorted([os.path.join(city_dir, f) for f in os.listdir(city_dir) if f.lower().endswith(exts)])
        self.idx = 0
        self.show_image()  # show first
        self.update_status()

    def refresh_images(self):
        self.city_changed()

    def _load_model(self, weights_path):
        try:
            t0 = time.time()
            self.model = YOLO(weights_path)
            dt = (time.time() - t0) * 1000
            self.status_var.set(f"Loaded model: {weights_path} ({dt:.0f} ms). CUDA: {self.model.device}.")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load model:\n{weights_path}\n\n{e}")
            self.model = None

    # --------------- Rendering ----------------
    def clear_canvas(self):
        self.canvas.delete("all")
        self.tkimg_cache = None

    def show_image(self):
        self.clear_canvas()
        if not self.images:
            self.status_var.set("No images found. Choose a Test root with city subfolders.")
            return
        path = self.images[self.idx]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            self.status_var.set(f"Failed to open: {path} | {e}")
            return

        # Resize for display (keep aspect), remember scale to map boxes
        w, h = pil.size
        scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
        disp_w = int(w * scale)
        disp_h = int(h * scale)
        pil_disp = pil.resize((disp_w, disp_h), Image.BILINEAR)

        # Draw the raw image first
        self.tkimg_cache = ImageTk.PhotoImage(pil_disp)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.create_image(0, 0, image=self.tkimg_cache, anchor="nw")

        # Write meta (top-left)
        self.canvas.create_text(8, 8, anchor="nw", fill="#f0f0f0",
                                text=f"{os.path.basename(path)}  [{w}x{h}]  ({self.idx+1}/{len(self.images)})")

        # If we already inferred for this image, draw the last result overlays
        if self.last_result is not None and self.last_result.get("path") == path:
            self._draw_overlays(self.last_result, scale)

    def _draw_overlays(self, result, scale):
        # boxes are in original coordinates. We need to draw scaled boxes on the canvas.
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        classes = result.get("classes", [])
        drawn = 0
        for b, s, c in zip(boxes, scores, classes):
            x1, y1, x2, y2 = [int(v * scale) for v in b]
            # rectangle
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00FF00", width=2)
            label = f"person {s:.2f}"
            self.canvas.create_text(x1+4, max(0, y1-12), anchor="nw", fill="#00FF00", text=label)
            if self.show_centroid_var.get():
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, outline="#00FF00", fill="#00FF00", width=1)
            drawn += 1

        # bottom-left stats
        self.canvas.create_text(8, self.canvas.winfo_height()-8, anchor="sw", fill="#f0f0f0",
                                text=f"Detections: {drawn} | conf≥{self.conf_var.get():.2f} IoU {self.iou_var.get():.2f}")

    # --------------- Inference ----------------
    def run_infer_current(self):
        if self.model is None:
            messagebox.showwarning("No model", "Load a YOLO model first.")
            return
        if not self.images:
            return
        path = self.images[self.idx]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            self.status_var.set(f"Failed to open: {path} | {e}")
            return

        img = np.array(pil)  # H,W,3 RGB
        conf = float(self.conf_var.get())
        iou = float(self.iou_var.get())

        t0 = time.time()
        # classes=[0] filters to 'person' on COCO
        res = self.model.predict(source=img, conf=conf, iou=iou, classes=[PERSON_CLASS_ID], verbose=False)[0]
        dt = (time.time() - t0) * 1000

        # Extract boxes in original coordinate space
        if res.boxes is not None and res.boxes.xyxy is not None:
            boxes = res.boxes.xyxy.cpu().numpy().tolist()    # [N,4] in (x1,y1,x2,y2)
            scores = res.boxes.conf.cpu().numpy().tolist()   # [N]
            classes = res.boxes.cls.cpu().numpy().tolist()   # [N]
        else:
            boxes, scores, classes = [], [], []

        self.last_result = {
            "path": path,
            "boxes": boxes,
            "scores": scores,
            "classes": classes
        }

        self.status_var.set(f"Infer: {os.path.basename(path)} | {len(boxes)} dets | {dt:.0f} ms | conf≥{conf:.2f}, IoU {iou:.2f}")
        self.show_image()

        # Optional save
        if self.save_drawn_var.get():
            self._save_drawn(path, boxes, scores)

    def _save_drawn(self, path, boxes, scores):
        try:
            pil = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(pil)
            w, h = pil.size
            for b, s in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, b)
                draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)
                draw.text((x1+2, max(0, y1-12)), f"person {s:.2f}", fill=(0,255,0))
                if self.show_centroid_var.get():
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    draw.ellipse([cx-3, cy-3, cx+3, cy+3], outline=(0,255,0), fill=(0,255,0), width=1)
            save_dir = os.path.join(os.path.dirname(path), "_yolo_out")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(path))
            pil.save(save_path, quality=95)
        except Exception as e:
            print(f"[WARN] Save failed for {path}: {e}")

    # --------------- Navigation ----------------
    def prev_image(self):
        if not self.images:
            return
        self.idx = (self.idx - 1) % len(self.images)
        self.last_result = None
        self.show_image()

    def next_image(self):
        if not self.images:
            return
        self.idx = (self.idx + 1) % len(self.images)
        self.last_result = None
        self.show_image()

    def update_status(self):
        n_img = len(self.images)
        city = self.city_var.get()
        self.status_var.set(f"City: {city or '-'} | Images: {n_img} | conf≥{self.conf_var.get():.2f} IoU {self.iou_var.get():.2f}")

if __name__ == "__main__":
    app = YoloViewer()
    app.mainloop()
