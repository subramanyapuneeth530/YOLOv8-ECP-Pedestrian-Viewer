# YOLOv8-ECP-Pedestrian-Viewer
Tkinter-based GUI to visualize YOLOv8 pedestrian detections on the ECP dataset (or any city-structured image folder), with support for swapping pretrained/fine-tuned YOLOv8 weights directly from the interface.

## Introduction

This project provides a simple desktop GUI for **visualizing pedestrian detections** using **YOLOv8** on the **ECP (Extended Cityscapes Pedestrian)** dataset or any similarly structured folder of city images. The goal is to make it easy to browse through images, run a pretrained (or fine-tuned) YOLO model, and visually inspect how well it detects pedestrians in real urban scenes.

### ECP (Extended Cityscapes Pedestrian) Dataset

The **Extended Cityscapes Pedestrian (ECP)** dataset is a benchmark focused specifically on **pedestrians in urban environments**. It extends the original Cityscapes dataset with **richer annotations** aimed at pedestrian detection and analysis.

Key characteristics:

- **Urban street scenes**  
  ECP contains images captured from vehicle-mounted cameras driving through real European cities. Scenes typically include roads, sidewalks, crossings, buildings, parked cars, and a variety of pedestrians.

- **Dense pedestrian annotations**  
  The dataset provides **bounding boxes** for pedestrians, and in many cases includes:
  - Standing and walking pedestrians
  - Occluded or partially visible pedestrians
  - Small, distant pedestrians
  This makes ECP more challenging than simple, clean pedestrian datasets.

- **Fine-grained difficulty levels (depending on subset)**  
  Images can be grouped by detection difficulty (e.g. heavy occlusion, crowded scenes, small scale), making it useful for evaluating how robust a detector is under realistic conditions.

- **Train / validation (and sometimes test) splits**  
  ECP is usually split into **train**, **validation**, and possibly **test** partitions. In this project, we typically work with something like:
  - `Train/Images/<city_name>/…` for training/inspection
  - `Val/Images/<city_name>/…` for validation/inspection  
  Each **city** (e.g. `amsterdam`, `zurich`) is represented as its own subfolder of images.

This viewer does **not** require labels directly; it focuses on visualizing **model predictions** on the ECP images (or any similarly organized dataset). However, the same dataset can also be used to **fine-tune** YOLOv8 models for more accurate pedestrian detection.

### YOLO and YOLOv8

**YOLO (You Only Look Once)** is a family of real-time object detection models that treat detection as a **single pass** over the image:

1. The image is passed through a convolutional neural network.
2. The network directly predicts:
   - Bounding boxes (where objects are)
   - Class labels (what the objects are)
   - Confidence scores (how sure the model is)

**YOLOv8** is a modern, improved version released by Ultralytics. It introduces:

- A flexible family of models (`yolov8n`, `yolov8s`, `yolov8m`, etc.) that trade off speed vs accuracy.
- An **anchor-free** detection head (simplifies and often improves box prediction).
- Better training and deployment pipeline through the Ultralytics Python API and CLI.

In this project:

- We load a YOLOv8 model (e.g. `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) via the **Ultralytics** library.
- We filter predictions to **COCO class ID 0: `person`**, since the focus is pedestrian detection.
- The GUI lets you swap out model weights easily — including your own **fine-tuned** YOLOv8 models trained on ECP.

### Pedestrian Detection in Urban Scenes

**Pedestrian detection** is a core task in:

- Autonomous driving and ADAS (advanced driver assistance systems)
- Smart city infrastructure (e.g. traffic analysis, crowd monitoring)
- Surveillance and safety systems
- Robotics and navigation in human environments

Urban pedestrian detection is challenging because:

- Pedestrians are often **partially occluded** by cars, poles, other people, etc.
- Many pedestrians are **small** or **far away** in the frame.
- Lighting and weather conditions can vary widely.
- Background clutter (signs, reflections, etc.) can cause false positives.

The ECP dataset is designed to reflect these real-world challenges, and YOLOv8 provides a fast, reasonably accurate baseline detector. This viewer brings those two together:

- You load a YOLOv8 model (pretrained or fine-tuned).
- You browse ECP images (organized by city).
- You visually inspect how well pedestrians are detected:
  - Are small, distant people found?
  - Are occluded pedestrians missed?
  - Are there many false detections?

### What This Project Provides

This repository focuses on **visual inspection and exploration**, not training (for now). It provides:

- A **Tkinter GUI** for:
  - Selecting a root image folder (with city subfolders).
  - Choosing a YOLOv8 weights file.
  - Navigating through images with buttons or keyboard shortcuts.
- A simple way to:
  - Run **inference** and overlay boxes/centroids for the `person` class.
  - Adjust **confidence** and **IoU** thresholds interactively.
  - Optionally **save annotated copies** of the images for reports or debugging.

Later, this can be extended with:

- **Fine-tuning scripts** for YOLOv8 on ECP.
- **Benchmarking** different YOLOv8 variants on ECP (accuracy vs speed).
- Advanced visualization features (video mode, tracking, etc.).

### TL;DR: modern detector (**YOLOv8**) on a realistic, pedestrian-focused dataset (**ECP**) and wrapping it in a small, practical tool to better understand how well pedestrians are detected in real city scenes.

## Features

### 🔍 Pedestrian-Focused YOLOv8 Inference

- Runs object detection using **Ultralytics YOLOv8** models.
- Filters detections to **COCO class `0` = `person`**, making the viewer focused on **pedestrian detection**.
- Supports any YOLOv8 weights:
  - Official pretrained models (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, etc.).
  - Your own **fine-tuned** models (e.g. `runs/detect/train/weights/best.pt`).

### 🖼️ Tkinter-Based Image Viewer

- Simple, standalone **desktop GUI** built with Tkinter (no web server required).
- Displays images scaled to fit the window while preserving aspect ratio.
- Overlays:
  - **Bounding boxes** around detected pedestrians.
  - Optional **centroid dots** at the center of each box.
- Shows basic image metadata:
  - File name
  - Original resolution
  - Current image index (e.g. `5 / 120`)

### 📂 City-Wise Dataset Browsing

- Designed to work naturally with the **ECP dataset layout** or any similar structure:
  - Root folder with **city subfolders** (`amsterdam/`, `zurich/`, etc.).
- Automatically:
  - Scans the root folder for city directories.
  - Populates a **City** dropdown in the GUI.
  - Lists image files within the selected city.
- Supports common image formats:
  - `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`

### 🎛️ Interactive Controls

- **Confidence slider**:
  - Adjust the minimum detection confidence threshold.
  - Useful for pruning low-confidence detections.
- **IoU slider**:
  - Adjust Non-Maximum Suppression (NMS) IoU threshold.
  - Helps control how overlapping boxes are merged/filtered.
- **Toggles**:
  - `Save drawn`: save annotated images with boxes/centroids.
  - `Show centroid`: show or hide the center dot of each pedestrian box.

### ⌨️ Keyboard Shortcuts & Navigation

- `←` / **Prev**: show the previous image in the current city.
- `→` / **Next**: show the next image in the current city.
- `Space` / **Infer**: run YOLOv8 inference on the current image.
- **Refresh List** button:
  - Re-scan the currently selected city folder for new or changed images.

### 💾 Saving Annotated Images

- When **“Save drawn”** is enabled:
  - Each inference saves an annotated copy of the image.
  - Annotations are drawn using **Pillow (PIL)**:
    - Green bounding boxes
    - Confidence labels (e.g. `person 0.87`)
    - Optional centroid dots
- Output is saved in a `_yolo_out` folder beside the original image, for example:
  ```text
  amsterdam/
    img_0001.jpg
    img_0002.jpg
    _yolo_out/
      img_0001.jpg   # annotated
      img_0002.jpg   # annotated
  ```

### 🧠 CPU and GPU Friendly

- Works on **CPU-only** setups (slower but widely compatible).
- If a **CUDA-enabled PyTorch** is installed and an **NVIDIA GPU** is available:
  - YOLOv8 will automatically use the GPU.
- The status bar shows the device in use:
  - `cuda:0` for GPU  
  - `cpu` for CPU
- Suitable for lightweight experimentation even on **laptop GPUs** (e.g. 4 GB VRAM).

### 🧩 Minimal, Extensible Codebase

- Single main script: `yolo_pedestrian_viewer.py`, with:
  - Clear separation of **GUI building**, **data loading**, **model loading**, **rendering**, and **inference**.
- Uses **Pillow** and **Tkinter** for visualization; **OpenCV** is imported optionally for potential future features:
  - e.g. video support, advanced preprocessing
- Easy to extend with:
  - Additional **class filters**
  - Different **model types**
  - Custom overlays (e.g. tracking IDs, difficulty levels, extra annotations)

## Requirements & Installation

### Environment

- **Python**: 3.9+ (3.9–3.11 recommended)
- **OS**: Windows / Linux (Tkinter + GPU support tested mainly on Windows)
- **GPU (optional)**: NVIDIA GPU with **CUDA** support (e.g. laptop RTX series)

> ✅ The viewer works on **CPU-only** as well.  
> 🚀 For **GPU acceleration**, you must have a **CUDA-enabled PyTorch** build.

### Python Libraries

Core dependencies:

- `torch` (with CUDA if you want GPU; CPU-only also works)
- `ultralytics` (YOLOv8)
- `pillow`
- `numpy`
- `opencv-python` *(optional, not required in current logic)*

### Installation

1. (Optional but recommended) Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or matching CUDA version
   pip install ultralytics pillow numpy opencv-python
   ```

## Dataset & Folder Structure

### EuroCity Persons (ECP) Dataset

This viewer is built around the **EuroCity Persons (ECP)** dataset – a large-scale benchmark for **person detection in urban traffic scenes**.

Key characteristics:

- **Scope & Diversity**  
  - Images recorded from a **moving vehicle** in **31 cities** across **12 European countries**.  
  - Covers a wide range of **locations, seasons, weather conditions** (dry, rain, snow), and **times of day and night**.  
  - This diversity makes it a strong testbed for real-world pedestrian detection.

- **Annotations**  
  - Over **238,200 manually labeled person instances**, including:
    - Pedestrians  
    - Riders (on bicycles, mopeds, motorbikes)  
  - Annotations go beyond plain bounding boxes and can include details like:
    - **Body orientation**
    - **Occlusion level**
  - These rich labels are highly valuable for training and benchmarking deep learning models.

- **Purpose**  
  - Designed as a **robust benchmark** for evaluating **person detection** in complex traffic environments.  
  - Directly relevant for:
    - Road safety research  
    - Autonomous driving and ADAS  
    - Urban perception systems (understanding people in street scenes)

- **Extended Versions**  
  - An enhanced release, **EuroCity Persons 2.0**, additionally provides **LiDAR data** to support **3D object localization** research.

- **Availability**  
  - The dataset is available for **non-commercial benchmarking** via the official EuroCity Persons website (registration/approval usually required).

> 🔎 This viewer focuses on **visualizing model predictions** on EuroCity Persons images.  
> Labels are not required for the GUI itself, but they are essential if you later train or fine-tune YOLO models on this dataset.

---

### Expected Folder Structure for the Viewer

The script expects a **root folder** (in this code it is either Train/Images , Validation/Images or Test) containing **city subfolders** with images.  
Below is the original file structure of the ECP Dataset

```text
EuroCityPersons/
  Train/
    Images/
      amsterdam/
        amsterdam_000000.png
        amsterdam_000001.png
        amsterdam_000002.png
        ...
      zurich/
        zurich_000000.png
        zurich_000001.png
        ...
      munich/
        ...
      ...
    Labels/
      amsterdam/
        amsterdam_000000.json
        amsterdam_000001.json
        amsterdam_000002.json
        ...
      zurich/
        zurich_000000.json
        zurich_000001.json
        ...
      munich/
        ...
      ...
  Val/
    Images/
      amsterdam/
        amsterdam_010000.png
        ...
      zurich/
        ...
      ...
    Labels/
      amsterdam/
        amsterdam_010000.json
        ...
      zurich/
        ...
      ...
  Test/
    amsterdam/
      amsterdam_010000.png
      ...
    zurich/
    ...
  ...
...
```

## Usage, Models & Runtime Notes

### Running the Viewer
**Start the application**

   -From the project root:

   ```bash
   python yolo_pedestrian_viewer.py
   ```
**Select the dataset root**

- At the top of the GUI, under **“Test root”**, either:
  - Use the default path (for example `E:\ECP DATASET\Train\Images`), or  
  - Choose the folder that contains your **city subfolders**, for example:
    - `EuroCityPersons/Train/Images/`

### Choose a City

- Use the **“City”** dropdown to select one of the subfolders (e.g. `amsterdam`).
- The first image from that city will be displayed.

---

### Run Inference

- Click **“Infer (Space)”** or press the **Space** bar.
- YOLOv8 will run on the current image and draw:
  - Green **bounding boxes** for persons (COCO class `0`).
  - Optional **centroid dots** in the center of each box.

- The status bar at the bottom shows:
  - File name and **number of detections**.
  - **Inference time** (ms).
  - Current **confidence** and **IoU** thresholds.

---

### Navigate Images

**Buttons:**

- **Prev (←)** → previous image in the selected city.  
- **Next (→)** → next image in the selected city.  
- **Refresh List** → re-scan the current city folder for images.

**Keyboard:**

- **Left Arrow (←)** → previous image.  
- **Right Arrow (→)** → next image.  
- **Space** → run inference on the current image.

---

### Adjust Thresholds & Options

**`conf` slider:**

- Sets the **minimum confidence** for displaying a detection.  
- Filters out low-confidence boxes.

**`IoU` slider:**

- Controls **NMS (Non-Maximum Suppression) overlap**.  
- A higher IoU means overlapping boxes are merged more strictly.

**`Save drawn` checkbox:**

- When enabled, each inference saves an annotated copy of the image.  
- Annotated images are written to a `_yolo_out` subfolder next to the original image.

**`Show centroid` checkbox:**

- Toggles drawing of a small dot at the **center of each bounding box**.  
- Affects both the GUI display and any saved annotated images.

### Model Weights & GPU/CPU Behavior

#### Selecting Model Weights

The **“YOLO weights”** field lets you load any compatible Ultralytics YOLOv8 model, for example:

- **Official pretrained weights:**
  - `yolov8n.pt` (nano – fastest, lightest)  
  - `yolov8s.pt` (small)  
  - `yolov8m.pt`, etc.

- **Your own trained or fine-tuned weights:**
  - `runs/detect/train/weights/best.pt`

**Typical workflow:**

1. Type or paste the model path into the **“YOLO weights”** textbox.  
2. Click **“Load Model”**.  
3. The status bar updates showing:
   - How long loading took.  
   - Which device the model is on (**CPU** or **GPU**).

> The viewer is **person-focused**: predictions are filtered to COCO class ID `0`, so only pedestrian-like detections are shown.

---

#### CPU vs GPU

The viewer runs on:

- **CPU-only**  
  - Always works.  
  - Slower, but fine for inspection and browsing.

- **GPU (CUDA)**  
  - Much faster inference if:
    - You have an **NVIDIA GPU**, and  
    - **PyTorch** is installed with **CUDA support**.

When a model is loaded, the **status bar** displays the active device. For example:

```text
Loaded model: yolov8n.pt (123 ms). CUDA: cuda:0
```
cuda:0 → the model is using your GPU and cpu → the model is running on CPU.

If you always see `cpu` even though you have an NVIDIA GPU, you likely installed a **CPU-only PyTorch build**. Reinstall PyTorch with a **CUDA-enabled wheel** to use the GPU.

For laptop GPUs with around **4 GB VRAM** (for example an **RTX 2050**), `yolov8n` and `yolov8s` are usually **safe and responsive choices** for inference in this viewer.

---

### Optional OpenCV Dependency

At the top of the script, OpenCV is imported **optionally**:

- The code tries to import `cv2` and sets a flag `HAS_CV = True` if successful, otherwise `False`.

In the **current version** of the viewer:

- All image loading, resizing, and drawing are handled via **Pillow (PIL)** and **Tkinter**.
- **OpenCV (`cv2`) is not used** in the main logic yet.

#### Why keep OpenCV?

It acts as a **future extension point** for:

- Video support (for example, running YOLO on dashcam or surveillance videos instead of only static images).  
- Advanced image preprocessing (blurring, color transforms, data augmentation).  
- More complex operations (optical flow, tracking, additional visual analytics).

#### Minimal setup

If you prefer a minimal setup, you can safely remove:

- The optional `cv2` import block from the script.  
- The `opencv-python` entry from your dependency list.

The viewer will continue to work **exactly the same** without OpenCV in its current form.

## Future Work

Going forward, I want to turn this project from a pure **viewer** into a full **pedestrian detection playground** built around the EuroCity Persons dataset. Some key directions:

### 1. Fine-Tuning YOLO Models on ECP

- Fine-tune **YOLOv8** variants on ECP:
  - `yolov8n`, `yolov8s`, `yolov8m` as main candidates.
- Experiment with **different training regimes**:
  - Freeze backbone vs full fine-tuning.
  - Different image sizes, batch sizes, and augmentation strategies.
- Compare:
  - **Pretrained COCO weights** vs **ECP fine-tuned** weights in terms of recall/precision on pedestrians.

### 2. Training Other YOLO Families

- Extend beyond YOLOv8 to other popular YOLO-based models:
  - **YOLOv5**, **YOLOv7**, or **YOLOv9** (if available in the ecosystem).
  - Compare speed/accuracy trade-offs on ECP.
- Provide unified training scripts so that:
  - Different YOLO families can be trained with **similar configs** (same splits, same augmentations).
  - Results can be easily compared in a common format.

### 3. Custom Architectures & Model Variants

- Design and implement **custom detection architectures** inspired by:
  - YOLO-style one-stage detectors.
  - Other popular backbones (e.g. ResNet, CSPNet, MobileNet) adapted for detection.
- Experiment with:
  - **Lightweight models** specialized for low-power devices (edge, embedded).
  - **Heavier models** focused purely on accuracy, using ECP as a stress test.
- Explore:
  - Different **feature pyramid / neck designs**.
  - **Anchor-based** vs **anchor-free** detection heads.

### 4. Exploiting the Rich ECP Annotations

The ECP dataset contains more than just basic bounding boxes. Possible directions:

- Use **occlusion levels** to:
  - Train models that are robust to **heavy occlusion**.
  - Study performance separately for **easy / medium / hard** visibility conditions.
- Use **body orientation** annotations (where available) to:
  - Train multi-task models that predict both **bounding box + orientation**.
  - Analyze how orientation affects detection quality.

### 5. Benchmarking & Analysis

- Build a **benchmark suite** for ECP-based pedestrian detection:
  - mAP, precision, recall for different models and configs.
  - City-wise breakdown (e.g. performance in `amsterdam` vs `zurich`).
- Add scripts to:
  - Plot **PR curves**, **confidence histograms**, and **error analysis** (false positives vs false negatives).
  - Generate per-city or per-condition (day/night, weather) performance reports.

### 6. Video, Tracking & Temporal Models

- Extend from static images to **video sequences**:
  - Use OpenCV to run YOLO on **continuous frames**.
  - Add **simple tracking** (e.g. SORT, DeepSORT) to track pedestrians over time.
- Investigate:
  - How stable detections are frame-to-frame.
  - Whether temporal information can help with small or occluded pedestrians.

### 7. Domain Adaptation & Generalization

- Use ECP as a **source domain** to test:
  - How well models trained on ECP generalize to other datasets (e.g. Cityscapes, COCO subsets, other pedestrian datasets).
  - Domain adaptation techniques (fine-tuning on a new city or dataset).
- Study:
  - Cross-city generalization: train on some cities, test on unseen cities within ECP.
  - Day → night / dry → rain domain shifts.

### 8. Semi-Supervised and Active Learning

- Use ECP’s large unlabeled or weakly-labeled portions (if available) to:
  - Explore **semi-supervised detection**, where some data is labeled and some is not.
- Implement simple **active learning loops**:
  - Run the detector on unlabeled images.
  - Select the most uncertain or highest-loss samples.
  - Pretend to “label” them (or simulate labeling) and re-train to see the gains.

### 9. 3D & Sensor Fusion (EuroCity Persons 2.0)

- For EuroCity Persons 2.0 with **LiDAR data**:
  - Explore 2D–3D fusion:
    - 2D pedestrian detection in the image.
    - 3D localization using LiDAR points.
  - Experiment with:
    - Projecting LiDAR into the camera frame.
    - Building joint models that reason in both 2D and 3D.

### 10. Tooling, Configs & Packaging

- Add:
  - **Config files** (YAML/JSON) for experiments: models, hyperparameters, dataset paths.
  - **CLI tools** to:
    - Launch training.
    - Run evaluations.
    - Export models for deployment.
- Package the viewer + training tools as:
  - A small library/module (`ecp_pedestrian_tools`).
  - Optional standalone apps (e.g. via PyInstaller) for people who just want the viewer.

---

In short, this project can evolve from a simple viewer into a complete **research and experimentation framework** for pedestrian detection on EuroCity Persons: from **fine-tuning YOLO** and **training custom architectures**, to **benchmarking**, **analysis**, **video tracking**, and even **3D perception** with the extended versions of the dataset.
