# FOMO-3D Detection — Foundation-Model-Assisted Long-Tail 3D Object Detection

A ROS2 Humble Python implementation of **FOMO-3D** (Yang, Tu, Dvornik, Li & Urtasun, CoRL 2025),
a two-stage multi-modal 3D object detector that fuses LiDAR point clouds with camera imagery
and leverages large vision foundation models — OWLv2 and Metric3Dv2 — to detect rare and
long-tail objects that LiDAR-only detectors routinely miss.

**No official source code exists.** The Waabi authors have not released an implementation.
This package is built directly from the paper (arXiv:2603.08611) and uses publicly available
open-source components for every sub-module.

---

## Table of Contents

1. [Motivation — The Long-Tail Problem in 3D Detection](#1-motivation--the-long-tail-problem-in-3d-detection)
2. [What FOMO-3D Does](#2-what-fomo-3d-does)
3. [Architecture Overview](#3-architecture-overview)
4. [Plain-English Walkthrough](#4-plain-english-walkthrough)
    - [4.1 What is a Long-Tail Object?](#41-what-is-a-long-tail-object)
    - [4.2 What is a Two-Stage Multi-Modal 3D Object Detector?](#42-what-is-a-two-stage-multi-modal-3d-object-detector)
    - [4.3 What is Recall, Precision, and mAP?](#43-what-is-recall-precision-and-map)
    - [4.4 What are Vision Foundation Models?](#44-what-are-vision-foundation-models)
    - [4.5 What is a Frustum?](#45-what-is-a-frustum)
    - [4.6 What is Metric3Dv2?](#46-what-is-metric3dv2)
    - [4.7 How Stage 1 and Stage 2 Work in Plain English](#47-how-stage-1-and-stage-2-work-in-plain-english)
    - [4.8 OWLv2's Dual Role](#48-owlv2s-dual-role)
5. [Mathematical Background](#5-mathematical-background)
    - [5.1 Camera Projection Model](#51-camera-projection-model)
    - [5.2 Camera-LiDAR Extrinsic Transform](#52-camera-lidar-extrinsic-transform)
    - [5.3 Frustum Definition](#53-frustum-definition)
6. [LiDAR Branch](#6-lidar-branch)
    - [6.1 Point Cloud Preprocessing](#61-point-cloud-preprocessing)
    - [6.2 PointPillars Backbone](#62-pointpillars-backbone)
    - [6.3 LiDAR 3D Proposals](#63-lidar-3d-proposals)
7. [Camera Branch — The Novel Contribution](#7-camera-branch--the-novel-contribution)
    - [7.1 OWLv2 Open-Vocabulary 2D Detection](#71-owlv2-open-vocabulary-2d-detection)
    - [7.2 Metric3Dv2 Monocular Depth](#72-metric3dv2-monocular-depth)
    - [7.3 Frustum Lift to 3D](#73-frustum-lift-to-3d)
    - [7.4 Frustum-Based Fusion Module](#74-frustum-based-fusion-module)
8. [Proposal Fusion — NMS Merge](#8-proposal-fusion--nms-merge)
9. [DETR-Style Cross-Attention Refinement](#9-detr-style-cross-attention-refinement)
10. [ROS2 Node Architecture](#10-ros2-node-architecture)
    - [10.1 Node I/O](#101-node-io)
    - [10.2 Processing Pipeline](#102-processing-pipeline)
    - [10.3 Topic Reference](#103-topic-reference)
11. [Package Structure](#11-package-structure)
12. [Dependencies](#12-dependencies)
13. [Installation](#13-installation)
14. [Running](#14-running)
15. [Configuration](#15-configuration)
16. [Tuning Guide](#16-tuning-guide)
17. [References](#17-references)
- [Appendix A — Full Frustum Lift Derivation](#appendix-a--full-frustum-lift-derivation)
- [Appendix B — NMS for 3D Boxes](#appendix-b--nms-for-3d-boxes)
- [Appendix C — Cross-Attention Formulation](#appendix-c--cross-attention-formulation)

---

## 1. Motivation — The Long-Tail Problem in 3D Detection

Standard 3D object detectors for autonomous vehicles — PointPillars, CenterPoint, VoxelNet —
are trained almost entirely on driving datasets (nuScenes, Waymo, KITTI). These datasets are
dominated by cars, trucks, and pedestrians; rare but safety-critical objects appear so
infrequently that detectors effectively ignore them:

| Object class | Relative frequency | Detector performance |
|---|---|---|
| Car | Very common | High mAP (~85%) |
| Pedestrian | Common | Moderate mAP (~70%) |
| Cyclist | Uncommon | Moderate mAP (~55%) |
| Construction worker | Rare | Very low mAP (~20%) |
| Wheelchair user | Very rare | Near zero mAP |
| Traffic cone (novel) | Rare | Near zero mAP |

A construction worker standing next to a breakdown on the highway is one of the most dangerous
road scenarios — and also one a standard LiDAR detector is most likely to miss. The model has
seen too few examples during training to reliably detect it.

FOMO-3D solves this by injecting knowledge from large **vision foundation models** — specifically
OWLv2, which has been trained on internet-scale image data containing millions of examples of
every conceivable object class. The foundation model cannot directly produce 3D detections, but
it can identify where in the image a rare object appears with high precision. FOMO-3D then lifts
those 2D detections into 3D by combining them with a dense monocular depth map and nearby LiDAR
geometry.

**Result:** +7.6 mAP on few-shot categories, +2.0 mAP even on common categories (paper §4).

---

## 2. What FOMO-3D Does

Given a single LiDAR scan and one or more camera images taken at the same timestamp, FOMO-3D
outputs a set of axis-aligned 3D bounding boxes, each annotated with a class label and
confidence score, in the LiDAR sensor coordinate frame.

**Inputs:**
- `sensor_msgs/PointCloud2` — LiDAR point cloud, N × (x, y, z, intensity)
- `sensor_msgs/Image` — camera image, H × W × 3, RGB
- `sensor_msgs/CameraInfo` — camera intrinsics K and distortion coefficients
- `geometry_msgs/TransformStamped` — extrinsic calibration T_lidar_cam (LiDAR frame ← camera frame)
- Text prompts — user-specified class names to detect (e.g. `["car", "pedestrian", "construction worker"]`)

**Outputs:**
- `vision_msgs/Detection3DArray` — 3D bounding boxes in LiDAR frame, one per detected object
- `visualization_msgs/MarkerArray` — RViz visualisation of the detections

---

## 3. Architecture Overview

```
                        ┌─────────────────────────────────────────────────────────┐
                        │                     FOMO-3D Node                        │
                        │                                                          │
  /lidar/points ───────►│  ┌──────────────────────────────────────────────────┐   │
                        │  │            STAGE 1: PROPOSAL GENERATION          │   │
                        │  │                                                  │   │
                        │  │  ┌─────────────────────┐                        │   │
                        │  │  │    LiDAR Branch      │                        │   │
                        │  │  │                      │                        │   │
                        │  │  │  1. Voxelise         │                        │   │
                        │  │  │  2. PointPillars     │──► LiDAR proposals     │   │
                        │  │  │     backbone         │    (accurate geometry, │   │
                        │  │  │  3. 3D anchor heads  │     common objects)    │   │
                        │  │  └─────────────────────┘         │               │   │
                        │  │                                   │               │   │
  /camera/image_raw ───►│  │  ┌─────────────────────┐         │               │   │
  /camera/camera_info ──►│  │  │    Camera Branch     │         │               │   │
                        │  │  │                      │         │               │   │
                        │  │  │  1. OWLv2            │         │               │   │
                        │  │  │     (2D boxes +      │         │               │   │
                        │  │  │      image features) │         ▼               │   │
                        │  │  │  2. Metric3Dv2       │──► Camera proposals    │   │
                        │  │  │     (dense depth)    │    (rare objects,      │   │
                        │  │  │  3. Frustum Lift     │     small objects,     │   │
                        │  │  └─────────────────────┘     long range)        │   │
                        │  │                                   │               │   │
                        │  │            ┌──────────────────────┘               │   │
                        │  │            ▼                                      │   │
                        │  │         NMS Merge                                 │   │
                        │  │   (concat + 3D IoU NMS)                          │   │
                        │  └──────────────────────────────────────────────────┘   │
                        │                      │                                   │
                        │                      ▼                                   │
                        │  ┌──────────────────────────────────────────────────┐   │
                        │  │           STAGE 2: CROSS-ATTENTION REFINEMENT    │   │
                        │  │                                                  │   │
                        │  │  For each merged proposal:                       │   │
                        │  │  - Project box centre into image                 │   │
                        │  │  - Query OWLv2 image feature map                 │   │
                        │  │    via cross-attention                           │   │
                        │  │  - Refine box location + confidence              │   │
                        │  └──────────────────────────────────────────────────┘   │
                        │                      │                                   │
                        └──────────────────────┼───────────────────────────────────┘
                                               │
                               ┌───────────────┼────────────────┐
                               ▼                                 ▼
                  /detections/objects                 /detections/markers
                  (Detection3DArray)                  (MarkerArray → RViz)
```

The two-stage structure mirrors DETR-based detectors: Stage 1 generates candidate proposals
from two complementary sensors; Stage 2 refines them with rich image semantics.

---

## 4. Plain-English Walkthrough

This section explains every key concept in plain language before the mathematics. Read this
first if any term in the architecture overview was unfamiliar.

---

### 4.1 What is a Long-Tail Object?

Imagine plotting every object class in a driving dataset sorted by how often it appears:

```
Frequency
   │
   █
   █
   █  █
   █  █
   █  █  █
   █  █  █  █
   █  █  █  █  █  █  █  █  █  █  █  █ ...
   └────────────────────────────────────────► Classes
   car truck ped  cyc  ...  wheelchair  cone  stroller
   ◄── HEAD ──►         ◄──────── LONG TAIL ──────────►
```

A small number of classes (cars, trucks, pedestrians) appear millions of times — the **head**.
Hundreds of other classes appear rarely — the **long tail**.

Neural networks learn from examples. A class that appears 500 000 times in training data is
learned very well. A class that appears 50 times is barely learned at all. The dangerous irony
in autonomous driving: the rarest objects are often the most safety-critical. A construction
worker by the road may appear in 0.001% of training frames, but missing one in deployment is
catastrophic. That asymmetry is the long-tail problem.

---

### 4.2 What is a Two-Stage Multi-Modal 3D Object Detector?

**3D Object Detector** — finds objects and outputs a 3D bounding box around each one: position
(x, y, z), size (length, width, height), and rotation (yaw). Unlike a 2D detector that draws a
rectangle on an image, a 3D detector tells you where something is in the real world and how big
it is.

**Multi-Modal** — uses more than one sensor type. FOMO-3D uses two:

- **LiDAR** — gives accurate geometry (exact distances, point positions) but no colour or texture
- **Camera** — gives rich appearance (colour, shape, texture) but no direct depth

Neither sensor alone is sufficient. LiDAR misses rare objects because it only sees shape.
Camera misses depth because a flat image cannot tell you how far away something is. Combining
them covers each other's weaknesses.

**Two-Stage** — detections are produced in two passes rather than one:

```
Stage 1 — "Where might there be something?"
   Scan the input quickly. Generate many rough candidate regions
   called proposals. Cast a wide net — high recall, don't miss
   anything, even at the cost of some false positives.

Stage 2 — "Of those candidates, what exactly is there?"
   Look more carefully at each proposal. Refine the box position.
   Confirm or reject the class label. Suppress false positives.
   Output only the confident, well-localised detections.
```

Doing both jobs in a single pass is harder. Splitting them lets Stage 1 be permissive and
fast, and Stage 2 be precise and selective.

---

### 4.3 What is Recall, Precision, and mAP?

**Recall** — of all real objects that exist in the scene, what fraction did you find?

```
Recall = objects correctly detected / all objects that actually exist
```

If there are 10 pedestrians and you detected 8: Recall = 80%. You missed 2 (false negatives).

**Precision** — of everything you reported as a detection, how many were actually real?

```
Precision = objects correctly detected / everything you reported as a detection
```

If you detected 8 real pedestrians but also flagged 4 bushes: Precision = 8/12 = 67%.
Those 4 bushes are false positives.

**The tradeoff** — you can always get perfect recall by flagging everything as an object, but
precision collapses. You can get perfect precision by only reporting when 100% certain, but
recall suffers. This is why Stage 1 uses a low threshold (prioritise recall — don't miss
anything) and Stage 2 uses a higher threshold (restore precision — remove false positives).

**Average Precision (AP)** — as you vary the confidence threshold, recall and precision trace
a curve. AP is the area under that curve for one class. AP = 1.0 is perfect; AP = 0.5 is
decent.

**mAP (mean Average Precision)** — the mean of AP across all classes:

```
mAP = (AP_car + AP_truck + AP_pedestrian + AP_wheelchair + ...) / num_classes
```

One number summarising overall detection performance. The paper reports +7.6 mAP on rare
classes and +2.0 mAP on common classes — meaning FOMO-3D raises the average area under the
precision-recall curve by those amounts relative to prior detectors.

In safety-critical robotics, missing a real object (low recall) is usually more dangerous than
a false alarm (low precision) — a robot that sees a phantom pedestrian will brake unnecessarily,
but a robot that misses a real pedestrian might hit them.

---

### 4.4 What are Vision Foundation Models?

Before foundation models, you trained a separate model for each task:

```
Training data for cars  → Model A → detects cars
Training data for depth → Model B → estimates depth
```

Each model only knows what it was explicitly shown. If you never showed it a wheelchair it
cannot detect one.

A **foundation model** is trained on massive internet-scale data — hundreds of millions to
billions of images — and develops a general visual understanding of the world:

```
Billions of images + text descriptions
           │
           ▼
    [ Giant neural network ]
    learns general visual
    understanding of everything
           │
           ▼
    Can be asked to do many
    things at inference time
    without retraining
```

The two foundation models in FOMO-3D:

**OWLv2** is a vision-language foundation model trained on image-text pairs. You give it any
text prompt at runtime — `"construction worker"`, `"wheelchair"` — and it finds that thing in
the image, even if the driving dataset it is deployed on contains zero examples of it. It has
seen wheelchairs in millions of internet photos: medical websites, sports events, accessibility
blogs. That knowledge is already inside it.

**Metric3Dv2** is a depth estimation foundation model trained on depth data from many camera
types and environments. It produces a per-pixel depth estimate in real metres from any camera
image. The "Metric" means real-world units — not just "A is closer than B" but "A is 3.2 m
away."

FOMO-3D does not retrain either model. It uses them as-is at inference time. The knowledge was
already there — FOMO-3D provides the pipeline to act on it in 3D space.

---

### 4.5 What is a Frustum?

A frustum is the volume of 3D space that projects onto a rectangular region in an image.

Every pixel in the image corresponds to a ray shooting out from the camera into the world. A
rectangular box of pixels is a bundle of rays. The 3D volume enclosed by those rays is the
frustum — a pyramid shape with the tip at the camera and the base far away:

```
         Camera
           ●
          /│\
         / │ \
        /  │  \
       /   │   \
      ┌────┼────┐   ← near face
      │    │    │
      └────┼────┘
            \  │  /
             \ │ /
        ┌─────┼─────┐   ← far face (wider, further away)
        │     │     │
        └─────┼─────┘
```

The crucial property: **anything that appears inside a 2D bounding box in the image must
physically exist somewhere inside the frustum in 3D.** You do not know where along the depth
axis — but you know it is in that cone.

So when OWLv2 says "construction worker is in this 2D box", you immediately know the person is
somewhere inside the frustum behind that box. You just need depth to pin down exactly where.

```
OWLv2 2D box
┌───────────┐
│           │         frustum in 3D
│   PERSON  │    ●───────────────────[  ?  ]
│           │         person is somewhere in here
└───────────┘
```

---

### 4.6 What is Metric3Dv2?

Metric3Dv2 takes a single camera image and outputs a depth value for every pixel — how far
away is the surface at that pixel, in real metres:

```
Input:                    Output (depth map):
RGB image                 same spatial size, float values

┌──────────────┐          ┌──────────────┐
│   🚗  🚶     │          │ 2m  2m 15m   │
│              │  ──────► │              │
│         🚧   │          │         8m   │
└──────────────┘          └──────────────┘
```

How does it know depth from a flat image? The same way humans do — learned visual cues: objects
further away appear smaller, perspective lines converge, texture gets finer at distance, known
object sizes imply distance. Metric3Dv2 learned all of these cues from enormous amounts of
depth training data across many cameras and scenes.

The depth map answers the question left open by the frustum: given that the construction worker
is somewhere inside this frustum cone, the depth at those pixels tells us approximately how far
along the cone they are. That pins down the 3D position.

---

### 4.7 How Stage 1 and Stage 2 Work in Plain English

**Stage 1 — Cast a wide net**

Two branches run in parallel and independently propose where objects might be:

- The **LiDAR branch** processes the point cloud through a neural network and generates 3D box
  proposals. It is very accurate for common objects that have many LiDAR returns (cars, trucks)
  but misses rare objects with sparse returns.

- The **camera branch** runs OWLv2 on the image to find 2D boxes, runs Metric3Dv2 to get
  depth, then uses the frustum geometry to lift each 2D box into a 3D proposal. It excels at
  rare objects that OWLv2 recognises from internet-scale training but LiDAR barely touches.

Both sets of proposals are merged and **NMS** (Non-Maximum Suppression) removes duplicates. If
both branches detected the same car, only the higher-confidence proposal survives. If only the
camera branch detected a wheelchair, that proposal passes through untouched.

**Stage 2 — Sanity check and sharpen**

Every surviving proposal — regardless of which branch it came from — is cross-checked against
the OWLv2 image feature map:

1. The 3D box centre is projected back into the image to find which pixel it corresponds to
2. The image feature map is queried at that location — what does the image say is there?
3. If the image agrees with the proposal, confidence goes up and the box position is tightened
4. If the image disagrees, confidence drops and the proposal is discarded

This is why even common LiDAR detections improve: the image provides a second opinion that
refines the box boundaries and confirms the class label.

---

### 4.8 OWLv2's Dual Role

OWLv2 runs **once** per frame and produces two outputs simultaneously:

```
Camera image
     │
     ▼
  [ OWLv2 ]
     │
     ├──► Output 1: 2D bounding boxes + class scores     (used in Stage 1)
     │              "construction worker at pixels (340,120)→(390,280)"
     │
     └──► Output 2: Dense image feature map              (used in Stage 2)
                    a grid of rich semantic vectors
                    describing what the image looks like everywhere
```

**In Stage 1** — the 2D boxes trigger the camera branch. Each box defines a frustum; the
frustum lift produces a 3D proposal. OWLv2's role: *find where objects are in the image.*

**In Stage 2** — the feature map (kept in memory from Stage 1, no second inference) is queried
by every merged proposal to confirm or deny it. OWLv2's role: *provide rich semantic evidence
to confirm proposals and suppress false positives.*

An important subtlety: Stage 2 is not simply rubber-stamping what OWLv2 already decided in
Stage 1. The feature map is the raw internal representation before OWLv2 made any detection
decision — it just describes the image. And the 3D-to-2D reprojection may land on a slightly
different pixel than the original 2D detection, especially if the frustum lift placed the box
at the wrong depth. Stage 2 can and does suppress proposals that OWLv2 weakly detected in
Stage 1 if the reprojected location does not confirm them.

---

## 5. Mathematical Background

### Notation Legend

| Symbol | Meaning |
|--------|---------|
| $K \in \mathbb{R}^{3\times3}$ | Camera intrinsic matrix |
| $f_x, f_y$ | Focal lengths in pixels |
| $(c_x, c_y)$ | Principal point in pixels |
| $(u, v)$ | 2D pixel coordinates |
| $z$ | Depth along the camera's optical axis (metres) |
| $\mathbf{p}_\text{cam} \in \mathbb{R}^3$ | 3D point in camera frame |
| $\mathbf{p}_\text{lidar} \in \mathbb{R}^3$ | 3D point in LiDAR frame |
| $T_{\text{lidar}\leftarrow\text{cam}} \in SE(3)$ | Extrinsic: transforms camera-frame points into LiDAR frame |
| $\mathbf{b} = (b_x, b_y, b_z, b_w, b_l, b_h, b_\theta)$ | 3D bounding box: centre, dimensions, yaw |
| $\text{IoU}_{3D}$ | Volumetric intersection-over-union of two 3D boxes |

---

### 4.1 Camera Projection Model

A 3D point $\mathbf{p}_\text{cam} = [X, Y, Z]^\top$ in the camera frame projects to pixel
$(u, v)$ via the standard pinhole model:

$$u = f_x \frac{X}{Z} + c_x, \qquad v = f_y \frac{Y}{Z} + c_y$$

In homogeneous form:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}, \qquad K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

The **inverse** (unprojection) recovers a 3D ray from a pixel given a depth $z$:

$$X = \frac{(u - c_x) \cdot z}{f_x}, \qquad Y = \frac{(v - c_y) \cdot z}{f_y}, \qquad Z = z$$

This is the core operation of the frustum lift ([§6.3](#63-frustum-lift-to-3d)).

---

### 4.2 Camera-LiDAR Extrinsic Transform

The camera and LiDAR are mounted at different positions and orientations on the vehicle. Their
relative pose — the **extrinsic calibration** — is a rigid body transform $T_{\text{lidar}\leftarrow\text{cam}} \in SE(3)$:

$$\mathbf{p}_\text{lidar} = R_{\text{lc}} \, \mathbf{p}_\text{cam} + \mathbf{t}_{\text{lc}}$$

where $R_{\text{lc}} \in SO(3)$ is a rotation matrix and $\mathbf{t}_{\text{lc}} \in \mathbb{R}^3$
is a translation vector. In homogeneous form:

$$\begin{bmatrix} \mathbf{p}_\text{lidar} \\ 1 \end{bmatrix} = T_{\text{lidar}\leftarrow\text{cam}} \begin{bmatrix} \mathbf{p}_\text{cam} \\ 1 \end{bmatrix}$$

In this package the extrinsic is read from the TF tree at the `/camera_optical_frame` →
`/lidar_frame` transform. All 3D points from the camera branch are expressed in the LiDAR
frame before any fusion.

---

### 4.3 Frustum Definition

Given a 2D bounding box with pixel corners $(u_\text{min}, v_\text{min})$,
$(u_\text{max}, v_\text{max})$, the **frustum** is the unbounded pyramidal volume in camera
space whose cross-section at depth $z$ spans:

$$X \in \left[\frac{(u_\text{min} - c_x) \cdot z}{f_x},\ \frac{(u_\text{max} - c_x) \cdot z}{f_x}\right]$$

$$Y \in \left[\frac{(v_\text{min} - c_y) \cdot z}{f_y},\ \frac{(v_\text{max} - c_y) \cdot z}{f_y}\right]$$

A LiDAR point $\mathbf{p}_\text{lidar}$ falls **inside** the frustum if and only if its
projection onto the image plane lands inside the 2D box:

$$\mathbf{p}_\text{cam} = T_{\text{cam}\leftarrow\text{lidar}} \, \mathbf{p}_\text{lidar}$$

$$u_\text{min} \le f_x \frac{X_\text{cam}}{Z_\text{cam}} + c_x \le u_\text{max}$$

$$v_\text{min} \le f_y \frac{Y_\text{cam}}{Z_\text{cam}} + c_y \le v_\text{max}$$

and $Z_\text{cam} > 0$ (the point is in front of the camera).

This test is $O(N)$ per box over the LiDAR cloud and requires no data structure beyond the
raw point array.

---

## 6. LiDAR Branch

The LiDAR branch follows the standard two-stage LiDAR detection pipeline. It provides accurate
geometry for common, well-represented object classes. Its proposals are combined with the camera
branch proposals in Stage 2.

### 5.1 Point Cloud Preprocessing

The raw `sensor_msgs/PointCloud2` message is filtered and normalised before the backbone sees it:

1. **Range filter** — remove points closer than `min_range` (default 0.5 m, removes the vehicle
   body) and farther than `max_range` (default 80 m, noise-dominated at long range)
2. **Height filter** — remove ground returns below `ground_z` (default −2.0 m) and sky returns
   above `max_z` (default 4.0 m)
3. **Intensity normalisation** — scale raw intensity values to $[0, 1]$
4. **Shuffle** — randomly shuffle points to avoid spatial ordering bias in the backbone

Output: a point list $\mathcal{P} = \{(x_i, y_i, z_i, r_i)\}$ of at most `max_points` points
(default 120 000), each point represented as a 4-vector $(x, y, z, \text{reflectivity})$.

### 5.2 PointPillars Backbone

This package uses the **PointPillars** backbone (Lang et al., CVPR 2019) via [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
as the LiDAR feature extractor. PointPillars was chosen because:

- It runs at real-time speed (CPU-friendly voxelisation, no 3D convolutions)
- Pre-trained weights on nuScenes are publicly available
- OpenPCDet provides a clean Python API that decouples the backbone from the detection head

**How PointPillars works:**

Divide the $(x, y)$ bird's-eye-view plane into a regular grid of *pillars* — columns of infinite
height with footprint $(\Delta x, \Delta y)$ (default 0.16 m × 0.16 m). For each occupied pillar
collect all the points inside it (up to `max_pts_per_pillar = 32`). Represent each point by a
9-element feature vector:

$$\mathbf{f}_i = [x_i,\ y_i,\ z_i,\ r_i,\ x_i - \bar{x}_p,\ y_i - \bar{y}_p,\ z_i - \bar{z}_p,\ x_i - x_p^c,\ y_i - y_p^c]$$

where $(\bar{x}_p, \bar{y}_p, \bar{z}_p)$ is the mean of all points in the pillar and
$(x_p^c, y_p^c)$ is the pillar centre in $(x, y)$.

A small PointNet-style MLP maps each point feature to a 64-dimensional embedding. The embeddings
within each pillar are max-pooled to produce a single 64-d pillar feature. These features are
scattered back into a pseudo-image of shape $[64, H, W]$ where $H, W$ are the BEV grid
dimensions. A standard 2D CNN (the backbone proper) produces a BEV feature map of shape
$[384, H/2, W/2]$.

### 5.3 LiDAR 3D Proposals

Detection heads operating on the BEV feature map produce axis-aligned 3D anchor boxes at
multiple scales. For each anchor the head predicts:

- A class score vector $\mathbf{s} \in \mathbb{R}^C$
- A box regression $\Delta\mathbf{b} = (\delta x, \delta y, \delta z, \delta l, \delta w, \delta h, \delta\theta)$

The anchor is decoded to an absolute 3D box and assigned the anchor's class label at the
argmax of $\mathbf{s}$. All boxes with score above `lidar_score_thresh` (default 0.3) become
LiDAR branch proposals.

Output: a set of LiDAR proposals
$\mathcal{B}_L = \{(\mathbf{b}_i, c_i, s_i)\}$ where $\mathbf{b}_i$ is a 3D box,
$c_i$ is a class index, and $s_i \in [0, 1]$ is a confidence score.

---

## 7. Camera Branch — The Novel Contribution

The camera branch is what distinguishes FOMO-3D from all prior work. It addresses the class
imbalance problem by querying OWLv2 — a vision-language model trained on internet-scale data —
with open-vocabulary text prompts, then lifting its 2D detections into 3D using depth and LiDAR
geometry. The camera branch excels exactly where the LiDAR branch fails: rare objects with few
LiDAR returns, small objects at long range, and visually distinctive objects (high-visibility
vests, wheelchair frames) that are better characterised by appearance than geometry.

### 6.1 OWLv2 Open-Vocabulary 2D Detection

**OWLv2** (Minderer et al., 2023) is a transformer-based open-vocabulary object detector built
on a CLIP-style vision-language backbone. Given an image $I$ and a set of text prompts
$\mathcal{T} = \{t_1, \ldots, t_M\}$, it outputs a set of 2D bounding boxes each paired with
a text similarity score for every prompt:

$$\text{OWLv2}(I, \mathcal{T}) \to \{(b_j^{2D}, \mathbf{s}_j)\}$$

where $b_j^{2D} = (u_\text{min}, v_\text{min}, u_\text{max}, v_\text{max})$ and
$\mathbf{s}_j \in \mathbb{R}^M$ is the similarity score vector across all prompts.

The key property that matters for FOMO-3D: OWLv2 also exposes a **dense image feature map**
$\mathbf{F}_\text{img} \in \mathbb{R}^{H'\times W'\times D}$ from the vision encoder before the
detection head. This feature map is used again in the cross-attention refinement step
([§8](#8-detr-style-cross-attention-refinement)).

Text prompts are specified in `config/params.yaml` and are fully configurable at launch time
without retraining. For the default autonomous driving context the prompts are:

```yaml
text_prompts:
  - "car"
  - "truck"
  - "pedestrian"
  - "cyclist"
  - "construction worker"
  - "wheelchair"
  - "traffic cone"
  - "barrier"
  - "stroller"
```

Only detections with OWLv2 score above `owl_score_thresh` (default 0.15) are passed to the
frustum lift. The lower threshold compared to the LiDAR branch is intentional — the camera
branch is expected to produce noisier proposals that the refinement step will clean up.

OWLv2 is loaded from HuggingFace:
```python
from transformers import Owlv2Processor, Owlv2ForObjectDetection
```

### 6.2 Metric3Dv2 Monocular Depth

**Metric3Dv2** (Hu et al., CVPR 2024) is a zero-shot monocular depth estimator. Given an image
$I$ and optionally the camera focal length, it produces a dense depth map
$D \in \mathbb{R}^{H\times W}$ where $D[v, u] = z$ is the estimated metric depth (in metres)
at pixel $(u, v)$ along the camera's optical axis.

"Metric" means the output is calibrated to real-world scale — not just relative depth ordering.
This is what makes it useful for lifting 2D detections to 3D: we need a real distance estimate,
not an ordinal depth ranking.

Metric3Dv2 is loaded from HuggingFace:
```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
```

Both OWLv2 and Metric3Dv2 run in a single forward pass on the same image each time step. Their
outputs are independent and can be parallelised across GPU streams if available.

### 6.3 Frustum Lift to 3D

The frustum lift converts each 2D bounding box from OWLv2 into a 3D box hypothesis using the
Metric3Dv2 depth map and nearby LiDAR points.

**Step 1 — Sample depth at box.** For each OWLv2 2D box $b_j^{2D}$, extract the depth values
from $D$ at all pixels within the box. Compute the **median** depth:

$$z_j = \text{median}\{D[v, u] \mid u \in [u_\text{min}, u_\text{max}],\ v \in [v_\text{min}, v_\text{max}]\}$$

The median is preferred over the mean because depth maps produced by monocular networks often
contain background pixels at the edges of a bounding box that would inflate the mean. The median
is robust to these outliers.

**Step 2 — Unproject box centre.** The 2D box centre $(u_c, v_c) = ((u_\text{min}+u_\text{max})/2,\ (v_\text{min}+v_\text{max})/2)$
is unprojected to 3D using $z_j$ and the camera intrinsics $K$:

$$X_c = \frac{(u_c - c_x) \cdot z_j}{f_x}, \quad Y_c = \frac{(v_c - c_y) \cdot z_j}{f_y}, \quad Z_c = z_j$$

This gives the 3D centre of the object in camera frame: $\mathbf{p}_c^\text{cam} = [X_c, Y_c, Z_c]^\top$.

**Step 3 — Transform to LiDAR frame.**

$$\mathbf{p}_c^\text{lidar} = T_{\text{lidar}\leftarrow\text{cam}} \begin{bmatrix}\mathbf{p}_c^\text{cam} \\ 1\end{bmatrix}_{[1:3]}$$

**Step 4 — Collect frustum LiDAR points.** Filter all LiDAR points $\mathbf{p}_i \in \mathcal{P}$
whose projection lands inside the 2D box (the frustum membership test from [§4.3](#43-frustum-definition)).
Let $\mathcal{F}_j = \{\mathbf{p}_i \in \mathcal{P} : \mathbf{p}_i \text{ inside frustum of } b_j^{2D}\}$.

**Step 5 — Estimate 3D box.** If $|\mathcal{F}_j| \ge$ `min_frustum_pts` (default 3):

- 3D box centre $\mathbf{b}_c = \text{mean}(\mathcal{F}_j)$ (if enough LiDAR points present),
  otherwise use the depth-unprojected centre $\mathbf{p}_c^\text{lidar}$
- Dimensions $(l, w, h)$: fit an axis-aligned bounding box to $\mathcal{F}_j$, with a minimum
  size prior from the class label (e.g. pedestrian: $0.8 \times 0.8 \times 1.8$ m)
- Yaw $\theta$: initialised to 0 (upright box); refined in the cross-attention step

If $|\mathcal{F}_j| <$ `min_frustum_pts`, the depth-only centre is used with a class-size prior
for dimensions and zero yaw. These proposals have lower confidence and are gated by a stricter
threshold in the refinement step.

Output: camera branch proposals $\mathcal{B}_C = \{(\mathbf{b}_j, c_j, s_j)\}$ in LiDAR frame.

### 6.4 Frustum-Based Fusion Module

The paper introduces a **frustum-based fusion module** that combines the Metric3Dv2 depth
estimate with the LiDAR frustum points more carefully than a simple average.

For each frustum, the depth map provides a dense but noisy depth estimate; the LiDAR provides
sparse but accurate depth measurements. The fusion module computes a weighted depth estimate:

$$z^* = \frac{w_L \cdot \bar{z}_L + w_D \cdot z_D}{w_L + w_D}$$

where:

- $\bar{z}_L = \text{median}\{Z^\text{cam}_i : \mathbf{p}_i \in \mathcal{F}_j\}$ — median LiDAR
  depth inside the frustum (in camera frame), available when $|\mathcal{F}_j| \ge$ `min_frustum_pts`
- $z_D$ — Metric3Dv2 median depth from Step 1
- $w_L = |\mathcal{F}_j|$ — number of LiDAR points (more points = more trust in LiDAR)
- $w_D =$ `depth_prior_weight` (default 2.0) — fixed trust in the depth map

When no LiDAR points fall inside the frustum ($w_L = 0$), the depth-only estimate is used.
When many LiDAR points are present, the LiDAR dominates.

---

## 8. Proposal Fusion — NMS Merge

At the end of Stage 1 we have two proposal sets from independent branches, both expressed in
the LiDAR frame:

$$\mathcal{B} = \mathcal{B}_L \cup \mathcal{B}_C$$

These proposals may overlap — the same physical object could be detected by both the LiDAR
backbone (as a common object) and the OWLv2 camera branch (as a specific rare subclass). The
**3D Non-Maximum Suppression (NMS)** step removes these duplicates.

**3D IoU.** Two axis-aligned 3D boxes are compared by their volumetric intersection-over-union:

$$\text{IoU}_{3D}(\mathbf{b}_1, \mathbf{b}_2) = \frac{V_\text{intersection}}{V_1 + V_2 - V_\text{intersection}}$$

The intersection volume is computed analytically for axis-aligned boxes:

$$V_\text{intersection} = \max(0, \Delta x) \cdot \max(0, \Delta y) \cdot \max(0, \Delta z)$$

where $\Delta x = \min(x_1^\text{max}, x_2^\text{max}) - \max(x_1^\text{min}, x_2^\text{min})$,
and likewise for $y$, $z$.

**NMS algorithm.** Sort all proposals in $\mathcal{B}$ by descending confidence score. Iterate:
accept the top-scoring proposal, reject all remaining proposals with $\text{IoU}_{3D} >
\tau_\text{NMS}$ (default 0.5) against it, then repeat with the next surviving proposal.

This is the same algorithm as 2D NMS, extended to 3D volumes. It is implemented in
`fusion.py:nms_3d()`.

Output: merged proposals $\mathcal{B}^* \subset \mathcal{B}$, typically a much smaller set
with no overlapping boxes.

---

## 9. DETR-Style Cross-Attention Refinement

Stage 2 refines each proposal in $\mathcal{B}^*$ by attending back to the dense image features
from OWLv2. The intuition: after the frustum lift, the 3D box is coarsely positioned, but the
image feature map carries fine-grained semantic and spatial information that can sharpen the
box boundaries and confirm the class label.

**Query construction.** Each proposal $(\mathbf{b}_i, c_i, s_i)$ becomes a query vector
$\mathbf{q}_i \in \mathbb{R}^D$ by encoding the 3D box parameters:

$$\mathbf{q}_i = \text{MLP}_\text{enc}([\mathbf{b}_i;\ \text{one-hot}(c_i)])$$

**Key-value construction.** The OWLv2 image feature map $\mathbf{F}_\text{img}$ is flattened
spatially to a sequence of $H' \cdot W'$ tokens, each of dimension $D$. These become the keys
$\mathbf{K}$ and values $\mathbf{V}$ for the attention mechanism.

**Spatial masking.** To focus attention on the relevant image region, each query only attends
to feature tokens that fall within a projected window: the 3D box centre $\mathbf{p}_c^\text{lidar}$
is projected into the image (via $T_{\text{cam}\leftarrow\text{lidar}}$ and $K$), and a spatial
mask of radius `attn_radius` (default 32 pixels) is applied around this projected centre.

**Cross-attention** (see [Appendix C](#appendix-c--cross-attention-formulation)):

$$\mathbf{q}_i' = \text{CrossAttention}(\mathbf{q}_i,\ \mathbf{K}_\text{masked},\ \mathbf{V}_\text{masked})$$

**Regression and classification heads.** Two small MLPs operating on $\mathbf{q}_i'$ output:

- Box refinement $\Delta\mathbf{b}_i \in \mathbb{R}^7$ — small correction to the proposal box
- Refined class scores $\tilde{\mathbf{s}}_i \in \mathbb{R}^C$ — updated confidence per class

The final box is $\mathbf{b}_i^* = \mathbf{b}_i \oplus \Delta\mathbf{b}_i$ and the final score
is $\tilde{s}_i = \max(\tilde{\mathbf{s}}_i)$.

Detections with $\tilde{s}_i <$ `final_score_thresh` (default 0.4) are discarded.

**Note on training.** The cross-attention refinement module contains learned weights and must
be trained on a labelled 3D detection dataset. Pre-trained weights from nuScenes are provided.
The OWLv2 and Metric3Dv2 backbones are frozen — only the frustum fusion module and cross-attention
refinement head are trained end-to-end.

---

## 10. ROS2 Node Architecture

### 9.1 Node I/O

```
Subscriptions
─────────────
/lidar/points          sensor_msgs/PointCloud2          LiDAR point cloud
/camera/image_raw      sensor_msgs/Image                Camera image (RGB8)
/camera/camera_info    sensor_msgs/CameraInfo           Intrinsics + distortion

Publications
────────────
/detections/objects    vision_msgs/Detection3DArray     Final 3D detections
/detections/markers    visualization_msgs/MarkerArray   RViz bounding boxes
/detections/depth_map  sensor_msgs/Image                Metric3Dv2 depth (float32, optional)
/detections/owl_image  sensor_msgs/Image                OWLv2 overlay image (optional, debug)
```

The extrinsic transform $T_{\text{lidar}\leftarrow\text{cam}}$ is read from the TF tree
at startup (frame IDs configured in `params.yaml`).

### 9.2 Processing Pipeline

The node uses **message synchronisation** to ensure the LiDAR scan and camera image are
captured at the same timestamp. `message_filters.ApproximateTimeSynchronizer` with a
`slop = 0.05 s` tolerance is used to pair messages.

Per synchronised message pair:

```
receive (PointCloud2, Image, CameraInfo)
         │
         ├── [GPU/CPU] LiDAR branch
         │       preprocess_cloud()
         │       lidar_backbone.forward()        # OpenPCDet PointPillars
         │       → B_L (LiDAR proposals)
         │
         ├── [GPU/CPU] Camera branch (parallel)
         │       owl_model.forward(image, prompts)    # OWLv2
         │       depth_model.forward(image)           # Metric3Dv2
         │       frustum_lift(owl_boxes, depth_map, cloud, K, T_lc)
         │       → B_C (camera proposals)
         │
         ├── nms_3d(B_L ∪ B_C, iou_thresh=0.5)
         │       → B_star (merged proposals)
         │
         ├── cross_attention_refine(B_star, owl_features, K, T_lc)
         │       → detections (refined 3D boxes + scores)
         │
         └── publish(Detection3DArray, MarkerArray)
```

### 9.3 Topic Reference

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/lidar/points` | `PointCloud2` | Sub | Must be in LiDAR sensor frame |
| `/camera/image_raw` | `Image` | Sub | RGB8 encoding |
| `/camera/camera_info` | `CameraInfo` | Sub | Synced with image |
| `/detections/objects` | `Detection3DArray` | Pub | Boxes in LiDAR frame |
| `/detections/markers` | `MarkerArray` | Pub | RViz visualisation |
| `/detections/depth_map` | `Image` | Pub | Float32 depth, debug only |
| `/detections/owl_image` | `Image` | Pub | OWLv2 overlay, debug only |

All topic names are remappable at launch time via `params.yaml` or command-line remapping.

---

## 11. Package Structure

```
fomo3d_detection/
├── CMakeLists.txt
├── package.xml
├── setup.py
│
├── config/
│   └── params.yaml              # all tunable parameters
│
├── launch/
│   └── fomo3d.launch.py         # launch file with topic remaps
│
├── fomo3d_detection/
│   ├── __init__.py
│   │
│   ├── fomo3d_node.py           # ROS2 node: subscription, sync, publish
│   │
│   ├── lidar_branch.py          # preprocess_cloud(), lidar_backbone wrapper
│   ├── camera_branch.py         # OWLv2 + Metric3Dv2 loading + inference
│   ├── frustum_lift.py          # frustum membership test, depth fusion, box fitting
│   ├── fusion.py                # nms_3d(), 3D IoU calculation
│   └── refinement.py            # cross-attention refinement head (PyTorch Module)
│
├── scripts/
│   └── fomo3d_node              # executable entry point (calls fomo3d_node.py)
│
└── weights/
    └── refinement_nuscenes.pt   # pre-trained cross-attention refinement weights
```

---

## 12. Dependencies

### ROS2 Packages

```
ros-humble-desktop
ros-humble-vision-msgs
ros-humble-sensor-msgs
ros-humble-geometry-msgs
ros-humble-tf2-ros
ros-humble-cv-bridge
ros-humble-message-filters
python3-colcon-common-extensions
```

### Python Packages

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.1 | PyTorch — cross-attention refinement, GPU inference |
| `transformers` | ≥ 4.38 | HuggingFace — OWLv2, Metric3Dv2 model loading |
| `numpy` | ≥ 1.24 | Array operations |
| `opencv-python` | ≥ 4.8 | Image decoding, debug overlays |
| `scipy` | ≥ 1.11 | Box fitting (ConvexHull for oriented boxes) |
| `open3d` | ≥ 0.17 | Optional — point cloud visualisation |

### External

**OpenPCDet** is required for the LiDAR branch PointPillars backbone. Install from source:

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
pip install -r requirements.txt
python setup.py develop
```

Pre-trained PointPillars weights for nuScenes are available from the
[OpenPCDet model zoo](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/MODEL_ZOO.md).

---

## 13. Installation

### Step 1 — Install ROS2 dependencies

```bash
sudo apt update
sudo apt install \
    ros-humble-desktop \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-message-filters \
    ros-humble-tf2-ros \
    python3-colcon-common-extensions
```

### Step 2 — Install Python dependencies

```bash
pip install torch torchvision transformers numpy opencv-python scipy
```

For GPU inference (recommended):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — Install OpenPCDet

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git ~/OpenPCDet
cd ~/OpenPCDet && pip install -r requirements.txt && python setup.py develop
```

Download the PointPillars nuScenes checkpoint:

```bash
mkdir -p ~/RoboticsTutorials/fomo3d_detection/weights
# follow the OpenPCDet model zoo instructions to download:
# pointpillar_nuscenes.pth → fomo3d_detection/weights/
```

### Step 4 — Download HuggingFace models

OWLv2 and Metric3Dv2 are downloaded automatically on first run by the `transformers` library.
To pre-download and cache them:

```bash
python3 -c "
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')
AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Base-hf')
AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Base-hf')
print('Models cached.')
"
```

### Step 5 — Build the package

```bash
cd ~/RoboticsTutorials
source /opt/ros/humble/setup.bash
colcon build --packages-select fomo3d_detection \
             --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source ~/RoboticsTutorials/install/setup.bash
```

---

## 14. Running

### Every new terminal

```bash
source /opt/ros/humble/setup.bash
source ~/RoboticsTutorials/install/setup.bash
```

### Launch the detection node

```bash
ros2 launch fomo3d_detection fomo3d.launch.py
```

The node will:
1. Load OWLv2 and Metric3Dv2 from the HuggingFace cache (~30 s on first run, ~5 s thereafter)
2. Load the PointPillars checkpoint from `weights/`
3. Load the cross-attention refinement weights from `weights/refinement_nuscenes.pt`
4. Subscribe to `/lidar/points`, `/camera/image_raw`, `/camera/camera_info`
5. Begin publishing detections at the rate of the synchronised input stream

### With topic remapping

```bash
ros2 launch fomo3d_detection fomo3d.launch.py \
    lidar_topic:=/velodyne/points \
    camera_topic:=/zed/left/image_rect_color \
    camera_info_topic:=/zed/left/camera_info
```

### Visualise in RViz

```bash
rviz2 -d ~/RoboticsTutorials/fomo3d_detection/config/fomo3d_viz.rviz
```

Add the following displays:
- `PointCloud2` → `/lidar/points` — the raw LiDAR scan
- `Detection3D` (via `vision_msgs`) → `/detections/objects` — the 3D boxes
- `MarkerArray` → `/detections/markers` — labelled bounding boxes with class names

### Monitor detection rate

```bash
ros2 topic hz /detections/objects
```

Expected throughput with a GPU: ~5–10 Hz (dominated by OWLv2 inference).
CPU-only: ~1–2 Hz.

---

## 15. Configuration

All parameters are in `config/params.yaml`:

```yaml
fomo3d:
  # Text prompts for OWLv2 open-vocabulary detection
  text_prompts:
    - "car"
    - "truck"
    - "pedestrian"
    - "cyclist"
    - "construction worker"
    - "wheelchair"
    - "traffic cone"
    - "barrier"

  # Frame IDs
  lidar_frame: "lidar_frame"
  camera_frame: "camera_optical_frame"

  # Input topic names
  lidar_topic: "/lidar/points"
  camera_topic: "/camera/image_raw"
  camera_info_topic: "/camera/camera_info"

  # Sync tolerance (seconds)
  sync_slop: 0.05

  # LiDAR preprocessing
  min_range: 0.5          # metres
  max_range: 80.0
  ground_z: -2.0
  max_z: 4.0
  max_points: 120000

  # LiDAR branch thresholds
  lidar_score_thresh: 0.30

  # Camera branch thresholds
  owl_score_thresh: 0.15

  # Frustum lift
  min_frustum_pts: 3      # minimum LiDAR points inside frustum to trust LiDAR geometry
  depth_prior_weight: 2.0 # trust weight of Metric3Dv2 depth vs LiDAR depth

  # NMS
  nms_iou_thresh: 0.50

  # Cross-attention refinement
  attn_radius: 32         # pixel radius for spatial attention mask
  final_score_thresh: 0.40

  # Debug outputs (expensive — disable for performance)
  publish_depth_map: false
  publish_owl_overlay: false

  # Weights paths
  lidar_weights: "weights/pointpillar_nuscenes.pth"
  refinement_weights: "weights/refinement_nuscenes.pt"
```

---

## 16. Tuning Guide

### OWLv2 Score Threshold (`owl_score_thresh`)

Controls the recall/precision tradeoff of the camera branch.

- **Lower threshold** → more proposals, higher recall on rare objects, more false positives
  entering the refinement step. The cross-attention refiner is expected to suppress most FPs.
- **Higher threshold** → fewer proposals, lower recall, fewer FPs. Safer if the refiner weights
  are poorly matched to your domain.

Start at 0.15. If you see many false positive detections in the final output, raise to 0.25.
If you are missing rare objects that are clearly visible in the camera image, lower to 0.10.

### LiDAR Score Threshold (`lidar_score_thresh`)

Same tradeoff on the LiDAR branch. At 0.30, the PointPillars head returns only high-confidence
proposals. Lowering to 0.20 recovers more detections at long range but increases NMS load.

### Final Score Threshold (`final_score_thresh`)

This is the gate on the refined detections — the only threshold the end user sees. Setting it
below 0.30 typically produces more false positives than are useful. Above 0.60 starts dropping
valid rare-object detections.

### `min_frustum_pts`

If your LiDAR has low point density at long range (e.g. 16-beam scanner), lower this to 1 or
even 0 (depth-only mode). For a 64-beam scanner at normal ranges, keep at 3–5.

### Sync Slop (`sync_slop`)

For a hardware-synced sensor rig (trigger-based) set this to 0.01 s. For unsynchronised
sensors with separate clocks, 0.05–0.10 s is safe. Beyond 0.10 s, moving objects will produce
projection mismatches between the LiDAR scan and the camera image.

### Common Issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Node starts but no detections published | Topic names don't match | Check `ros2 topic list`, update `params.yaml` |
| Only LiDAR detections, no camera branch outputs | OWLv2 score too high, or camera/LiDAR timestamps not syncing | Lower `owl_score_thresh`; check `sync_slop` |
| High latency (>500 ms per frame) | Running OWLv2 on CPU | Install CUDA PyTorch; add `device: "cuda"` to params |
| Many ghost detections on reflective surfaces | LiDAR branch threshold too low | Raise `lidar_score_thresh` to 0.40 |
| Correct object class from OWL but wrong 3D position | Extrinsic calibration wrong | Verify `T_lidar_cam` by projecting LiDAR points onto image and checking alignment |
| Depth scale wrong (boxes too near or too far) | Metric3Dv2 scale mismatch for your camera focal length | Pass `--focal_length_px` to the depth model; see Metric3Dv2 docs |

---

## 17. References

\[1\] A. J. Yang, J. Tu, N. Dvornik, E. Li, R. Urtasun,
"FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection,"
*CoRL 2025*, Proceedings of Machine Learning Research vol. 305, pp. 5526–5556.
[arXiv:2603.08611](https://arxiv.org/abs/2603.08611)

\[2\] M. Minderer, A. Gritsenko, N. Houlsby, "Scaling Open-Vocabulary Object Detection,"
*NeurIPS 2023*. (OWLv2)

\[3\] W. Hu, T. Wang, C. Chen, et al., "Metric3D v2: A Versatile Monocular Geometric Foundation
Model for Zero-shot Metric Depth and Surface Normal Estimation," *CVPR 2024*. (Metric3Dv2)

\[4\] A. H. Lang, S. Vora, H. Caesar, et al., "PointPillars: Fast Encoders for Object Detection
from Point Clouds," *CVPR 2019*. (LiDAR backbone)

\[5\] N. Carion, F. Massa, G. Synnaeve, et al., "End-to-End Object Detection with Transformers,"
*ECCV 2020*. (DETR — cross-attention detection paradigm)

\[6\] OpenPCDet Development Team, "OpenPCDet: An Open-source Toolbox for 3D Object Detection
from Point Clouds," [github.com/open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet), 2020.

---

## Appendix A — Full Frustum Lift Derivation

This appendix works through the geometry of the frustum membership test and the box fitting
step in detail.

### A.1 Projecting a LiDAR point into the image

Given a LiDAR point $\mathbf{p}^\text{lidar} = [X_L, Y_L, Z_L]^\top$ and the extrinsic
transform $T_{\text{cam}\leftarrow\text{lidar}} = (R_\text{cl}, \mathbf{t}_\text{cl})$:

**Step 1 — Transform to camera frame:**

$$\mathbf{p}^\text{cam} = R_\text{cl} \mathbf{p}^\text{lidar} + \mathbf{t}_\text{cl} = [X_C, Y_C, Z_C]^\top$$

**Step 2 — Depth check.** If $Z_C \le 0$, the point is behind the camera — skip it.

**Step 3 — Project to pixel:**

$$u = f_x \frac{X_C}{Z_C} + c_x, \qquad v = f_y \frac{Y_C}{Z_C} + c_y$$

**Step 4 — Bounds check.** The point is inside the 2D box $b^{2D}$ if and only if:

$$u_\text{min} \le u \le u_\text{max} \quad \text{and} \quad v_\text{min} \le v \le v_\text{max}$$

In code (`frustum_lift.py`):

```python
def in_frustum(pts_lidar: np.ndarray, box2d: np.ndarray,
               K: np.ndarray, T_cam_lidar: np.ndarray) -> np.ndarray:
    # pts_lidar: (N, 3), box2d: (4,) = [u_min, v_min, u_max, v_max]
    pts_cam = (T_cam_lidar[:3, :3] @ pts_lidar.T + T_cam_lidar[:3, 3:]).T  # (N, 3)
    valid = pts_cam[:, 2] > 0.1                                               # in front of camera
    u = pts_cam[:, 0] / pts_cam[:, 2] * K[0, 0] + K[0, 2]
    v = pts_cam[:, 1] / pts_cam[:, 2] * K[1, 1] + K[1, 2]
    in_box = (u >= box2d[0]) & (u <= box2d[2]) & \
             (v >= box2d[1]) & (v <= box2d[3])
    return valid & in_box
```

### A.2 Axis-aligned box fitting

Given a set of $N$ LiDAR points $\{\mathbf{p}_i\} \in \mathbb{R}^3$ inside the frustum, the
axis-aligned 3D bounding box is:

$$x_\text{min} = \min_i p_i^x, \quad x_\text{max} = \max_i p_i^x, \quad \text{and similarly for } y, z$$

$$\text{centre} = \frac{1}{2}\begin{bmatrix}x_\text{min}+x_\text{max} \\ y_\text{min}+y_\text{max} \\ z_\text{min}+z_\text{max}\end{bmatrix}, \quad
l = x_\text{max} - x_\text{min}, \quad w = y_\text{max} - y_\text{min}, \quad h = z_\text{max} - z_\text{min}$$

A minimum size prior $(\ell_\text{min}, w_\text{min}, h_\text{min})$ is applied per class
to avoid degenerate near-zero boxes when only a few LiDAR points land in the frustum:

$$l \leftarrow \max(l, \ell_\text{min}), \quad w \leftarrow \max(w, w_\text{min}), \quad h \leftarrow \max(h, h_\text{min})$$

---

## Appendix B — NMS for 3D Boxes

Standard 2D NMS suppresses overlapping 2D boxes by IoU. The 3D version generalises
straightforwardly for axis-aligned boxes.

### B.1 3D IoU Calculation

For two axis-aligned 3D boxes $\mathbf{b}_1 = (c_1, l_1, w_1, h_1)$ and
$\mathbf{b}_2 = (c_2, l_2, w_2, h_2)$, define the half-extents:

$$[x_1^\text{min}, x_1^\text{max}] = [c_1^x - l_1/2,\ c_1^x + l_1/2], \quad \text{similarly for } y_1, z_1, x_2, y_2, z_2$$

The intersection interval along each axis:

$$\delta_x = \min(x_1^\text{max}, x_2^\text{max}) - \max(x_1^\text{min}, x_2^\text{min})$$

and similarly for $\delta_y$, $\delta_z$. If any $\delta < 0$, the boxes do not intersect.

$$V_\text{int} = \max(0, \delta_x) \cdot \max(0, \delta_y) \cdot \max(0, \delta_z)$$

$$V_1 = l_1 w_1 h_1, \qquad V_2 = l_2 w_2 h_2$$

$$\text{IoU}_{3D} = \frac{V_\text{int}}{V_1 + V_2 - V_\text{int}}$$

### B.2 NMS Algorithm

```python
def nms_3d(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = np.array([iou_3d(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][ious <= iou_thresh]
    return keep
```

Time complexity: $O(N^2)$ where $N = |\mathcal{B}|$. For typical proposal counts
($N \le 200$) this is negligible at inference time.

---

## Appendix C — Cross-Attention Formulation

The refinement head uses a single-head cross-attention layer. This appendix gives the full
matrix formulation.

### C.1 Attention Computation

Let $\mathbf{q}_i \in \mathbb{R}^D$ be the encoded query (proposal $i$) and let
$\mathbf{K}_\text{mask} \in \mathbb{R}^{M\times D}$ and $\mathbf{V}_\text{mask} \in \mathbb{R}^{M\times D}$
be the $M$ spatially-masked image feature tokens (keys and values):

$$\text{Attention output} = \text{softmax}\!\left(\frac{\mathbf{q}_i \mathbf{K}_\text{mask}^\top}{\sqrt{D}}\right) \mathbf{V}_\text{mask} \in \mathbb{R}^D$$

The $\sqrt{D}$ scaling prevents the dot products from growing large in magnitude as $D$
increases, keeping the softmax from saturating (Vaswani et al. 2017, §3.2.1).

### C.2 Spatial Masking

Before computing attention, tokens outside a circular window of radius `attn_radius` pixels
around the projected box centre are masked to $-\infty$ before the softmax, effectively
zeroing their attention weight. This focuses the model on the relevant image region and reduces
computation from $H'\cdot W'$ tokens to the $O(\text{attn\_radius}^2)$ unmasked tokens.

### C.3 MLP Heads

```
query_i' = LayerNorm(query_i + Attention_output)
         → MLP(256 → 256 → 7)  : box regression Δb
         → MLP(256 → 256 → C)  : class logits
```

The layer norm and residual connection follow the standard Transformer encoder block
(Vaswani et al. 2017) and stabilise training.
