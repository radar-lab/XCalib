# Public A9 Demo

This folder is a small standalone demo for the accepted-paper public release.
It installs `xcalib` editable from the repo source (so the env always tracks
`src/xcalib`), a tiny committed A9-style sample under `frames/a9_sample/`, and the
released A9 model weights through `xcalib`.

The committed sample is for step-by-step usage walkthroughs. For realistic
paper-scale runs, place `data/a9_test.h5` locally or let `xcalib.load_dataset()`
resolve the released Hugging Face dataset cache.

The sample frames come from the TUM Traffic (A9) dataset, which carries its own
CC BY-NC-ND 4.0 terms (see the root `README.md`).

## Homography notebook: from pairs to the homography, without an extrinsic

`xcalib_homography_en.ipynb` takes a raw delivery to the homography
`[u, v, 1] ~ H @ [position_x, position_y, 1]`, which maps the
`position_x, position_y` of a row of the delivered LiDAR table, used as
delivered, to the pixel of that object's centre. It never asks for an
extrinsic: `H` is fitted to saved pairs, each a camera box and the LiDAR box of
the same object. The pairs come from three sources: a folder of pairs saved
earlier, the pairing window, and a model trained on the saved pairs, whose
suggestions are checked in the window and saved with **Apply**.

From this folder, run:

```bash
pixi install
pixi run notebook-homography
```

The task opens the notebook in JupyterLab. It first runs `pixi run kernel`,
which registers the notebook's kernel "XCalib demo (.pixi)" inside this
environment (nothing outside it changes). Run the cells one at a time, from
the top. The first cell creates
`flow = Workflow(base_epochs=30, refine_epochs=10)` from `xcalib.gui.customer`,
with the epochs of the base training and of the fine-tuning. One epoch each
only checks that everything runs; it is not a trained-model quality claim.
Every later cell is one call on `flow`, and every folder or file is chosen in a
dialog. A cell that opens the pairing window waits until the window is closed
and then reports what was saved. The dialogs and the window are Qt and need a
local display.

The delivery must be rectified already: undistorted images, and camera
detections made on them. Two delivery layouts are recognized: `cam_sync/`,
`LiDAR_529_sync/`, `Camera_perception/tracks_detailed.csv` and
`LiDAR_perception.csv`, or `img_rect/`, `pcd/`, `camera_perception.txt` (MOT
text) and `perception.csv`; the import names the parts a folder lacks. A
delivery in the first layout whose images and camera boxes are still distorted
is repaired once with
`pixi run python ../scripts/undistort_raw_delivery.py --raw <delivery> --output <new folder>`.
It reads the distortion from the delivery's `undistortion_parameters.json` and
writes a rectified copy; the delivery itself is not changed.

The camera intrinsic is read from `undistortion_parameters.json` in the data
folder and saved with each scene. When the file is not there, the import asks
for it in a second dialog and takes the delivery's undistortion parameters
(`K_undistort`), the `calib/intrinsic` file of a scene, or a file with the
camera matrix alone (`intrinsic`, `camera_matrix` or `K`).

The notebook has three parts, and only the ones that are needed are run (always
after the first cell). Part 1, steps 1 to 4, trains a base model on the base
data. Part 2, steps 5 to 8, refines a base model on the refine data, other raw
data than the base data; the base model is the one of step 4 or one made
earlier, so Part 1 is not needed for it. Step 9 writes the homography of the
refine data at the end of Part 2, with the base data's pairs too when Part 1
ran in the same session: it needs the refine part, not Part 1.

The steps:

1. Import the base data: `flow.import_raw("base")` asks for a raw data folder
   and copies its synchronized images and point clouds, all frames, into one
   scene in the SUSTechPOINTS layout, the base scene, under
   `datasets/raw_workflow/` in the repository. It writes the delivery's camera
   detections as the camera boxes and its LiDAR detections as 3D boxes in the
   point-cloud frame, and shows check images. A folder imported before is not
   copied again, and a base scene imported before can be chosen in the same
   dialog.

   Optional, look at the scene in SUSTechPOINTS: `flow.view_sustech("base")`
   opens the base scene in SUSTechPOINTS in the web browser (point clouds,
   LiDAR boxes with their ids, and the camera image once the camera is chosen
   in its **Camera** menu), together with the refine scene when one is chosen.
   A LiDAR box fine-tuned there (click its name, then drag its borders in the
   three views on the left) is saved into the scene at once; please do not
   delete a box or change its id, because the saved pairs name LiDAR boxes by
   their ids. SUSTechPOINTS is a separate program; what it needs is listed in
   [Look at a scene in SUSTechPOINTS](../docs/review-gui.md#look-at-a-scene-in-sustechpoints).

   Optional, use pairs you already have: `flow.import_pairs("base")` asks for
   the scene folder of an earlier session, its `label_fusion/camera/<camera>`
   folder, or a folder of saved label files (one `<frame>.json` per frame).
   `xcalib.gui.saved_pairs.import_pairs` copies every pair and every "no LiDAR
   box" mark onto the same camera box (found by its rectangle, else by its
   track) and keeps what the scene holds already. It leaves out a pair whose
   LiDAR box the scene does not have, or whose LiDAR box another camera box of
   the frame holds already.

2. Confirm pairs: `flow.pair_for_homography("base")` opens the pairing window,
   `xcalib-review`, on the base scene.
3. View the pairs: `flow.view_pairs("base")` prints what is saved (pairs,
   frames, objects, how the pairs that the homography can use spread over the
   left, middle and right of the image) and shows check images in which a pair
   has one colour in the camera image and in the point cloud seen from above.
4. Train the pairing model: `flow.train("base")` trains a model on the saved
   pairs, in a new folder under `runs/`.
5. Import the refine data: `flow.import_raw("refine")` asks for a raw data
   folder and imports it as step 1 does: all of its frames become one scene,
   the refine scene. It should be other raw data than the base data: frames
   the base model has not seen. Data that also holds the base model's frames
   (the whole recording, for example) works as well: every step runs and step 9
   uses every saved pair, but the fine-tuning learns little that is new.
   Without `undistortion_parameters.json` in the
   folder it takes the camera intrinsic of the base scene of step 1, or asks
   for the file. A refine scene imported before can be chosen in the same
   dialog.

   Optional, look at the refine scene in SUSTechPOINTS:
   `flow.view_sustech("refine")`, as after step 1. With a base scene chosen
   too, the page holds both scenes (choose one in its scene list, then a frame
   in its frame list); a page that showed the base scene alone starts again at
   a new address. Boxes fine-tuned there are saved into their own scene.

   Optional, use a base model you already have: `flow.choose_model("base")`
   asks for the folder of a model trained before (the `model` folder of a
   training run, with `crlite.pth` and `crlite.yaml`). Steps 6 and 7 then use
   it, so steps 1 to 4 are not needed by someone who only refines.

   Optional, use pairs you already have for the refine data:
   `flow.import_pairs("refine")` copies them onto the refine scene as
   `flow.import_pairs("base")` does for the base scene. Step 6 shows them, the
   model suggests pairs only for the camera boxes without a decision, and
   steps 7 and 9 use them.

6. Confirm the model's suggested pairs: `flow.pair("refine")` puts the base
   model into the refine scene (a model that was there is kept beside it) and
   opens the pairing window with the model's suggestions marked `(proposal)`.
   Change the wrong ones; **Apply this frame** saves the frame as it is shown,
   so a suggestion left as it is counts as right.
7. Fine-tune the model: `flow.train("refine")` trains the base model further on
   the refine frames with saved pairs. The result is a new model folder; the
   base model stays as it is. To fine-tune again with other epochs, change the
   first cell and run it again (a new session: nothing on disk is lost, the
   chosen scenes and models are forgotten), then step 1 (when Part 1 ran) and
   step 5 with the same folders, `flow.choose_model("base")` with the base
   model used before (the `model` folder of step 4's run, or the model chosen
   earlier) and step 7, which names the base model it starts from. Each
   fine-tuning starts from that base model and learns from every pair saved at
   that time, the pairs applied in step 8 included.
8. Let the model pair the remaining frames: `flow.pair_remaining("refine")`
   puts the fine-tuned model into the refine scene, keeps the previous one
   beside it, and shows check images of its pairs. The pairing window then
   opens with these pairs, marked `(to confirm)`, for the camera boxes without
   a decision.
9. Write the homography: `flow.homography()`, run after Part 2, fits one
   homography to the saved pairs of the refine scene, and of the base scene
   when Part 1 ran in the same session. A homography belongs to one camera
   position and one LiDAR table, so the step says for each scene how many of
   its usable pairs agree with the fit and warns when a scene's pairs do not
   (the camera moved between the recordings). Part 1 is not needed: the refine
   pairs alone are used then, and they have to cover the left, the middle and
   the right of the image.

Training exports the frames with saved pairs: their pairs, their camera boxes
saved as `— none —` and their other LiDAR boxes. A camera box without a
decision in such a frame counts as "no LiDAR box", so decide every camera box
of a frame in which you save a pair. The model's config says
`score_mode: absolute`, so it can leave a camera box without a LiDAR box
unpaired. Training prints what it runs on, keeps one line current with the
epoch, the last validation result and the minutes left, and reports how the
kept epoch does on the validation frames. The training runs in a process of its
own; stopping its cell (the ■ button) ends that process too, as it does the
model's pairing in step 8. The pairs a model suggests come from
its scores alone: a one-to-one pick among the LiDAR boxes it scores 0.5 or
more.

Step 9 goes through `xcalib.gui.pair_homography.export_pair_homography`. It
fits `H` by RANSAC, refined by least squares, to the centres of the camera
boxes of the saved pairs (at least 12, spread over the image). Not every right
pair is a good point for it, so the pairs are prepared first. A camera box cut
by the image edge is left out, and so is one wider than 8% of the image, an
object close to the camera: the centre of such a box is not the object's
centre. An object counts once per place in the image, so that a parked object
paired in every frame is one point. And the places are taken evenly over the
image, at most 25 from each of six image columns, so that the side with more
traffic does not carry the fit alone. All pairs stay in the labels and in the
training data. The step warns when the left, middle or right third of the
image has no support. It writes to a new folder under `runs/`:

- `homography.json`: `H` (`position_to_image`), its inverse, and the checks,
  counted over the pairs whose camera box is whole and not wide: how many agree with `H`
  within 25 px, and how many have their object's dot inside the paired camera
  box.
- `projection_inputs.py`, when the scene has its camera intrinsic: the camera
  matrix, `H` and the file locations as Python assignments, for a projection
  script that reads the delivered images and table.
- `<frame>_rectified_projected.png`: camera images with a red dot and the
  LiDAR id for every object of the frame, placed through `H`.

`H` is scaled so that the third coordinate is positive in front of the camera
and negative behind it; the images leave out the objects behind the camera.
When `position_x, position_y` hold longitude and latitude, `H` needs all its
digits: take it from the files.

The notebook opens the pairing window with `--notebook-workflow`: its
**Calibrate** then takes the intrinsic saved in the scene instead of asking
for a file. The homography needs no calibration. The
[review GUI guide](../docs/review-gui.md) describes the window.

This environment installs the CUDA build of torch on Windows and Linux PCs, so
training uses an NVIDIA GPU when there is one and the CPU otherwise.

## Quickstart notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/radar-lab/XCalib/blob/main/demo/xcalib_quickstart.ipynb)

`xcalib_quickstart.ipynb` is the most direct way to see the API: install →
`Matcher.from_pretrained` (Hugging Face weights) → `load_dataset` (Hugging Face
dataset, with an offline fallback to `frames/a9_sample`) → `match()` →
`oneshot()` + `calibrate()`, with the overlays rendered inline. Sections 6 to 8 cover
what is new in 0.3: one-to-one pairs with the geometric gate, `calibrate(refine=True)`,
`SceneRecorder` with the `xcalib-review` window, and `oriented_extents`.

- **Colab:** click the badge above — it installs `xcalib` and shallow-clones this
  repo for the sample.
- **Locally:** `pixi install` here, then `pixi run notebook` to open it in
  JupyterLab — or open `xcalib_quickstart.ipynb` in VS Code and pick this `pixi`
  environment as the kernel (the env ships `jupyterlab`, `ipykernel`, and
  `matplotlib`).

The numbered scripts below are the same flow as plain, CI-friendly Python.

## Continual-refine notebook

`refine_unification.ipynb` walks the **continual** learning tier end-to-end in the `xcalib`
package: build a `Scene` and a base model from the A9 sample → `match_review` →
`record_verified_pairs` into the durable corpus → inspect it → `scene_frame_loader` →
`run_continual_refine` (gated; the notebook relaxes the gate so a handful of frames can
publish) → load the published revision. It is the worked example for accumulating
human-verified pairs into a corpus and training the matcher from it over time — the durable
complement to one-shot `adapt()`. The [continual refine guide](../docs/continual-refine.md)
explains the corpus, the gate, and published revisions.

Run it locally from a checkout. `pixi run notebook` opens the quickstart, so open
`refine_unification.ipynb` from the JupyterLab file browser. It reads `frames/a9_sample` and
takes the A9 weights from the model hub, or from
`../checkpoints/crlite_a9_dataset_r02_s01_best.pth` when that file exists. One optional cell
launches the live `xcalib-review` window (`%gui qt`) on the demo scene, so you can do the
review by hand — step frames, fix pairs, Apply — then continue in the notebook, whose
cells record the saved decisions and run the refine. Skip that cell on a headless machine.

## Review window

`xcalib-review` is a Qt window for correcting camera-LiDAR pairs on a saved scene. This
demo environment gets PySide6 as a dependency of its matplotlib package; elsewhere run `pip install "xcalib[gui]"`.

Save a scene with `SceneRecorder` (section 7 of the quickstart notebook does this with the
sample frames), then open it:

```bash
xcalib-review --scene <scene_dir> --camera <camera_name>
```

The window loads the matcher from `<scene_dir>/calib/xcalib/<camera_name>_model/` (write it
with `matcher.save_pretrained(...)`). When `<scene_dir>/calib/camera/<camera_name>.json`
exists, each candidate LiDAR box shows how far its centre projects from the camera box, and ⚠
marks one the geometry contradicts. The mark decides nothing: the suggestions come from the
matcher's scores alone.

## Steps

1. `run_00_prepare_assets.py`: report the `xcalib` install, optional local
   weights, optional local dataset cache, and ONNX outputs.
2. `run_01_match_one_frame.py`: load one committed A9 sample frame and call
   `Matcher.match()`.
3. `run_02_calibrate_stream.py`: stream A9 frames through `matcher.oneshot()`
   and solve calibration.
4. `run_03_full_demo.py`: run matching and calibration together, like a small
   edge loop.

## Install

From inside this `demo/` folder:

```bash
pixi install
```

`pixi.toml` installs `xcalib` **editable** from the repo source (`../`), so the
demo env always tracks `src/xcalib` — no wheel to rebuild. (Released installs use
the published wheel / `pip install XCalib`; the notebook's Colab cell uses pip.)

## Run

From inside `demo/`:

```bash
pixi run run-00
pixi run run-01
pixi run run-02
pixi run run-03
```

By default the demo uses the committed sample frames when
`frames/a9_sample/manifest.json` exists:

```text
frames/a9_sample/
  manifest.json
  frame_0000/image.png
  frame_0000/point_cloud.pcd
  frame_0000/detections.json
```

The same scripts fall back to HDF5 loading when the sample manifest is absent.
For full A9 runs, use:

```text
data/a9_test.h5
weights/crlite_a9_dataset_r02_s01_best.pth
onnx_export/crlite_a9_dataset_r02_s01/
```

If the local HDF5 cache is missing, the demo can resolve released Hub artifacts
after the dataset license gate is complete.

## What The Demo Calls

Matching:

```python
result = matcher.match(
    frame.image,
    frame.point_cloud,
    frame.bboxes_2d,
    frame.bboxes_3d,
)
print(result.matches)
```

Calibration:

```python
session = matcher.oneshot(intrinsics, match_threshold=0.5)
report = session.observe(image, point_cloud, bboxes_2d, bboxes_3d)
calibration = session.calibrate(min_pairs=12)
if calibration.success and calibration.accepted:
    print(calibration.projection)
```

Pairing, optionally gated by the calibration (`session.calibration` is `None` until one is
accepted):

```python
result = matcher.match(
    frame.image,
    frame.point_cloud,
    frame.bboxes_2d,
    frame.bboxes_3d,
    top_k=1,
    match_threshold=0.5,
    one_to_one=True,
    calibration=session.calibration,
)
print(result.matches)     # best LiDAR box per camera box; a LiDAR box can repeat
print(result.assignment)  # optimal pairs, each LiDAR box used at most once
```

## Intrinsics

`intrinsics_a9_example.json` holds the released A9 intrinsics for the committed sample's
camera (`s110_camera_basler_south1_8mm`):

```json
{
  "fx": 1293.5,
  "fy": 1320.2,
  "cx": 946.1,
  "cy": 622.0,
  "distortion": []
}
```

For another camera or your own rig, use its own intrinsics. With the A9 cache,
`frame.intrinsics[camera]` from `load_dataset()` carries the released `K`.

## Runtime Contract

The real-time pipeline should provide the same four arrays loaded by the demo:

- `image`: `[H, W, 3] uint8` RGB.
- `point_cloud`: `[P, >=3] float32` XYZ in meters.
- `bboxes_2d`: `[K, 4] float32`, `(x1, y1, x2, y2)` pixels.
- `bboxes_3d`: `[M, 6] float32`, extents or center-dim boxes. Pass boxes that enclose
  the object; for oriented detections use `xcalib.data.oriented_extents`.

If calibration does not solve, collect more frames or lower the demo match
threshold after checking the score distribution for the selected model/site.

## Private UTC Data

Do not place UTC HDF5 caches, UTC extracted frames, or private UTC weights in
this public repository. The `.gitignore` rules block the usual UTC demo paths.
