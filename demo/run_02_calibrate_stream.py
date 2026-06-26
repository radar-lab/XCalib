"""Demo step 2: stream public A9 frames, visualize matches, and solve calibration."""

from __future__ import annotations

from xcalib.visualization import draw_matching_overlay

from demo_common import (
    CalibrationState,
    calibrate_when_ready,
    iter_demo_frames,
    load_demo_matcher,
    load_intrinsics,
    print_calibration_failure,
    save_overlay,
)
from demo_config import (
    CAMERA_NAME,
    DEVICE,
    INTRINSICS_JSON,
    LOCAL_DATASET,
    LOCAL_WEIGHTS,
    MATCH_THRESHOLD,
    MIN_CALIBRATION_PAIRS,
    MODEL,
    NUM_FRAMES,
    SAMPLE_FRAMES,
    SITE,
    SPLIT,
    VIZ_DIR,
)


def main() -> int:
    matcher = load_demo_matcher(MODEL, site=SITE, device=DEVICE, local_weights=LOCAL_WEIGHTS)
    intrinsics = load_intrinsics(INTRINSICS_JSON)
    session = matcher.oneshot(intrinsics, match_threshold=MATCH_THRESHOLD)
    state = CalibrationState()

    for frame in iter_demo_frames(
        sample_frames=SAMPLE_FRAMES,
        site=SITE,
        split=SPLIT,
        camera_name=CAMERA_NAME,
        limit=NUM_FRAMES,
        local_dataset=LOCAL_DATASET,
    ):
        # Visualize the camera<->LiDAR matches that feed the calibration buffer.
        result = matcher.match(
            frame.image,
            frame.point_cloud,
            frame.bboxes_2d,
            frame.bboxes_3d,
            top_k=1,
            match_threshold=MATCH_THRESHOLD,
        )
        overlay = draw_matching_overlay(
            frame.image,
            frame.point_cloud,
            frame.bboxes_2d,
            frame.bboxes_3d,
            result.matches,
            match_threshold=MATCH_THRESHOLD,
        )
        overlay_path = save_overlay(overlay, VIZ_DIR / f"{frame.frame_key}_match.png")

        report = session.observe(
            frame.image,
            frame.point_cloud,
            frame.bboxes_2d,
            frame.bboxes_3d,
        )
        state.buffered_pairs += report.n_confident_matches
        print(
            f"frame {frame.frame_key}: "
            f"confident_pairs={report.n_confident_matches}, "
            f"buffered_total~={state.buffered_pairs} "
            f"-> {overlay_path.relative_to(VIZ_DIR.parent)}"
        )

        if calibrate_when_ready(session, state, min_pairs=MIN_CALIBRATION_PAIRS):
            print(f"match overlays saved under {VIZ_DIR}/")
            return 0

    print_calibration_failure(state, MIN_CALIBRATION_PAIRS)
    print(f"match overlays saved under {VIZ_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
