"""Demo step 3: match, calibrate, and visualize (match overlays + LiDAR projection)."""

from __future__ import annotations

from xcalib.visualization import draw_calibration_overlay, draw_matching_overlay

from demo_common import (
    CalibrationState,
    calibrate_when_ready,
    iter_demo_frames,
    load_demo_matcher,
    load_intrinsics,
    print_calibration_failure,
    print_matches,
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
        result = matcher.match(
            frame.image,
            frame.point_cloud,
            frame.bboxes_2d,
            frame.bboxes_3d,
            top_k=1,
            match_threshold=MATCH_THRESHOLD,
        )
        print_matches(frame, result.matches)
        save_overlay(
            draw_matching_overlay(
                frame.image,
                frame.point_cloud,
                frame.bboxes_2d,
                frame.bboxes_3d,
                result.matches,
                match_threshold=MATCH_THRESHOLD,
            ),
            VIZ_DIR / f"{frame.frame_key}_match.png",
        )

        report = session.observe(
            frame.image,
            frame.point_cloud,
            frame.bboxes_2d,
            frame.bboxes_3d,
        )
        state.buffered_pairs += report.n_confident_matches
        print(f"  calibration buffer pairs: {state.buffered_pairs}")

        if calibrate_when_ready(
            session,
            state,
            min_pairs=MIN_CALIBRATION_PAIRS,
            not_ready_prefix=None,
        ):
            # Visualize the solved calibration: project the LiDAR onto the image.
            overlay = draw_calibration_overlay(
                frame.image,
                frame.point_cloud,
                projection=state.last_result.projection,
                bboxes_3d=frame.bboxes_3d,
            )
            out = save_overlay(overlay, VIZ_DIR / f"{frame.frame_key}_calibration.png")
            print(f"  projection overlay -> {out.relative_to(VIZ_DIR.parent)}")
            print(f"overlays saved under {VIZ_DIR}/")
            return 0

    print_calibration_failure(state, MIN_CALIBRATION_PAIRS)
    print(f"overlays saved under {VIZ_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
