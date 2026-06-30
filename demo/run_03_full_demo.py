"""Demo step 3: match, calibrate, and visualize (match overlays + LiDAR projection)."""

from __future__ import annotations

from xcalib.visualization import draw_calibration_overlay, draw_matching_overlay

from demo_common import (
    CalibrationState,
    iter_demo_frames,
    load_demo_matcher,
    load_intrinsics,
    print_calibration,
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

        # Continual refinement: calibrate() re-solves over the WHOLE buffer and,
        # with its default degenerate-pose gate (since 0.2), adopts the latest
        # non-clump pose as objects sweep the scene. A near-coplanar early solve
        # that folds the cloud into a clump is rejected (result.accepted False);
        # we only (re)draw the projection overlay when a pose is accepted.
        cal = session.calibrate(min_pairs=MIN_CALIBRATION_PAIRS)
        if cal.success and cal.accepted:
            state.last_result = cal
            overlay = draw_calibration_overlay(
                frame.image,
                frame.point_cloud,
                projection=cal.projection,
                bboxes_3d=frame.bboxes_3d,
            )
            out = save_overlay(overlay, VIZ_DIR / f"{frame.frame_key}_calibration.png")
            print(
                f"  calibration refined: {cal.n_inliers}/{cal.n_correspondences} inliers, "
                f"buffer-median {cal.buffer_reproj_px:.1f}px -> {out.relative_to(VIZ_DIR.parent)}"
            )
        elif cal.success:
            print(
                f"  solve not accepted: inlier {cal.reproj_error_px:.2f}px, "
                f"buffer-median {cal.buffer_reproj_px:.1f}px (degenerate planar pose)"
            )

    if state.last_result is None:
        print_calibration_failure(state, MIN_CALIBRATION_PAIRS)
    else:
        print_calibration(state.last_result)
        print(
            f"\nfinal calibration: latest accepted pose, "
            f"{state.last_result.n_correspondences} pairs, "
            f"buffer-median {state.last_result.buffer_reproj_px:.1f}px "
            f"(best seen {session.best_reproj_px:.1f}px)"
        )
    print(f"overlays saved under {VIZ_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
