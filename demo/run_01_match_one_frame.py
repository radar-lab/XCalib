"""Demo step 1: load one A9 frame and run detection matching."""

from __future__ import annotations

from demo_common import iter_demo_frames, load_demo_matcher, print_matches
from demo_config import (
    CAMERA_NAME,
    DEVICE,
    LOCAL_DATASET,
    LOCAL_WEIGHTS,
    MATCH_THRESHOLD,
    MODEL,
    SAMPLE_FRAMES,
    SITE,
    SPLIT,
)


def main() -> int:
    matcher = load_demo_matcher(MODEL, site=SITE, device=DEVICE, local_weights=LOCAL_WEIGHTS)
    frame = next(
        iter_demo_frames(
            sample_frames=SAMPLE_FRAMES,
            site=SITE,
            split=SPLIT,
            camera_name=CAMERA_NAME,
            limit=1,
            local_dataset=LOCAL_DATASET,
        )
    )

    result = matcher.match(
        frame.image,
        frame.point_cloud,
        frame.bboxes_2d,
        frame.bboxes_3d,
        top_k=1,
        match_threshold=MATCH_THRESHOLD,
    )
    print_matches(frame, result.matches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
