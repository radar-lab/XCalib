"""Import a rectified raw delivery as SUSTech base/refine scenes."""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .dataset_prepare import prepare_raw_workflow_scene
from .rename_retry import rename_when_free


def camera_record(path):
    """The rectified camera's matrix and image size read from an intrinsic file.

    Three kinds of file are understood: the undistortion parameters of a
    delivery (``K_undistort`` and ``image_processing.image_size``), the
    ``calib/intrinsic`` file of a scene (``intrinsic``, ``image_size``,
    ``image_state: rectified``), and a file that only names the camera matrix
    (``intrinsic``, ``camera_matrix`` or ``K``; the image size is then taken
    from the images). A file that describes a distorted camera is refused.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except ValueError:
        raise ValueError(f'{path.name} is not a JSON file') from None
    if not isinstance(data, dict):
        raise ValueError(f'{path.name} holds no camera matrix')
    if 'K_undistort' in data:
        processing = data.get('image_processing') or {}
        if processing.get('crop_to_valid_roi'):
            raise ValueError('Cropped calibration needs an explicit principal-point adjustment')
        return {'intrinsic': data['K_undistort'], 'image_size': processing.get('image_size')}
    key = next((name for name in ('intrinsic', 'camera_matrix', 'K') if data.get(name) is not None), None)
    if key is None:
        raise ValueError(f'{path.name} holds no camera matrix (K_undistort, intrinsic, camera_matrix or K)')
    if data.get('image_state', 'rectified') != 'rectified' or any(data.get('distortion') or []):
        raise ValueError(f'{path} does not describe a rectified camera')
    return {'intrinsic': data[key], 'image_size': data.get('image_size')}


def _camera_calibration(raw, intrinsic):
    """The rectified camera's intrinsic record for a scene, or None when unknown."""
    parameters = raw / 'undistortion_parameters.json'
    if parameters.is_file():
        record = camera_record(parameters)
    elif intrinsic is not None:
        record = camera_record(intrinsic)
    else:
        return None
    try:
        K = np.asarray(record['intrinsic'], dtype=float)
    except (TypeError, ValueError):
        raise ValueError('Invalid camera calibration') from None
    if K.size != 9 or not np.isfinite(K).all():
        raise ValueError('Invalid camera calibration')
    size = record.get('image_size')
    return {'intrinsic': K.ravel().tolist(), 'distortion': [0.0] * 5,
            'image_size': [int(v) for v in size] if size else None, 'image_state': 'rectified'}


def label(raw, output, *, scene_prefix='scene', role=None, intrinsic=None, progress=print):
    """Create new scenes; never rewrite the input.

    The frames become ``<prefix>_base`` and ``refine/<prefix>_refine``, or with
    ``role`` "base" or "refine" that one scene holding the whole delivery.

    The delivery must already be rectified: its images, and the camera
    detections made on them, use the K_undistort matrix of
    undistortion_parameters.json. That matrix is saved with each scene as
    calib/intrinsic/cam_sync.json and the camera boxes become drafts. A delivery
    without that file takes ``intrinsic`` instead: an intrinsic file of the same
    rectified camera in one of the forms ``camera_record`` reads (undistortion
    parameters kept elsewhere, the calib/intrinsic file of a scene, or a file
    with the camera matrix alone). With neither, the scenes get
    no intrinsic and cannot be calibrated yet. A delivery with distorted images needs
    scripts/undistort_raw_delivery.py first. LiDAR vendor rows are only copied
    to vendor/lidar_perception.csv here; lidar_import.label_lidar turns them
    into 3D boxes in the PCD frame.
    """
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(output)
    record = _camera_calibration(raw, intrinsic)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.label-') as temporary:
        stage = Path(temporary) / 'scenes'
        report = prepare_raw_workflow_scene(raw, stage, scene_prefix=scene_prefix, role=role,
                                            progress=progress)
        for name in report['frame_counts'] if record else ():
            scene = stage / name if name.endswith('_base') else stage / 'refine' / name
            first = min((scene / 'camera' / 'cam_sync').iterdir())
            image = cv2.imread(str(first))
            if image is not None and record['image_size'] is None:   # a file with the matrix alone
                record['image_size'] = [int(v) for v in image.shape[1::-1]]
            if image is None or list(image.shape[1::-1]) != record['image_size']:
                raise ValueError(f'Image size does not match calibration: {first.name}')
            # Not calib/camera: every file there is read as a complete calibration,
            # with an extrinsic. The pairing window reads this one for its first calibration.
            calibration = scene / 'calib' / 'intrinsic'
            calibration.mkdir(parents=True, exist_ok=True)
            (calibration / 'cam_sync.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        report['camera_image_state'] = 'rectified'
        report['camera_intrinsic'] = np.reshape(record['intrinsic'], (3, 3)).tolist() if record else None
        (stage / 'split_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        rename_when_free(stage, output, progress)
    return report
