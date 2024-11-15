from self_net import self_net
from typing import Tuple, List
from warnings import warn
from pathlib import Path
from PIL import Image
import numpy as np
import torch

def get_iandu_path(predPath: str, annPath: str, classIdx: int) -> Tuple[int, int]:
    """
    This method calculate a prediction's a specified class iou from its ann with their paths.
    We use "PIL.Image.open" to open a image, then calculate and return the i nad u.
    "pred" and "ann" should contain uint8-like values from 0 to 3.

    Args:
        predPath: Your single image prediction's path, e.g. with suffix "png" or ...
        annPath: Your single image annotation's path, e.g. with suffix "png" or ...
        classIdx: Your specified class index.

    Returns:
        i, u: intersection and union respectively.
    """
    pred = np.array(Image.open(predPath).convert("L"))
    ann = np.array(Image.open(annPath).convert("L"))
    return get_iandu_array(pred, ann, classIdx)


def get_iandu_npy(predPath: str, annPath, classIdx: int) -> Tuple[int, int]:
    """
    This method is similar to "get_iandu_path".
    We use "numpy.load" to open a .npy file, then calculate and return the i and u.
    "pred" and "ann" should contain uint8-like values from 0 to 3.

    Args:
        predPath: Your single .npy prediction's path.
        annPath: Your single .npy annotation's path.
        classIdx: Your specified class index.

    Returns:
        i, u: intersection and union respectively.
    """
    pred = np.load(predPath)
    ann = np.load(annPath)
    return get_iandu_array(pred, ann, classIdx)


def get_iandu_array(pred, ann, classIdx: int) -> Tuple[int, int]:
    """
    This method calculate the intersection and union between array-like "pred" and "ann".

    Args:
        pred: Your prediction array.
        ann: Your annotation array.
        classIdx: Your target class index.

    Returns:
        i, u: intersection and union respectively.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(ann, torch.Tensor):
        ann = ann.numpy()
    if not (0 <= classIdx and classIdx <= 3):
        raise ValueError(f'You should pass "classIdx" between 0 and 3 but you got {classIdx}.')
    if not (pred.shape == ann.shape):
        raise ValueError(
            f'You should pass "pred" and "ann" with same shape but you got a "pred" shape {pred.shape} and an "ann" shape {ann.shape}.'
        )
    if not (pred.shape[-2:] == (200, 200)):
        warn(
            "You'd better use array of (200, 200) shape but we didn't test this.",
            UserWarning,
        )
    i = np.sum(np.logical_and(np.equal(pred, ann), np.equal(ann, classIdx)))
    u = np.sum(np.logical_or(np.equal(pred, classIdx), np.equal(ann, classIdx)))
    return i, u


def get_arrays_dir(dir: str) -> List[np.ndarray]:
    """
    This method returns "numpy.ndarray"s in uint8-like dtype in a directory.
    We transform files into numpy.ndarray and save them in a "list" then return.
    The searching order is "png, jpg, npy" and we only support them.

    Args:
        dir: Your directory path which saves many files.

    Returns:
        arrays: Lists of your ndarrays.
    """
    dir, xs = Path(dir), None
    assert dir.is_dir()

    pngs = sorted(dir.glob("*.png"))
    if len(pngs) == 840:
        xs = pngs
    jpgs = sorted(dir.glob("*.jpg"))
    if len(jpgs) == 840:
        xs = jpgs
    if xs is not None:
        return [np.array(Image.open(p).convert("L")) for p in xs]

    npys = sorted(dir.glob("*.npy"))
    if len(npys) == 840:
        xs = npys
    if xs is not None:
        return [np.load(p) for p in xs]

    raise FileNotFoundError("Missing file in test directory.")


def get_ious_dir(preds_dir: str, anns_dir: str) -> Tuple[float, float, float]:
    """
    This method get 2 directory and transform them into arrays, then calculate the 3 types of iou.

    Args:
        preds_dir: Your directory where predictions are saved.
        anns_dir: Your directory where annotations are saved.

    Returns:
        iou_inclusions, iou_patches, iou_scratches: Same as name.
    """
    preds = get_arrays_dir(preds_dir)
    anns = get_arrays_dir(anns_dir)
    # We did not use "re" to strongly assert two list are matched here but you can

    i1, u1, i2, u2, i3, u3 = 0, 0, 0, 0, 0, 0
    for pred, ann in zip(preds, anns):
        i, u = get_iandu_array(pred, ann, 1)
        i1, u1 = i1 + i, u1 + u

        i, u = get_iandu_array(pred, ann, 2)
        i2, u2 = i2 + i, u2 + u

        i, u = get_iandu_array(pred, ann, 3)
        i3, u3 = i3 + i, u3 + u
    return i1 / u1, i2 / u2, i3 / u3


if __name__ == "__main__":
    # baseline = np.zeros(4, dtype=float)
    # baseline[:-1] = get_ious_dir("submit/test_ground_truths", "submit/baseline_predictions")
    # baseline[-1] = np.mean(baseline[:-1])
    # print(
    #     f"Baseline UNet\tIoU1: {baseline[0]:.4f}, IoU2: {baseline[1]:.4f}, IoU3: {baseline[2]:.4f}, mIoU: {baseline[3]:.4f}"
    # )
    # ours = np.zeros(4, dtype=float)
    # ours[:-1] = get_ious_dir("submit/test_ground_truths", "submit/test_predictions")
    # ours[-1] = np.mean(ours[:-1])
    # print(f"Ours\tIoU1: {ours[0]:.4f}, IoU2: {ours[1]:.4f}, IoU3: {ours[2]:.4f}, mIoU: {ours[3]:.4f}")
    # delta = ours - baseline
    # score = np.zeros(3, dtype=float)
    # for i, _ in enumerate(score):
    #     if delta[i] >= 0.06:
    #         score[i] = 100
    #     elif delta[i] > 0:
    #         score[i] = 40 + (130 * delta[i]) ** 2
    #     else:
    #         score[i] = 0
    # print(
    #     f"Score1: {score[0]:.2f}/{100:.2f}, Score2: {score[1]:.2f}/{100:.2f}, Score3: {score[2]:.2f}/{100:.2f}, IoU Score: {np.sum(score):.2f}/{300:.2f}"
    # )
    model = self_net()
    norm_params = sum([param.nelement() for param in model.parameters()]) / 1e6
    if norm_params >= 50:
        score_para = 0
    elif norm_params > 17:
        score_para = 10
    elif norm_params > 1:
        score_para = 70 - (norm_params - 1) * (15 / 4)
    elif norm_params > 0:
        score_para = 70
    else:
        score_para = 0
        print("Error!")
    print(f"Number of params: {norm_params:.2f}M, score: {score_para:.2f}/{70:.2f}")
    #total_score = np.sum(score) + score_para if score_para else 0
    #print(f"Total Score: {total_score:.2f}/{370:.2f}")