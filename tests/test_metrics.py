from unimapgen.metrics import compute_chamfer_ap, compute_miou


def _record():
    lanes = [
        {
            "start_point": [10, 10],
            "start_type": "<start_point>",
            "end_point": [100, 10],
            "end_type": "<end_point>",
            "sample_points": [[10, 10], [100, 10]],
            "category": category,
        }
        for category in ("Curb", "Laneline", "Virtualline")
    ]
    # Offset categories so they do not overwrite one another in semantic masks.
    for index, lane in enumerate(lanes):
        lane["sample_points"] = [[10, 10 + index * 20], [100, 10 + index * 20]]
        lane["start_point"], lane["end_point"] = lane["sample_points"]
    return {"_pred_lanes": lanes, "_gt_lanes": lanes, "token_logprobs": []}


def test_perfect_prediction_metrics():
    record = _record()
    assert compute_miou([record])["mIoU"] == 100.0
    chamfer = compute_chamfer_ap([record])
    assert all(value == 1.0 for value in chamfer["mAP"].values())
