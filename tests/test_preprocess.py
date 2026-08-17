from unimapgen.preprocess import clip_lane_to_tile, sliding_positions


def test_sliding_positions_can_preserve_v6_or_cover_edge():
    assert sliding_positions(4096, 896, 896, False) == [0, 896, 1792, 2688]
    assert sliding_positions(4096, 896, 896, True)[-1] == 3200


def test_clip_lane_marks_both_crossings():
    lanes = clip_lane_to_tile(
        {"category": "Lane line", "points": [[-10, 10], [910, 10]]},
        x0=0,
        y0=0,
    )
    assert len(lanes) == 1
    lane = lanes[0]
    assert lane["start_type"] == "<cut_point>"
    assert lane["end_type"] == "</cut_point>"
    assert lane["start_point"] == [0, 10]
    assert lane["end_point"] == [895, 10]
    assert lane["category"] == "Laneline"
