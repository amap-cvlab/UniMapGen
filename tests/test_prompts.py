from unimapgen.prompts import build_stitching_prompt, parse_tile_id


def test_tile_name_parser_keeps_prefix_underscores():
    tile = parse_tile_id("PIT_3_0_sat_896_0_1792_896.png")
    assert tile.prefix == "PIT_3_0_sat"
    assert tile.key == ("PIT_3_0_sat", 896, 0)


def test_left_neighbor_becomes_current_left_ref():
    left = [
        {
            "start_point": [895, 100],
            "start_type": "<cut_point>",
            "end_point": [700, 200],
            "end_type": "<end_point>",
            "sample_points": [[895, 100], [700, 200]],
            "category": "Curb",
        }
    ]
    prompt = build_stitching_prompt(left, None)
    assert "线结束提示点为" in prompt
    assert "|ref_point|:[<c_0><c_100>]" in prompt
    assert "<Curb>" in prompt
