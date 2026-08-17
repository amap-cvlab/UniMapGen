import json

from unimapgen.codec import decode_lanes, encode_lanes


def test_lane_codec_round_trip():
    lanes = [
        {
            "start_point": [0, 12],
            "start_type": "<cut_point>",
            "end_point": [895, 40],
            "end_type": "</cut_point>",
            "sample_points": [[0, 12], [400, 20], [895, 40]],
            "category": "Laneline",
        }
    ]
    encoded = encode_lanes(lanes)
    assert "|sample_points|" in encoded
    assert "<c_895>" in encoded
    assert decode_lanes(encoded) == lanes


def test_codec_ignores_out_pad_points():
    encoded = (
        "[{|start_point|:[<c_0>,<c_0>],|start_type|:<start_point>,"
        "|end_point|:[<c_1>,<c_1>],|end_type|:<end_point>,"
        "|sample_points|:[[<c_0>,<c_0>],[<out_pad>,<out_pad>],[<c_1>,<c_1>]],"
        "|category|:<Curb>}]"
    )
    assert decode_lanes(encoded)[0]["sample_points"] == [[0, 0], [1, 1]]
