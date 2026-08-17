from unimapgen.inference import normalize_vllm_config


def test_qwen2vl_default_mrope_marker_is_normalized():
    config = {
        "model_type": "qwen2_vl",
        "rope_scaling": {
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default",
        },
    }
    normalized, changed = normalize_vllm_config(config)
    assert changed is True
    assert normalized["rope_scaling"]["rope_type"] == "mrope"
    assert normalized["rope_scaling"]["type"] == "mrope"
    assert config["rope_scaling"]["rope_type"] == "default"
