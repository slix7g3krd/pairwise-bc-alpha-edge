import pytest
import graphics.bc_alpha_edge as mod

CASES = [
    ("alpha_edge", (1)),
    ("normal_z", (1)),
    ("color_space", (1)),
    ("mip_pre", (1)),
    ("encode_determinism", (1)),
    ("block_align", (1)),
    ("srgb_linear", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("quality_metric", (1)),
    ("metrics", (1)),
    ("format_select", (1)),
    ("batch_repeat", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
