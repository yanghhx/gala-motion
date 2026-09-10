from pathlib import Path

import numpy as np

from datasets.kit_raw import hashed_text_embedding, motion_duration_frames


def test_hash_embedding_is_deterministic_and_normalized():
    first = hashed_text_embedding("A person walks forward")
    second = hashed_text_embedding("A person walks forward")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first), 1.0, atol=1e-6)


def test_mmm_duration_conversion(tmp_path: Path):
    xml = tmp_path / "x.xml"
    xml.write_text("<MMM><Motion><MotionFrames><MotionFrame><Timestep>0</Timestep></MotionFrame><MotionFrame><Timestep>2</Timestep></MotionFrame></MotionFrames></Motion></MMM>")
    assert motion_duration_frames(xml, target_fps=20) == 41
