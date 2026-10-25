"""Обёртка над скриптом участника 3: проверяем контракт, а не качество алгоритма."""

import json

import numpy as np
import pytest

from pcblayers.layer_separation import separate_layers
from pcblayers.main import main

SYNTH_LOG = """\
[GeometryCorrection]
Angle=45
Delta Z=1.0

[Acquisition]
Number Of Files=24
Object to Source (mm)=100
Camera to Source (mm)=200
Scaled Image Pixel Size (um)=125
Rotation Step (deg)=15
Rotation Direction=CW
"""


@pytest.fixture()
def synth_log(tmp_path):
    path = tmp_path / "synthetic.log"
    path.write_text(SYNTH_LOG, encoding="utf-8")
    return path


def test_separate_layers_follows_contract(scene, synth_log, tmp_path):
    layer_1, layer_2 = separate_layers(scene.frames, log_path=synth_log, out_dir=tmp_path)
    shape = scene.frames[0].shape
    for layer in (layer_1, layer_2):
        assert layer.shape == shape and layer.dtype == np.float32
        assert np.isfinite(layer).all() and layer.std() > 0
    assert not np.allclose(layer_1, layer_2)
    assert (tmp_path / "module3_debug" / "focus_curve.png").is_file()


def test_unused_arguments_are_accepted(scene, synth_log):
    shifts = np.zeros((len(scene.frames), 2, 2))
    result = separate_layers(scene.frames, shifts=shifts, reliable=None, strength=None,
                             log_path=synth_log)
    assert len(result) == 2


def test_input_validation(scene, synth_log):
    with pytest.raises(ValueError, match="at least 4 frames"):
        separate_layers(scene.frames[:3], log_path=synth_log)
    with pytest.raises(ValueError, match="scale"):
        separate_layers(scene.frames, log_path=synth_log, scale=0)


def test_cli_with_module3_backend_and_log(scene_gif, synth_log, tmp_path):
    out = tmp_path / "out"
    code = main([str(scene_gif), "-o", str(out), "--backend", "module3",
                 "--log", str(synth_log), "--no-plots"])
    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["backend"] == "module3"
    assert (out / "layer_1.png").is_file() and (out / "layer_2.png").is_file()
