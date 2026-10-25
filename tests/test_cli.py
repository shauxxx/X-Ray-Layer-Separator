import json

import pytest

from pcblayers.main import main


def test_demo_runs_with_one_command(tmp_path, capsys):
    code = main(["--demo", "-o", str(tmp_path / "out"), "--no-plots"])
    assert code == 0
    out = capsys.readouterr().out
    assert "layer_1" in out and "layer_2" in out
    assert (tmp_path / "out" / "layer_1.png").is_file()
    assert (tmp_path / "out" / "layer_2.png").is_file()
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["backend"] == "builtin"


def test_make_demo_only_writes_gif(tmp_path):
    target = tmp_path / "demo.gif"
    assert main(["--make-demo", str(target), "--demo-frames", "8", "--demo-size", "96"]) == 0
    assert target.is_file()
    assert not (tmp_path / "layer_1.png").exists()


def test_input_gif_with_options(scene_gif, tmp_path):
    code = main([str(scene_gif), "-o", str(tmp_path), "--backend", "builtin",
                 "--bright-metal", "--no-plots", "--iterations", "2"])
    assert code == 0
    assert not (tmp_path / "shifts.png").exists()


def test_missing_input_returns_error_code(tmp_path):
    assert main([str(tmp_path / "missing.gif"), "-o", str(tmp_path / "out")]) == 2


def test_no_arguments_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_missing_log_file_is_clean_error(scene_gif, tmp_path):
    code = main([str(scene_gif), "-o", str(tmp_path), "--backend", "module3",
                 "--log", str(tmp_path / "nope.log")])
    assert code == 2


def test_bad_backend_is_rejected_by_argparse(scene_gif):
    with pytest.raises(SystemExit) as exc:
        main([str(scene_gif), "--backend", "magic"])
    assert exc.value.code == 2
