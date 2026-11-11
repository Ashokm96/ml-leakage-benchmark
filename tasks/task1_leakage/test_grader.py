import pathlib


def test_solution_uses_pipeline():
    src = pathlib.Path(__file__).parent.joinpath("solution.py").read_text()
    assert "Pipeline" in src
    assert "fit_transform(X)" not in src


def test_solution_accuracy():
    from tasks.task1_leakage.solution import run

    accuracy = run()
    assert 0.7 < accuracy < 1.0
    assert run() == accuracy


def test_starter_has_leakage():
    src = pathlib.Path(__file__).parent.joinpath("starter.py").read_text()
    assert "fit_transform(X)" in src
