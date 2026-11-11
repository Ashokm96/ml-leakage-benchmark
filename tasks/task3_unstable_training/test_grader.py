import pathlib


def test_solution_uses_scaler_and_seed():
    src = pathlib.Path(__file__).parent.joinpath("solution.py").read_text()
    assert "StandardScaler" in src
    assert "Pipeline" in src
    assert "random_state=42" in src or "RandomState(42)" in src


def test_solution_stable_and_reproducible():
    from tasks.task3_unstable_training.solution import run

    first = run()
    second = run()
    assert first == second
    assert first > 0.85


def test_starter_unstable():
    src = pathlib.Path(__file__).parent.joinpath("starter.py").read_text()
    assert "RandomState(None)" in src or "eta0=1.0" in src
    assert "StandardScaler" not in src
