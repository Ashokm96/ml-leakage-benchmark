import pathlib


def test_solution_uses_timeseries_split():
    src = pathlib.Path(__file__).parent.joinpath("solution.py").read_text()
    assert "TimeSeriesSplit" in src
    assert "shuffle=True" not in src
    assert "train_test_split" not in src


def test_solution_score():
    from tasks.task2_validation.solution import run

    score = run()
    assert 0.8 < score <= 1.0
    assert run() == score


def test_starter_uses_random_split():
    src = pathlib.Path(__file__).parent.joinpath("starter.py").read_text()
    assert "train_test_split" in src
    assert "shuffle=True" in src
