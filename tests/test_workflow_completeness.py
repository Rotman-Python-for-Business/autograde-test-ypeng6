from pathlib import Path
import re

import nbformat


NOTEBOOK_PATH = Path("module4_machine_learning.ipynb")


def load_notebook():
    assert NOTEBOOK_PATH.exists(), "module4_machine_learning.ipynb is missing."
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def get_cell_by_id(nb, cell_id):
    for cell in nb.cells:
        if cell.get("id") == cell_id:
            return cell
    raise AssertionError(f"Could not find required notebook cell id: {cell_id}")


def source_text(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def has_real_attempt(text):
    cleaned = text.strip().lower()

    placeholders = {
        "",
        "# your code here",
        "# your code here — try a different stock",
        "# your code here - try a different stock",
    }

    if cleaned in placeholders:
        return False

    non_comment_lines = [
        line for line in cleaned.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    return len(non_comment_lines) >= 2


def test_notebook_exists_and_opens():
    nb = load_notebook()
    assert len(nb.cells) >= 20, "Notebook seems incomplete or corrupted."


def test_required_student_cells_exist():
    nb = load_notebook()

    required_cell_ids = [
        "checkpoint-1-code",
        "dyh9pjy5l5c",
        "ex1-code",
        "ex2-code",
    ]

    for cell_id in required_cell_ids:
        cell = get_cell_by_id(nb, cell_id)
        assert cell["cell_type"] == "code", f"{cell_id} should be a code cell."


def test_checkpoint_k3_attempted():
    nb = load_notebook()
    cell = get_cell_by_id(nb, "checkpoint-1-code")
    text = source_text(cell)

    assert has_real_attempt(text), "Checkpoint K=3 cell still looks incomplete."

    expected_patterns = [
        r"KMeans",
        r"n_clusters\s*=\s*3",
        r"fit\s*\(",
    ]

    matched = sum(bool(re.search(p, text)) for p in expected_patterns)
    assert matched >= 2, "K=3 checkpoint should attempt KMeans with 3 clusters."


def test_random_forest_tuning_attempted():
    nb = load_notebook()
    cell = get_cell_by_id(nb, "dyh9pjy5l5c")
    text = source_text(cell)

    assert has_real_attempt(text), "Random Forest tuning cell still looks incomplete."

    expected_patterns = [
        r"RandomForestClassifier",
        r"n_estimators",
        r"max_depth",
        r"accuracy",
        r"acc_",
    ]

    matched = sum(bool(re.search(p, text, re.IGNORECASE)) for p in expected_patterns)
    assert matched >= 2, (
        "Random Forest tuning should adjust model parameters and examine accuracy."
    )


def test_exercise_41_elbow_method_attempted():
    nb = load_notebook()
    cell = get_cell_by_id(nb, "ex1-code")
    text = source_text(cell)

    assert has_real_attempt(text), "Exercise 4.1 elbow-method cell still looks incomplete."

    expected_patterns = [
        r"inertias",
        r"range\s*\(\s*1\s*,\s*7\s*\)",
        r"KMeans",
        r"inertia_",
        r"plt\.plot",
    ]

    matched = sum(bool(re.search(p, text)) for p in expected_patterns)
    assert matched >= 3, (
        "Exercise 4.1 should compute inertia for multiple K values and plot the elbow curve."
    )


def test_exercise_42_different_stock_attempted():
    nb = load_notebook()
    cell = get_cell_by_id(nb, "ex2-code")
    text = source_text(cell)

    assert has_real_attempt(text), "Exercise 4.2 different-stock cell still looks incomplete."

    non_aapl_tickers = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]

    tried_different_stock = any(ticker in text.upper() for ticker in non_aapl_tickers)

    assert tried_different_stock, (
        "Exercise 4.2 should try a stock other than AAPL, such as NVDA or MSFT."
    )


def test_student_saved_some_outputs_or_execution_counts():
    nb = load_notebook()

    code_cells = [cell for cell in nb.cells if cell["cell_type"] == "code"]

    executed_cells = [
        cell for cell in code_cells
        if cell.get("execution_count") is not None or cell.get("outputs")
    ]

    assert len(executed_cells) >= 5, (
        "Please run the notebook and save it before pushing. "
        "At least 5 code cells should show execution counts or outputs."
    )
