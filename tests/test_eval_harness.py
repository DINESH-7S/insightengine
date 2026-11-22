import json

from eval import run_eval


def test_harness_runs_offline_and_finds_obvious_answer(tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text(
        "The quarterly revenue grew by twelve percent.\n\n"
        "The office picnic is scheduled for friday afternoon.\n\n"
        "Engineering hired three new backend developers.",
        encoding="utf-8",
    )
    qs = tmp_path / "q.jsonl"
    qs.write_text(json.dumps({"question": "when is the office picnic", "doc": "notes.txt", "pages": [1]}) + "\n")

    code = run_eval.main(["--docs", str(docs), "--questions", str(qs), "--embedder", "hash",
                          "--strategies", "dense", "hybrid", "--top-k", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "| dense | 1.00" in out


def test_question_rows_are_validated(tmp_path):
    import pytest
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "x"}\n')
    with pytest.raises(ValueError):
        run_eval.load_questions(bad)
