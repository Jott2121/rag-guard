from rag_guard import cli

def _seed(tmp_path):
    (tmp_path / "a.md").write_text("shipping takes three days")
    (tmp_path / "b.md").write_text("returns within thirty days")

def test_run_reports_support(tmp_path):
    _seed(tmp_path)
    out = cli.run(["--roots", str(tmp_path), "--cache", str(tmp_path / "i.json")],
                  "how long is shipping")
    assert out["query"] == "how long is shipping" and out["hits"]
    assert out["support"] > 0 and out["grounded"] is True

def test_run_low_support_not_grounded(tmp_path):
    _seed(tmp_path)
    out = cli.run(["--roots", str(tmp_path), "--cache", str(tmp_path / "i.json")],
                  "zzzznonsense termxyz")
    assert out["grounded"] is False
