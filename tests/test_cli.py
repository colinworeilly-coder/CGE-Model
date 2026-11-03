from regcge.cli import main


def test_cli_check(capsys):
    main(["check"])
    out = capsys.readouterr().out
    assert "Benchmark replication" in out and "Closure (scenario): OK" in out


def test_cli_run(tmp_path, capsys):
    main(["run", "scenarios/baseline.yaml", "--out", str(tmp_path), "--quiet"])
    assert (tmp_path / "baseline.xlsx").exists()
