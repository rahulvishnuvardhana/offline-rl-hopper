import json

from offrl.runmeta import write_metadata


def test_metadata_has_required_fields(tmp_path):
    meta = write_metadata(tmp_path)
    on_disk = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert on_disk == meta
    for key in ("git_commit", "git_dirty", "python", "packages", "hostname", "gpu", "started_utc"):
        assert key in meta
    assert meta["packages"]["torch"] is not None
