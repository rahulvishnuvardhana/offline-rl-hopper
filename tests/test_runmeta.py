import json
import platform

from offrl.runmeta import cpu_model, write_metadata


def test_metadata_has_required_fields(tmp_path):
    meta = write_metadata(tmp_path)
    on_disk = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert on_disk == meta
    for key in ("git_commit", "git_dirty", "python", "packages", "hostname", "gpu", "started_utc"):
        assert key in meta
    assert meta["packages"]["torch"] is not None


def test_cpu_model_reads_proc_cpuinfo(tmp_path):
    info = tmp_path / "cpuinfo"
    info.write_text(
        "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz\n",
        encoding="utf-8",
    )
    assert cpu_model(info) == "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz"
    assert cpu_model(tmp_path / "missing") == (platform.processor() or None)  # not Linux: platform's


def test_metadata_records_the_cpu_model(tmp_path):
    assert "cpu_model" in write_metadata(tmp_path)
