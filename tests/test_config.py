from pathlib import Path


def test_config_defaults():
    from radiology_vqa.config import Settings

    s = Settings()
    assert s.data_dir == Path("./data")
    assert s.slake_dir == Path("./data/raw/Slake1.0")
    assert s.log_level == "INFO"
    assert s.vqa_rad_dataset == "flaviagiammarino/vqa-rad"
    assert s.pathvqa_dataset == "flaviagiammarino/path-vqa"


def test_config_env_override(monkeypatch):
    from radiology_vqa.config import Settings

    monkeypatch.setenv("SLAKE_DIR", "/custom/slake")
    s = Settings()
    assert s.slake_dir == Path("/custom/slake")
