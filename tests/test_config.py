import pytest

from conftest import ROOT
from ma_x3d.config import load_config


def test_base_matches_dataclass_defaults():
    cfg = load_config(ROOT / "configs/base.yaml")
    assert cfg.data.frames == 16
    assert cfg.train.unfreeze == ["res2", "res3", "res5", "head"]
    assert cfg.model.wk_stages == ["res2", "res3"]


def test_inheritance_and_overrides():
    cfg = load_config(ROOT / "configs/ablation/a1_ma.yaml",
                      ["train.epochs=3", "model.eaa=true", "train.lr_new=1e-3"])
    assert cfg.name == "a1_ma"
    assert cfg.model.wide_kernel == "none"
    assert cfg.model.motion_attention is True  # inherited
    assert cfg.train.epochs == 3 and cfg.model.eaa is True
    assert cfg.train.lr_new == pytest.approx(1e-3)


def test_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        load_config(ROOT / "configs/base.yaml", ["train.epoch=3"])


@pytest.mark.parametrize("path", sorted((ROOT / "configs").rglob("*.yaml")), ids=lambda p: p.stem)
def test_every_config_loads(path):
    cfg = load_config(path)
    assert cfg.model.wide_kernel in ("none", "reparam", "dense")
    assert cfg.data.protocol in ("holdout", "test_as_val")
    assert cfg.train.param_match in ("exact", "legacy")


def test_experiment_names_are_unique():
    names = [load_config(p).name for p in (ROOT / "configs").rglob("*.yaml")
             if p.name != "base.yaml"]
    # configs/ma_x3d.yaml and configs/x3d_m.yaml intentionally mirror a3/a0 under other names
    assert len(names) == len(set(names))
