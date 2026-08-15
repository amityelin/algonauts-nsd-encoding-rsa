import contextlib
import sys
import types

import numpy as np

from algonauts_rsa.features import extract_clip_features


def test_clip_extractor_preserves_notebook_loader_amp_and_transfer_contract(monkeypatch):
    calls = {}

    class FakeEncoded:
        def norm(self, dim, keepdim):
            calls["norm"] = (dim, keepdim)
            return 1

        def __truediv__(self, other):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([[1.0, 2.0]])

    class FakeBatch:
        def to(self, device, non_blocking):
            calls["transfer"] = (device, non_blocking)
            return self

    class FakeModel:
        def eval(self):
            calls["eval"] = True

        def encode_image(self, batch):
            return FakeEncoded()

    class FakeDataset:
        pass

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            calls["loader"] = kwargs

        def __iter__(self):
            return iter([FakeBatch()])

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = contextlib.nullcontext
    fake_torch.amp = types.SimpleNamespace(
        autocast=lambda device, enabled: calls.update(amp=(device, enabled)) or contextlib.nullcontext()
    )
    fake_torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: calls.update(empty_cache=True)
    )
    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")
    fake_data.DataLoader = FakeLoader
    fake_data.Dataset = FakeDataset
    fake_open_clip = types.ModuleType("open_clip")
    fake_open_clip.create_model_and_transforms = (
        lambda *args, **kwargs: (FakeModel(), None, lambda image: image)
    )
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = object()
    fake_tqdm_auto = types.ModuleType("tqdm.auto")
    fake_tqdm_auto.tqdm = lambda iterator, desc: calls.update(desc=desc) or iterator

    for name, module in {
        "torch": fake_torch,
        "torch.utils": fake_utils,
        "torch.utils.data": fake_data,
        "open_clip": fake_open_clip,
        "PIL": fake_pil,
        "tqdm.auto": fake_tqdm_auto,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    features, tag = extract_clip_features(
        ["image.jpg"], 7, "subj01", device="cuda", num_workers=2,
        pin_memory=True, persistent_workers=True, prefetch_factor=3, use_amp=True,
    )

    assert calls["loader"] == {
        "batch_size": 7, "shuffle": False, "num_workers": 2,
        "pin_memory": True, "persistent_workers": True, "prefetch_factor": 3,
    }
    assert calls["amp"] == ("cuda", True)
    assert calls["transfer"] == ("cuda", False)
    assert calls["norm"] == (-1, True)
    assert calls["desc"] == "CLIP feats [subj01]"
    assert calls["empty_cache"] is True
    np.testing.assert_array_equal(features, [[1.0, 2.0]])
    assert tag == "CLIP_ViT-B-32"
