from app.pipeline.cuda_perf import resolve_inference_device


def test_resolve_inference_device_returns_string() -> None:
    dev = resolve_inference_device()
    assert isinstance(dev, str)
    assert dev in ("cpu", "mps", "cuda:0") or dev.startswith("cuda")
