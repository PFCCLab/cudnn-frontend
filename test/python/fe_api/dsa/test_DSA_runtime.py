# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.L0
def test_torch_stream_context_preserves_legacy_default_stream():
    try:
        from cuda.bindings import driver as cuda
        from cudnn.deepseek_sparse_attention.utils.runtime import torch_stream_context
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    outer_stream = torch.cuda.Stream()
    with torch.cuda.stream(outer_stream):
        with torch_stream_context(cuda.CUstream(0)):
            assert torch.cuda.current_stream().cuda_stream == 0
        assert torch.cuda.current_stream().cuda_stream == outer_stream.cuda_stream


def _grad_loss_validator():
    try:
        from cudnn.deepseek_sparse_attention.indexer_backward.api import _validate_grad_loss_tensor
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    return _validate_grad_loss_tensor


@pytest.mark.L0
def test_validate_grad_loss_tensor_accepts_and_rejects():
    validate = _grad_loss_validator()
    device = torch.device("cuda", torch.cuda.current_device())

    for shape in ((), (1,), (1, 1)):
        out = validate(torch.ones(shape, dtype=torch.float32, device=device), device)
        assert tuple(out.shape) == (1,)
        assert out.dtype == torch.float32

    with pytest.raises(TypeError):
        validate(1.0, device)
    with pytest.raises(ValueError):
        validate(torch.ones(2, dtype=torch.float32, device=device), device)
    with pytest.raises(ValueError):
        validate(torch.ones((), dtype=torch.float16, device=device), device)


@pytest.mark.L0
def test_validate_grad_loss_tensor_stays_host_side(monkeypatch):
    """The guard must not read the tensor back to the host.

    Under a torch-compat proxy (Paddle's ``enable_compat``) ``numel()`` is an op
    returning a 0-D tensor, so comparing it forces a blocking device-to-host copy
    on the legacy default stream -- a full-device barrier that serialises this
    backward against any collective in flight. ``shape`` is host metadata, so the
    predicate must be built from it instead.
    """
    validate = _grad_loss_validator()
    device = torch.device("cuda", torch.cuda.current_device())
    grad_loss = torch.ones((), dtype=torch.float32, device=device)

    def _boom(self, *args, **kwargs):
        raise AssertionError("grad_loss validation must not call numel()")

    monkeypatch.setattr(torch.Tensor, "numel", _boom, raising=True)
    assert tuple(validate(grad_loss, device).shape) == (1,)
