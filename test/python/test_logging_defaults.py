# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This package's log records must not appear because the application asked for its own.

A library logger inherits its level from the root logger, so an application that
sets its root to DEBUG -- the default in several training frameworks -- switches
on every ``logger.debug`` in this package at once. Several of those sit on
per-execute paths, so the result is one line per kernel launch, per layer, per
step, drowning the application's own output. ``cudnn/__init__.py`` therefore
sets an explicit level on the package logger, which is what these tests pin.

Two halves, because either alone would pass for the wrong reason:

* the default has to be overridable, or a user debugging cuDNN cannot see
  anything -- both from the environment and from the application;
* every logger in the package has to actually BE a descendant of the package
  logger, or the default does not reach it. A bare
  ``getLogger(self.__class__.__name__)`` produces a *root* child named e.g.
  ``IndexerBackward``, which no amount of configuring ``cudnn`` can silence.

Each behavioural check runs in a FRESH interpreter: the level lives on a
process-global logger, and the environment is read at import, so an in-process
assertion would be contaminated by whatever ran before it -- pytest's own
logging plugin included.
"""

import ast
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.L0

_ENV_VAR = "CUDNN_FRONTEND_PY_LOG_LEVEL"

# A logger under the package that no module owns; naming a real one would tie
# the test to that module staying imported.
_PROBE = """
import logging
import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s|%(name)s|%(message)s")
import cudnn
{setup}
log = logging.getLogger("cudnn.probe.child")
log.debug("DEBUG-RECORD")
log.info("INFO-RECORD")
log.warning("WARNING-RECORD")
"""

_MARKERS = ("DEBUG-RECORD", "INFO-RECORD", "WARNING-RECORD")


def _emitted(setup: str = "", env: dict = None) -> set:
    """The record markers that reached a root handler, in a fresh interpreter."""
    run = subprocess.run(
        [sys.executable, "-c", _PROBE.format(setup=setup)],
        capture_output=True,
        text=True,
        env=None if env is None else {**os.environ, **env},
    )
    # Not skip: a probe that cannot `import cudnn` is exactly the regression a
    # skip would hide.
    assert run.returncode == 0, run.stderr.strip()[-2000:]
    out = run.stdout + run.stderr
    return {marker for marker in _MARKERS if marker in out}


def test_traces_are_off_even_when_the_application_root_is_debug():
    """WARNING, not INFO: a few INFO lines in this package are also per-call.

    Pinning INFO would newly surface those in an application whose root sits at
    WARNING, because propagation re-checks handler levels but not the root
    logger's own level.
    """
    assert _emitted() == {"WARNING-RECORD"}


def test_env_var_restores_debug():
    assert _emitted(env={_ENV_VAR: "DEBUG"}) == set(_MARKERS)


def test_env_var_accepts_a_numeric_level():
    """logging's levels are numbers, so a user will try the number."""
    assert _emitted(env={_ENV_VAR: "10"}) == set(_MARKERS)


def test_env_var_can_ask_for_info_alone():
    assert _emitted(env={_ENV_VAR: "info"}) == {"INFO-RECORD", "WARNING-RECORD"}


def test_a_malformed_env_var_does_not_break_the_import():
    """A debugging knob must not be able to take `import cudnn` down."""
    assert _emitted(env={_ENV_VAR: "verbose-please"}) == {"WARNING-RECORD"}


def test_the_application_can_still_turn_it_on():
    """An explicit setLevel by the application outranks the package default."""
    assert _emitted(setup='logging.getLogger("cudnn").setLevel(logging.DEBUG)') == set(_MARKERS)


def test_a_single_subtree_can_be_turned_on():
    """Module-derived names are what make a prefix a usable filter."""
    assert _emitted(setup='logging.getLogger("cudnn.probe").setLevel(logging.DEBUG)') == set(_MARKERS)


def test_the_package_logger_has_a_null_handler():
    """The library convention: no "last resort" stderr write before configuration."""
    setup = 'assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger("cudnn").handlers)'
    assert _emitted(setup=setup) == {"WARNING-RECORD"}


def _package_source() -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parents[2] / "python" / "cudnn"
    if not root.is_dir():
        pytest.skip(f"package source not next to the tests ({root})")
    return root


def test_every_logger_name_is_derived_from_the_module():
    """No ``getLogger`` may name a logger outside the package tree.

    Both failure modes are silent. A string literal like ``"cudnn.pygraph"``
    looks right but keeps pointing at the stock ``cudnn`` root when this package
    is vendored under another one, so the default set in ``__init__.py`` no
    longer applies to it. A bare class name is worse: it is a root child, so it
    inherits the *application's* level and there is no name an application could
    configure to reach the whole package.
    """
    root = _package_source()
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name != "getLogger":
                continue
            arg = ast.unparse(node.args[0])
            if "__name__" not in arg and "__module__" not in arg:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}: getLogger({arg})")
    assert not offenders, "logger names must come from __name__/__module__:\n" + "\n".join(offenders)
