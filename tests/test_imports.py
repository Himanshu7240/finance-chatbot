"""Every module imports and every CLI parses.

Trivial-looking, and it exists for a concrete reason: a syntax error was once shipped in
`ui.py` and the whole suite stayed green, because nothing imported it. Tests that assert
behaviour are worth more than these - but only for code the tests actually load.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "src.app.corpus",
    "src.app.model",
    "src.app.pipeline",
    "src.app.router",
    "src.app.stock",
    "src.app.ui",
    "src.dataset.build",
    "src.dataset.clean",
    "src.dataset.qa_prompt",
    "src.dataset.qa_validate",
    "src.dataset.schema",
    "src.dataset.split",
    "src.scraping.article",
    "src.scraping.collect",
    "src.scraping.fetcher",
    "src.scraping.matcher",
    "src.scraping.robots",
    "src.scraping.sitemaps",
    "src.training.format",
    "src.training.merge",
    "src.training.metrics",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", [m for m in MODULES if m.split(".")[-1] in {
    "build", "clean", "collect", "corpus", "merge", "model", "pipeline", "router",
    "split", "stock", "ui",
}])
def test_cli_parser_builds(name, monkeypatch):
    """`main()` up to the point of doing anything - argparse config is easy to break."""
    module = importlib.import_module(name)
    monkeypatch.setattr("sys.argv", [name, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    assert exit_info.value.code == 0


def test_the_gradio_app_builds():
    gradio = pytest.importorskip("gradio")
    from src.app.ui import ChatSession, build_demo

    demo = build_demo(ChatSession(use_embeddings=False))
    assert isinstance(demo, gradio.Blocks)
