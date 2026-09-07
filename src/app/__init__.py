"""Retrieval and serving layer for the finance chatbot.

Design notes in docs/guides/06-retrieval-and-the-app-layer.md. Modules are imported
directly (``from src.app.pipeline import RAGPipeline``) and each is runnable:

    python -m src.app.pipeline "What is Tata Steel trading at?"
    python -m src.app.corpus   "Why is Tata Steel restructuring in the UK?"
    python -m src.app.stock    "Reliance"
    python -m src.app.router   "how much does one infosys share cost"
"""
