"""Phase 7 web application.

A thin FastAPI layer over the frozen Phase 6 DWT-SVD pipeline. This package
contains no watermarking mathematics of its own: every numerical operation is
delegated to ``src.watermark`` and ``src.evaluation``.
"""
