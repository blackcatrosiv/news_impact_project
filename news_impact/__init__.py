"""Historical news-to-market event-study layer for the portfolio research pipeline.

The inputs are article metadata and market bars; the outputs are event-level forward returns
and descriptive summaries. Optional labels such as ``novelty_label`` are passed through, not
created here, so the RAG/forecasting work can develop independently and join by ``article_id``.
This package does not retrieve news, generate labels, or predict returns.

Use :func:`build_event_dataset` for the main event-study path, or call the lower-level matching,
I/O, calendar, and summary helpers exported below.
"""
from .core import (  # noqa: F401
    DEFAULT_HORIZONS,
    TradingGrid,
    build_event_dataset,
    build_trading_grid,
    calculate_abnormal_return,
    calculate_return,
    match_news_to_prices,
)
from .io import load_news, load_prices, save_event_dataset  # noqa: F401
from .summary import compare_groups, summarize_event_impacts  # noqa: F401
from .trading_calendar import NYSECalendar, check_calendar_against_prices  # noqa: F401

__version__ = "0.1.0"
