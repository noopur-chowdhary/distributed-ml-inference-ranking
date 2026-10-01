from __future__ import annotations

RAW_COLUMNS = [
    "user_id",
    "stream_id",
    "streamer",
    "start_time",
    "stop_time",
]

# Twitch timestamps in this dataset are 10-minute intervals.
INTERVAL_MINUTES = 10
RECENT_WINDOWS = (6, 18, 42)

# Semantic feature order from Notebook 02.
# The HGB ranker still reads its exact fitted order from the model artifact.
FEATURE_COLUMNS = [
    # User behavior
    "user_interactions",
    "user_unique_streamers",
    "user_total_watch_minutes",
    "user_avg_watch_minutes",
    "user_median_watch_minutes",
    "user_active_span_intervals",
    "user_last_activity_recency",
    "user_repeat_rate",
    "user_interactions_last_6_intervals",
    "user_interactions_last_18_intervals",
    "user_interactions_last_42_intervals",

    # Streamer behavior
    "streamer_interactions",
    "streamer_unique_viewers",
    "streamer_total_watch_minutes",
    "streamer_avg_watch_minutes",
    "streamer_popularity_share",
    "streamer_log_interactions",
    "streamer_log_viewers",
    "streamer_repeat_viewer_rate",
    "streamer_interactions_last_6",
    "streamer_interactions_last_18",
    "streamer_interactions_last_42",
    "streamer_interactions_prev_42",
    "streamer_trend_ratio",

    # User-streamer affinity
    "pair_interactions",
    "pair_watch_minutes",
    "pair_avg_watch_minutes",
    "pair_recency_intervals",
    "pair_active_span_intervals",
    "pair_repeat_strength",
    "pair_interactions_last_6",
    "pair_interactions_last_18",
    "pair_interactions_last_42",
    "seen_before",
    "pair_interaction_share",
    "pair_watch_share",
]

assert len(FEATURE_COLUMNS) == 36

HISTORY_SOURCE_K = 60
SVD_SOURCE_K = 300
POPULARITY_SOURCE_K = 200
MAX_CANDIDATES = 200

DEFAULT_RRF_K = 60
DEFAULT_SOURCE_WEIGHTS = {
    "history": 1.00,
    "svd": 1.00,
    "popularity": 0.55,
}
