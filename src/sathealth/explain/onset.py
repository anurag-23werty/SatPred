"""When did the deviation really start?

An alarm fires when the fused surprise reaches 3 (rarer than 1 in 1,000 nominal steps). Most
anomalies start earlier, at a level that is unusual but not yet alarming. The onset is found
by walking back from the alarm start while the surprise stays at or above ``level``
(default 1, i.e. rarer than 1 in 10), for at most ``max_lookback`` steps. It only looks at
steps up to the alarm, so it is available the moment the alarm is raised.
"""

from sathealth.data.schema import FloatArray


def estimate_onset(
    surprise: FloatArray, alarm_start: int, *, level: float = 1.0, max_lookback: int = 500
) -> int:
    if not 0 <= alarm_start < surprise.size:
        raise ValueError(f"alarm start {alarm_start} outside series of {surprise.size}")
    earliest = max(0, alarm_start - max_lookback)
    onset = alarm_start
    while onset - 1 >= earliest and surprise[onset - 1] >= level:
        onset -= 1
    return onset
