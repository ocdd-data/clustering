"""Reporting clock for the scheduled clustering jobs.

All Slack reports are read as Singapore-local: "this week" means the SGT week,
and the monthly summary is expected on the SGT 1st of the month.

The GitHub Actions runner clock is UTC, so a job cannot use ``datetime.today()``
to decide which week it is reporting on. The weekly crons deliberately fire on
*Sunday* evening UTC (20:xx) so that GitHub's best-effort scheduler queue delay
still lands the Slack post in the SGT Monday morning. On that Sunday-UTC clock
``today - today.weekday()`` walks *back* to the previous Monday, so every weekly
report came out a full week stale.

Anchoring to SGT fixes it: Sunday 20:09 UTC is already Monday 04:09 SGT, so the
reference Monday is the correct one, and any queue delay landing anywhere later
in that SGT week still resolves to the same Monday.

Singapore is UTC+8 year-round with no DST, so a fixed offset is used rather than
a named zone -- that keeps the job free of a tzdata dependency on Windows.
"""

from datetime import datetime, timedelta, timezone

SGT = timezone(timedelta(hours=8))


def sgt_today() -> datetime:
    """Current SGT date as a naive datetime at midnight.

    Naive so it stays interchangeable with the ``datetime.today()`` it replaces
    and with the date arithmetic downstream.
    """
    now = datetime.now(SGT)
    return datetime(now.year, now.month, now.day)


def sgt_reference_monday() -> datetime:
    """Monday of the current SGT week, as a naive datetime at midnight."""
    today = sgt_today()
    return today - timedelta(days=today.weekday())
