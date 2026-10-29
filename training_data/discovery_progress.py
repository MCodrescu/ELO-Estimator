"""Display discovery quotas separately from the archive search budget."""

from contextlib import ExitStack
from time import monotonic

from tqdm.auto import tqdm


class DiscoveryProgress:
    """Keep live per-band shortages and a conservative quota-time estimate."""

    def __init__(self, category: str, quotas: dict[str, int], archives: int, enabled: bool):
        """Prepare notebook-friendly displays without starting any search.

        Args:
            category: Time category being searched.
            quotas: Requested usernames per rating band, including bridge bands.
            archives: Maximum monthly archives to process.
            enabled: Whether progress should be rendered.

        Returns:
            None.
        """
        self.quotas = {band: quota for band, quota in quotas.items() if quota > 0}
        self.counts = dict.fromkeys(self.quotas, 0)
        self.stack = ExitStack()
        self.started = monotonic()
        self.last_gain = self.started
        self.summary = self.stack.enter_context(tqdm(
            total=sum(self.quotas.values()), desc=f"{category}: quotas filled",
            unit="player", disable=not enabled, position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        ))
        self.archives = self.stack.enter_context(tqdm(
            total=archives, desc=f"{category}: archive budget",
            unit="archive", disable=not enabled, position=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, budget ETA {remaining}{postfix}]",
        ))
        self.bands = {
            band: self.stack.enter_context(tqdm(
                total=quota, desc=f"{category}: {band}", unit="player",
                disable=not enabled, position=index + 2,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]",
            ))
            for index, (band, quota) in enumerate(self.quotas.items())
        }
        self._refresh(self.counts)

    def __enter__(self):
        """Return the active progress display.

        Args:
            None.

        Returns:
            This display with all progress bars open.
        """
        return self

    def __exit__(self, exception_type, exception, traceback):
        """Close every bar on completion, failure or interruption.

        Args:
            exception_type: Raised exception class, if any.
            exception: Raised exception instance, if any.
            traceback: Exception traceback, if any.

        Returns:
            False; exceptions are never suppressed.
        """
        return self.stack.__exit__(exception_type, exception, traceback)

    def _eta(self, counts: dict[str, int], now: float) -> str:
        """Estimate the slowest unfinished band from its observed acceptance rate.

        Args:
            counts: Capped accepted usernames by requested band.
            now: Monotonic time after the latest archive.

        Returns:
            Approximate duration, or unknown when progress cannot support an ETA.
        """
        pending = [band for band in self.quotas if counts[band] < self.quotas[band]]
        if not pending:
            return "00:00"
        if any(counts[band] == 0 for band in pending):
            return "unknown (unseen bands)"
        if now - self.last_gain >= 60:
            return "unknown (quota gains stalled)"
        elapsed = now - self.started
        remaining = max(
            (self.quotas[band] - counts[band]) * elapsed / counts[band]
            for band in pending
        )
        return "~" + tqdm.format_interval(remaining)

    def _refresh(self, counts: dict[str, int], reason: str | None = None) -> None:
        """Refresh quota counts without confusing them with encountered identities.

        Args:
            counts: Observed accepted usernames by band.
            reason: Optional final search stopping reason.

        Returns:
            None; all visible quota counts and shortages are updated.
        """
        now = monotonic()
        capped = {band: min(max(counts.get(band, 0), 0), quota) for band, quota in self.quotas.items()}
        retained = sum(capped.values())
        if retained > sum(self.counts.values()):
            self.last_gain = now
        filled = sum(capped[band] == quota for band, quota in self.quotas.items())
        self.summary.update(retained - sum(self.counts.values()))
        status = {
            "remaining": sum(self.quotas.values()) - retained,
            "bands_filled": f"{filled}/{len(self.quotas)}",
            "quota_ETA": self._eta(capped, now),
        }
        if reason is not None:
            status["quota_ETA"] = "00:00" if retained == sum(self.quotas.values()) else "not reached"
            status["stopped"] = reason
        self.summary.set_postfix(status)
        for band, bar in self.bands.items():
            bar.update(capped[band] - self.counts[band])
            bar.set_postfix(remaining=self.quotas[band] - capped[band])
        self.counts = capped

    def update(self, owner: str, counts: dict[str, int]) -> None:
        """Record one processed archive and refresh actual quota progress.

        Args:
            owner: Username whose monthly archive was processed.
            counts: Accepted counts, capped independently for each quota.

        Returns:
            None.
        """
        self.archives.update(1)
        self.archives.set_postfix(owner=owner)
        self._refresh(counts)

    def finish(self, reason: str) -> None:
        """Leave final shortages visible even when the search stops early.

        Args:
            reason: Actual quota, budget, frontier or stagnation stopping reason.

        Returns:
            None.
        """
        self._refresh(self.counts, reason)
        self.archives.set_postfix_str(reason)