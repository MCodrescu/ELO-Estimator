"""Display discovery quotas separately from the archive search budget."""

from contextlib import ExitStack
from html import escape
from time import monotonic

from tqdm.auto import tqdm

try:
    from IPython import get_ipython
    from IPython.display import HTML, display
except ImportError:
    get_ipython = None
    HTML = None
    display = None


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
        self.category = category
        self.archive_limit = archives
        self.archives_processed = 0
        self.owner = ""
        self.status = {}
        self.handle = None
        self.last_render = float("-inf")
        shell = get_ipython() if get_ipython is not None else None
        self.notebook = enabled and getattr(shell, "kernel", None) is not None
        self.enabled = enabled
        self.stack = ExitStack()
        self.started = monotonic()
        self.last_gain = self.started
        self.summary = self.stack.enter_context(tqdm(
            total=sum(self.quotas.values()), desc=f"{category}: quotas filled",
            unit="player", disable=not enabled or self.notebook, position=0,
            leave=True, mininterval=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        ))
        self.archives = self.stack.enter_context(tqdm(
            total=archives, desc=f"{category}: archive budget",
            unit="archive", disable=not enabled or self.notebook, position=1,
            leave=True, mininterval=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, budget ETA {remaining}{postfix}]",
        ))
        self.bands = {
            band: self.stack.enter_context(tqdm(
                total=quota, desc=f"{category}: {band}", unit="player",
                disable=not enabled or self.notebook, position=index + 2,
                leave=True, mininterval=1,
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
        if exception_type is not None:
            self.status["stopped"] = "interrupted or failed"
        self._render(force=True)
        return self.stack.__exit__(exception_type, exception, traceback)

    def _render(self, force: bool = False) -> None:
        """Update one persistent notebook output, or throttle terminal refreshes.

        Args:
            force: Render final state even within the refresh interval.

        Returns:
            None; existing notebook content is updated without clearing the cell.
        """
        if not self.enabled:
            return
        now = monotonic()
        if not force and now - self.last_render < 1:
            return
        self.last_render = now
        if not self.notebook:
            for bar in (self.summary, self.archives, *self.bands.values()):
                bar.refresh()
            return
        total = sum(self.quotas.values())
        retained = sum(self.counts.values())
        status = escape(
            f"Remaining: {total - retained:,} | Bands filled: {self.status['bands_filled']} "
            f"| Quota ETA: {self.status['quota_ETA']}"
        )
        rows = "".join(
            f"<tr><td>{escape(band)}</td><td>"
            f'<progress value="{self.counts[band]}" max="{quota}"></progress></td>'
            f"<td>{self.counts[band]}/{quota}</td><td>{quota - self.counts[band]}</td></tr>"
            for band, quota in self.quotas.items()
        )
        stopped = escape(self.status.get("stopped", ""))
        content = HTML(
            f"<div><strong>{escape(self.category)}: quotas filled</strong> "
            f"{retained:,}/{total:,}<br>"
            f'<progress value="{retained}" max="{total}" style="width:100%;max-width:640px"></progress>'
            f"<div>{status}</div><table><thead><tr><th>Rating band</th>"
            f"<th>Progress</th><th>Accepted/Target</th><th>Remaining</th></tr></thead>"
            f"<tbody>{rows}</tbody></table><div>Archives processed: "
            f"{self.archives_processed:,}/{self.archive_limit:,} | Owner: {escape(self.owner)}</div>"
            f"<div>{stopped}</div></div>"
        )
        if self.handle is None:
            self.handle = display(content, display_id=True)
        else:
            self.handle.update(content)

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
        self.status = status
        self.summary.set_postfix(status, refresh=False)
        for band, bar in self.bands.items():
            bar.update(capped[band] - self.counts[band])
            bar.set_postfix(remaining=self.quotas[band] - capped[band], refresh=False)
        self.counts = capped
        self._render(force=reason is not None)

    def update(self, owner: str, counts: dict[str, int]) -> None:
        """Record one processed archive and refresh actual quota progress.

        Args:
            owner: Username whose monthly archive was processed.
            counts: Accepted counts, capped independently for each quota.

        Returns:
            None.
        """
        self.archives_processed += 1
        self.owner = owner
        self.archives.update(1)
        self.archives.set_postfix(owner=owner, refresh=False)
        self._refresh(counts)

    def finish(self, reason: str) -> None:
        """Leave final shortages visible even when the search stops early.

        Args:
            reason: Actual quota, budget, frontier or stagnation stopping reason.

        Returns:
            None.
        """
        self._refresh(self.counts, reason)
        self.archives.set_postfix_str(reason, refresh=False)