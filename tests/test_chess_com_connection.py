"""Shared API transport tests, independent of model representation."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

from chessdotcom.errors import ChessDotComClientError
from requests.exceptions import ConnectionError, SSLError, Timeout

from training_data import ChessComConnection, DatasetConfig
from training_data.chess_com_connection import logger


class ConnectionTests(unittest.TestCase):
    """Retain transport, retry, ordering and concurrency regression coverage."""

    def test_connection_uses_config_and_preserves_player_order(self):
        """Check client configuration and deduplicated player order.

        Args:
            None.

        Returns:
            None; API calls and configured proxy/retry behavior are asserted.
        """
        config = DatasetConfig(
            country_codes=("ca", "CA", "US"), player_limit=3, max_workers=1
        )
        with patch(
            "training_data.chess_com_connection.ChessDotComClient"
        ) as client_class:
            connection = ChessComConnection(config)
        client = client_class.return_value
        client.get_country_players.side_effect = [
            SimpleNamespace(players=["Alice", "Bob"]),
            SimpleNamespace(players=["alice", "Carol", "Dave"]),
        ]
        self.assertEqual(connection.get_players(), ["Alice", "Bob", "Carol"])
        self.assertEqual(client.get_country_players.call_count, 2)
        client.get_country_players.assert_any_call("CA", tts=1.0)
        client.get_player_games_by_month.return_value = SimpleNamespace(games=[1])
        self.assertEqual(list(connection.iter_games("Alice")), [1])
        client.get_player_games_by_month.assert_called_once_with(
            "Alice", 2026, 10, tts=1.0
        )
        self.assertNotIn("timeout", client_class.call_args.kwargs["request_config"])
        self.assertEqual(client_class.call_args.kwargs["rate_limit_handler"].retries, 0)
        self.assertEqual(
            client_class.call_args.kwargs["request_config"]["proxies"],
            {"http": config.proxy_url, "https": config.proxy_url},
        )
        client.get_country_players.side_effect = RuntimeError("403 Forbidden")
        with self.assertRaisesRegex(RuntimeError, "403"):
            connection.get_players()

    def test_wrapper_requests_use_one_timeout(self):
        """Verify both endpoints use the SDK timeout and configured headers.

        Args:
            None.

        Returns:
            None; timeout, proxy and user-agent values are asserted.
        """
        connection = ChessComConnection(DatasetConfig(request_delay=0))
        with patch("chessdotcom.client.requests.get") as get:
            get.return_value.status_code = 200
            get.return_value.text = '{"players": ["Alice"]}'
            self.assertEqual(connection.get_players(), ["Alice"])
            self.assertEqual(get.call_args.kwargs["timeout"], 30)
            self.assertEqual(
                get.call_args.kwargs["proxies"],
                {
                    "http": connection.config.proxy_url,
                    "https": connection.config.proxy_url,
                },
            )
            self.assertEqual(
                get.call_args.kwargs["headers"]["User-Agent"],
                connection.config.user_agent,
            )
            get.return_value.text = '{"games": []}'
            self.assertEqual(list(connection.iter_games("Alice")), [])
            self.assertEqual(get.call_count, 2)
            self.assertEqual(get.call_args.kwargs["timeout"], 30)

    def test_transport_failures_retry_both_endpoints(self):
        """Retain transient transport retry and logging coverage.

        Args:
            None.

        Returns:
            None; bounded calls, backoff and resource names are asserted.
        """
        connection = ChessComConnection(DatasetConfig(request_delay=0))
        responses = [
            ConnectionError("Connection aborted", ConnectionResetError(10054)),
            Mock(status_code=200, text='{"players": ["Alice"]}'),
            Timeout("Timed out"),
            Mock(status_code=200, text='{"games": []}'),
        ]
        with (
            patch("chessdotcom.client.requests.get", side_effect=responses) as get,
            patch("retryable.time.sleep") as sleep,
            self.assertLogs("retryable.retry", level="WARNING") as logs,
        ):
            self.assertEqual(connection.get_players(), ["Alice"])
            self.assertEqual(list(connection.iter_games("Alice")), [])
        self.assertEqual(get.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.0, 5.0])
        self.assertIn("country/CA/players", logs.output[0])
        self.assertIn("player/Alice/games/2026/10", logs.output[1])

    def test_transport_retries_are_bounded_and_preserve_error(self):
        """Propagate the original exception when transport retries are exhausted.

        Args:
            None.

        Returns:
            None; original error identity and retry schedule are asserted.
        """
        connection = ChessComConnection(DatasetConfig(request_delay=0))
        reset = ConnectionError("Connection reset")
        with (
            patch("chessdotcom.client.requests.get", side_effect=reset) as get,
            patch("retryable.time.sleep") as sleep,
            self.assertLogs("retryable.retry", level="WARNING"),
            self.assertRaises(ConnectionError) as raised,
        ):
            connection.get_players()
        self.assertIs(raised.exception, reset)
        self.assertEqual(get.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.0, 10.0])

    def test_certificate_errors_are_not_retried(self):
        """Do not retry certificate validation failures.

        Args:
            None.

        Returns:
            None; one request and no backoff are asserted.
        """
        connection = ChessComConnection(DatasetConfig(request_delay=0))
        with (
            patch(
                "chessdotcom.client.requests.get", side_effect=SSLError("Certificate")
            ) as get,
            patch("retryable.time.sleep") as sleep,
            self.assertLogs("retryable.retry", level="WARNING"),
            self.assertRaises(SSLError),
        ):
            connection.get_players()
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_parallel_requests_overlap_with_bounded_workers_and_order(self):
        """Keep bounded concurrent downloads and deterministic result order.

        Args:
            None.

        Returns:
            None; observed worker peak, order and request settings are asserted.
        """
        config = DatasetConfig(
            country_codes=("CA", "US"),
            months=((2026, 9), (2026, 10)),
            request_delay=0,
            max_workers=2,
        )
        connection = ChessComConnection(config)
        barrier, lock = Barrier(2, timeout=5), Lock()
        active = peak = 0

        def respond(*, url, **kwargs):
            """Synchronize mocked responses to prove actual request overlap.

            Args:
                url: Requested endpoint.
                kwargs: Transport configuration forwarded by the SDK.

            Returns:
                Mock successful country or archive response.
            """
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait()
                self.assertEqual(
                    kwargs["proxies"],
                    {"http": config.proxy_url, "https": config.proxy_url},
                )
                self.assertEqual(kwargs["timeout"], 30)
                payload = (
                    {"players": [url.split("/")[-2]]}
                    if "/country/" in url
                    else {"games": []}
                )
                return Mock(status_code=200, text=json.dumps(payload))
            finally:
                with lock:
                    active -= 1

        with patch("chessdotcom.client.requests.get", side_effect=respond) as get:
            players = connection.get_players()
            self.assertEqual(players, ["CA", "US"])
            results = list(connection.iter_player_games(players))
        self.assertEqual(
            [username for username, _ in results], ["CA", "CA", "US", "US"]
        )
        self.assertEqual(get.call_count, 6)
        self.assertEqual(peak, config.max_workers)
        self.assertEqual(active, 0)

    def test_worker_progress_is_logged_before_earlier_request_finishes(self):
        """Keep per-worker progress visible while an earlier request is blocked.

        Args:
            None.

        Returns:
            None; completion order and worker/resource logs are asserted.
        """
        connection = ChessComConnection(DatasetConfig(request_delay=0, max_workers=2))
        first_started, release_first, second_logged = Event(), Event(), Event()

        def respond(*, url, **kwargs):
            """Block only the first archive request.

            Args:
                url: Requested archive endpoint.
                kwargs: Unused forwarded transport options.

            Returns:
                Empty successful archive response.
            """
            if "/Alice/" in url:
                first_started.set()
                if not release_first.wait(timeout=5):
                    raise TimeoutError("Test did not release the first request")
            return Mock(status_code=200, text='{"games": []}')

        original_info = logger.info

        def report(message, *args):
            """Signal Bob's download log without suppressing original logging.

            Args:
                message: Logger format string.
                args: Formatting values identifying the archive owner.

            Returns:
                None.
            """
            original_info(message, *args)
            if message.startswith("Downloaded") and args[1] == "Bob":
                second_logged.set()

        with (
            self.assertLogs("training_data.chess_com_connection", level="INFO") as logs,
            patch("chessdotcom.client.requests.get", side_effect=respond),
            patch.object(logger, "info", side_effect=report),
            ThreadPoolExecutor(max_workers=1) as consumer,
        ):
            future = consumer.submit(
                list, connection.iter_player_games(["Alice", "Bob"])
            )
            try:
                self.assertTrue(first_started.wait(timeout=5))
                self.assertTrue(second_logged.wait(timeout=5))
                self.assertFalse(future.done())
            finally:
                release_first.set()
            self.assertEqual(len(future.result(timeout=5)), 2)
        self.assertTrue(
            any(record.threadName.startswith("chess-api") for record in logs.records)
        )
        self.assertTrue(any("Request started: player/" in line for line in logs.output))
        self.assertTrue(
            any("Request completed: player/" in line for line in logs.output)
        )

    def test_retryable_handles_transient_http_errors_for_both_endpoints(self):
        """Retry supported transient statuses for country and archive endpoints.

        Args:
            None.

        Returns:
            None; configured exponential backoff and success are asserted.
        """
        config = DatasetConfig(
            request_delay=0, retry_delay=3, retry_backoff=3, max_workers=1
        )
        for status in (429, 500, 502, 503, 504):
            for endpoint in ("players", "games"):
                with self.subTest(status=status, endpoint=endpoint):
                    connection = ChessComConnection(config)
                    text = (
                        '{"players": ["Alice"]}'
                        if endpoint == "players"
                        else '{"games": []}'
                    )
                    responses = [Mock(status_code=status, text="{}", headers={})] * 2
                    with (
                        patch(
                            "chessdotcom.client.requests.get",
                            side_effect=[*responses, Mock(status_code=200, text=text)],
                        ) as get,
                        patch("retryable.time.sleep") as sleep,
                        self.assertLogs("retryable.retry", level="WARNING"),
                    ):
                        if endpoint == "players":
                            self.assertEqual(connection.get_players(), ["Alice"])
                        else:
                            self.assertEqual(list(connection.iter_games("Alice")), [])
                    self.assertEqual(get.call_count, 3)
                    self.assertEqual(
                        [call.args[0] for call in sleep.call_args_list], [3, 9]
                    )

    def test_http_retry_exhaustion_raises_instead_of_dropping_games(self):
        """Do not silently drop archive games after repeated HTTP failures.

        Args:
            None.

        Returns:
            None; terminal status and bounded retry count are asserted.
        """
        connection = ChessComConnection(DatasetConfig(request_delay=0))
        response = Mock(status_code=429, text="{}", headers={})
        with (
            patch("chessdotcom.client.requests.get", return_value=response) as get,
            patch("retryable.time.sleep") as sleep,
            self.assertLogs("retryable.retry", level="WARNING"),
            self.assertRaises(ChessDotComClientError) as raised,
        ):
            list(connection.iter_player_games(["Alice"]))
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(get.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10])

    def test_permanent_http_errors_are_not_retried(self):
        """Propagate nonretryable HTTP statuses after one request.

        Args:
            None.

        Returns:
            None; terminal status and absence of backoff are asserted.
        """
        for status in (400, 401, 403, 404, 410):
            connection = ChessComConnection(DatasetConfig(request_delay=0))
            response = Mock(status_code=status, text="{}", headers={})
            with (
                self.subTest(status=status),
                patch("chessdotcom.client.requests.get", return_value=response) as get,
                patch("retryable.time.sleep") as sleep,
                self.assertLogs("retryable.retry", level="WARNING"),
                self.assertRaises(ChessDotComClientError) as raised,
            ):
                connection.get_players()
            self.assertEqual(raised.exception.status_code, status)
            self.assertEqual(get.call_count, 1)
            sleep.assert_not_called()

    def test_retry_after_seconds_and_http_dates_are_honored(self):
        """Respect Retry-After in seconds and HTTP-date form.

        Args:
            None.

        Returns:
            None; server wait plus configured retry backoff are asserted.
        """
        for retry_after in ("7", "Thu, 01 Jan 2026 00:00:07 GMT"):
            connection = ChessComConnection(DatasetConfig(request_delay=0))
            failure = Mock(
                status_code=429, text="{}", headers={"Retry-After": retry_after}
            )
            success = Mock(status_code=200, text='{"players": ["Alice"]}')
            with (
                self.subTest(retry_after=retry_after),
                patch(
                    "chessdotcom.client.requests.get", side_effect=[failure, success]
                ),
                patch("retryable.time.sleep") as sleep,
                patch("training_data.chess_com_connection.datetime") as clock,
                self.assertLogs("retryable.retry", level="WARNING"),
            ):
                clock.now.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc)
                self.assertEqual(connection.get_players(), ["Alice"])
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [7, 5])

    def test_invalid_configuration(self):
        """Retain validation of shared connection configuration.

        Args:
            None.

        Returns:
            None; invalid configurations must raise before network use.
        """
        cases = (
            {"country_codes": ()},
            {"country_codes": ("USA",)},
            {"months": ()},
            {"months": ((2026, 13),)},
            {"months": ((2026, 10), (2026, 10))},
            {"player_limit": 0},
            {"request_delay": -1},
            {"retry_delay": -1},
            {"retries": -1},
            {"retry_backoff": 0.5},
            {"max_workers": 0},
            {"log_level": "INVALID"},
            {"user_agent": ""},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DatasetConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
