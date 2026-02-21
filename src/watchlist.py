"""
watchlist.py — Background polling engine for trade watchlist queries.

Runs as an async task, offloads HTTP to thread executor (using requests),
and broadcasts results via WebSocket.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Coroutine, Optional
from urllib.parse import quote

import requests

from config import (
    TRADE_API_BASE,
    WATCHLIST_DEFAULT_POLL_INTERVAL,
    WATCHLIST_FETCH_COUNT,
    WATCHLIST_MAX_QUERIES,
    WATCHLIST_MIN_REQUEST_INTERVAL,
)

logger = logging.getLogger("watchlist")

USER_AGENT = "POE2-Price-Overlay-Watchlist/1.0"


@dataclass
class WatchlistResult:
    query_id: str
    total: int = 0
    listings: list = field(default_factory=list)
    price_low: Optional[str] = None
    price_high: Optional[str] = None
    last_checked: Optional[float] = None
    error: Optional[str] = None
    search_id: str = ""
    trade_url: str = ""


class WatchlistWorker:
    """Background poller for trade watchlist queries."""

    def __init__(self, league: str, broadcast_fn: Callable[..., Coroutine],
                 log_buffer=None):
        self.league = league
        self._broadcast = broadcast_fn
        self._log_buffer = log_buffer  # server's deque for log persistence
        self._queries: list[dict] = []
        self._poll_interval = WATCHLIST_DEFAULT_POLL_INTERVAL
        self._results: dict[str, WatchlistResult] = {}
        self._last_request_time = 0.0
        self._retry_after_until = 0.0
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._force_refresh_ids: set[str] = set()
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self.query_states: dict[str, dict] = {}  # per-query state tracking

    def set_session_id(self, poesessid: str):
        """Set or clear the POESESSID cookie for authenticated trade fetches."""
        if poesessid:
            self._session.cookies.set("POESESSID", poesessid, domain=".pathofexile.com")
        else:
            self._session.cookies.clear()

    def update_queries(self, queries: list[dict], poll_interval: int = None):
        """Update the query list (called when settings change)."""
        self._queries = queries[:WATCHLIST_MAX_QUERIES]
        if poll_interval is not None:
            self._poll_interval = max(60, poll_interval)
        # Remove results/states for deleted queries, init states for new ones
        active_ids = {q.get("id") for q in self._queries}
        for qid in list(self._results.keys()):
            if qid not in active_ids:
                del self._results[qid]
        for qid in list(self.query_states.keys()):
            if qid not in active_ids:
                del self.query_states[qid]
        for q in self._queries:
            qid = q.get("id")
            if qid and qid not in self.query_states:
                state = "idle" if q.get("enabled", True) else "disabled"
                self.query_states[qid] = {"state": state}

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the async polling task."""
        if self._task and not self._task.done():
            return
        self._loop = loop
        self._task = loop.create_task(self._poll_loop())
        logger.info("Watchlist worker started")

    def stop(self):
        """Cancel polling."""
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Watchlist worker stopped")

    def force_refresh(self, query_id: str):
        """Queue an immediate refresh for a single query."""
        self._force_refresh_ids.add(query_id)

    def get_results(self) -> dict[str, dict]:
        """Return all cached results as dicts."""
        return {qid: asdict(r) for qid, r in self._results.items()}

    def get_query_states(self) -> dict[str, dict]:
        """Return per-query state info for the frontend."""
        return dict(self.query_states)

    async def _set_query_state(self, qid: str, state: str, **kwargs):
        """Update a query's state and broadcast to clients."""
        self.query_states[qid] = {"state": state, **kwargs}
        try:
            await self._broadcast({
                "type": "watchlist_state",
                "states": self.get_query_states(),
            })
        except Exception:
            pass

    async def _poll_loop(self):
        """Main polling loop — round-robins through enabled queries."""
        try:
            # Brief startup delay
            await asyncio.sleep(5)

            while True:
                # Handle force-refresh requests first
                while self._force_refresh_ids:
                    qid = self._force_refresh_ids.pop()
                    query = next((q for q in self._queries if q.get("id") == qid), None)
                    if query and query.get("enabled", True):
                        await self._execute_and_broadcast(query)

                # Regular polling cycle
                for query in self._queries:
                    qid = query.get("id")
                    if not qid:
                        continue

                    if not query.get("enabled", True):
                        if self.query_states.get(qid, {}).get("state") != "disabled":
                            await self._set_query_state(qid, "disabled")
                        continue

                    # Check if this query was polled recently enough
                    result = self._results.get(qid)
                    if result and result.last_checked:
                        elapsed = time.time() - result.last_checked
                        if elapsed < self._poll_interval:
                            continue

                    await self._execute_and_broadcast(query)

                    # Check for force refreshes between queries
                    if self._force_refresh_ids:
                        break

                await asyncio.sleep(15)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Watchlist poll loop error: {e}", exc_info=True)
            try:
                await self._broadcast_log(f"Watchlist: poll loop crashed — {e}", "#a83232")
            except Exception:
                pass

    async def _execute_and_broadcast(self, query: dict):
        """Execute a query and broadcast the result."""
        qid = query["id"]
        label = query.get("label", qid)
        # Set state to querying
        await self._set_query_state(qid, "querying")
        # Log start to dashboard console
        await self._broadcast_log(f"Watchlist: searching \"{label}\"...")
        result = await self._loop.run_in_executor(None, self._execute_query, query)
        self._results[qid] = result
        # Log result to dashboard console
        if result.error:
            await self._set_query_state(qid, "error", error=result.error)
            await self._broadcast_log(f"Watchlist: \"{label}\" — {result.error}", "#a83232")
        elif result.total == 0:
            next_poll = time.time() + self._poll_interval
            await self._set_query_state(qid, "cooldown", next_poll=next_poll)
            await self._broadcast_log(f"Watchlist: \"{label}\" — no results found", "#8c7a5c")
        else:
            next_poll = time.time() + self._poll_interval
            await self._set_query_state(qid, "cooldown", next_poll=next_poll)
            price_info = result.price_low or "N/A"
            await self._broadcast_log(
                f"Watchlist: \"{label}\" — {result.total} listed, cheapest {price_info}",
                "#4a7c59",
            )
        try:
            await self._broadcast({
                "type": "watchlist_result",
                "result": asdict(result),
            })
        except Exception as e:
            logger.warning(f"Watchlist broadcast error: {e}")

    async def _broadcast_log(self, message: str, color: str = "#6b8f71"):
        """Send a log entry to the dashboard console panel + log buffer."""
        log_entry = {
            "time": time.strftime("%H:%M:%S"),
            "message": message,
            "color": color,
        }
        if self._log_buffer is not None:
            self._log_buffer.append(log_entry)
        try:
            await self._broadcast({"type": "log", **log_entry})
        except Exception:
            pass

    def _rate_wait(self):
        """Enforce minimum interval between requests + respect 429 backoff."""
        now = time.time()

        # Respect Retry-After backoff
        if now < self._retry_after_until:
            wait = self._retry_after_until - now
            logger.info(f"Watchlist rate-limited, waiting {wait:.1f}s")
            time.sleep(wait)

        # Enforce minimum interval
        elapsed = time.time() - self._last_request_time
        if elapsed < WATCHLIST_MIN_REQUEST_INTERVAL:
            time.sleep(WATCHLIST_MIN_REQUEST_INTERVAL - elapsed)

        self._last_request_time = time.time()

    def _execute_query(self, query: dict) -> WatchlistResult:
        """Run a single trade search + fetch cycle (blocking, called from executor)."""
        qid = query.get("id", "unknown")
        body = query.get("body")
        if not body:
            return WatchlistResult(query_id=qid, error="Empty query body",
                                   last_checked=time.time())

        league_encoded = quote(self.league, safe="")
        search_url = f"{TRADE_API_BASE}/search/poe2/{league_encoded}"

        try:
            # Step 1: POST search
            self._rate_wait()
            resp = self._session.post(search_url, json=body, timeout=30)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "60"))
                self._retry_after_until = time.time() + retry_after
                logger.warning(f"Watchlist 429, backing off {retry_after}s")
                return WatchlistResult(query_id=qid,
                                       error=f"Rate limited ({retry_after}s)",
                                       last_checked=time.time())
            if resp.status_code != 200:
                # Extract readable error from API response
                api_error = ""
                try:
                    err_data = resp.json()
                    api_error = err_data.get("error", {}).get("message", "")
                except Exception:
                    pass
                error_msg = api_error or f"HTTP {resp.status_code}"
                logger.warning(f"Watchlist search {resp.status_code}: {resp.text[:200]}")
                return WatchlistResult(query_id=qid,
                                       error=error_msg,
                                       last_checked=time.time())

            data = resp.json()
            search_id = data.get("id", "")
            result_ids = data.get("result", [])
            total = data.get("total", 0)

            trade_url = f"https://www.pathofexile.com/trade2/search/poe2/{league_encoded}/{search_id}" if search_id else ""

            if not result_ids:
                return WatchlistResult(query_id=qid, total=total,
                                       last_checked=time.time(),
                                       search_id=search_id, trade_url=trade_url)

            # Step 2: GET fetch (first N results)
            fetch_ids = result_ids[:WATCHLIST_FETCH_COUNT]
            ids_str = ",".join(fetch_ids)
            fetch_url = f"{TRADE_API_BASE}/fetch/{ids_str}?query={search_id}"

            self._rate_wait()
            resp = self._session.get(fetch_url, timeout=30)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "60"))
                self._retry_after_until = time.time() + retry_after
                logger.warning(f"Watchlist fetch 429, backing off {retry_after}s")
                return WatchlistResult(query_id=qid, total=total,
                                       error=f"Rate limited ({retry_after}s)",
                                       last_checked=time.time())
            if resp.status_code != 200:
                return WatchlistResult(query_id=qid, total=total,
                                       error=f"Fetch error: HTTP {resp.status_code}",
                                       last_checked=time.time())

            fetch_data = resp.json()

            # Step 3: Extract listings
            listings = []
            prices = []
            for item in fetch_data.get("result", []):
                listing = item.get("listing", {})
                price_info = listing.get("price", {})
                account = listing.get("account", {})
                amount = price_info.get("amount", 0)
                currency = price_info.get("currency", "")

                price_str = f"{amount} {currency}" if amount else "N/A"
                prices.append((amount, currency))

                whisper = listing.get("whisper", "")
                indexed = listing.get("indexed", "")

                # Item info for display
                item_data = item.get("item", {})
                item_name = item_data.get("name", "")
                type_line = item_data.get("typeLine", "")
                display_name = f"{item_name} {type_line}".strip()

                listings.append({
                    "price": price_str,
                    "amount": amount,
                    "currency": currency,
                    "account": account.get("name", ""),
                    "character": account.get("lastCharacterName", ""),
                    "whisper": whisper,
                    "indexed": indexed,
                    "item_name": display_name,
                    "whisper_token": listing.get("whisper_token", ""),
                    "hideout_token": listing.get("hideout_token", ""),
                })

            # Compute price range
            price_low = None
            price_high = None
            if prices:
                first_currency = prices[0][1] if prices else ""
                same_currency = [p for p in prices if p[1] == first_currency and p[0]]
                if same_currency:
                    amounts = [p[0] for p in same_currency]
                    price_low = f"{min(amounts)} {first_currency}"
                    price_high = f"{max(amounts)} {first_currency}"

            return WatchlistResult(
                query_id=qid,
                total=total,
                listings=listings,
                price_low=price_low,
                price_high=price_high,
                last_checked=time.time(),
                search_id=search_id,
                trade_url=trade_url,
            )

        except Exception as e:
            logger.error(f"Watchlist query {qid} error: {e}", exc_info=True)
            return WatchlistResult(query_id=qid, error=str(e),
                                   last_checked=time.time())
