import io
import os
import sys
import asyncio
import logging
import itertools
import collections

from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.logging import RichHandler
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.segment import Segment


class LogHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.records = collections.deque(maxlen=1000)

    def emit(self, record):
        self.records.append(record)

    def __rich_console__(self, console, options):
        for record in itertools.islice(
            self.records, max(0, len(self.records) - options.height), len(self.records)
        ):
            text = Text.from_markup(super().format(record))
            text.truncate(options.max_width, overflow="ellipsis", pad=False)
            yield text


class CoordinatorUI:
    def __init__(self, coordinator, logger):
        self.coordinator = coordinator
        self.log_handler = LogHandler()
        logger.addHandler(self.log_handler)

    async def render_ui(self):
        layout = Layout()
        layout.split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split_column(
            Layout(name="top"),
            Layout(name="bottom"),
        )

        with Live(layout, auto_refresh=False, transient=True) as live:
            while True:
                await asyncio.sleep(1)
                rates = self.coordinator.aggregate_rates()

                stats_table = Table(title="Status")
                stats_table.add_column("Metric")
                stats_table.add_column("Value")

                stats_table.add_row(
                    "Active Workers", f"{len(self.coordinator.workers)}"
                )

                stats_table.add_row(
                    "\u03a3 States Processed",
                    f"{self.coordinator.aggregate_stats.states_processed:.2f}",
                )

                stats_table.add_row(
                    "\u03a3 Tokens Processed",
                    f"{self.coordinator.aggregate_stats.tokens_processed:.2f}",
                )

                stats_table.add_row(
                    "\u03a3 Tokens Generated",
                    f"{self.coordinator.aggregate_stats.tokens_generated:.2f}",
                )

                stats_table.add_row("Active Prompts", "N/A")

                layout["left"]["top"].update(stats_table)

                rate_table = Table(title="Rates")
                rate_table.add_column("Metric")
                rate_table.add_column("Rate (Hz)")
                rate_table.add_row("States Processed", f"{rates.states_processed:.2f}")
                rate_table.add_row("Tokens Processed", f"{rates.tokens_processed:.2f}")
                rate_table.add_row("Tokens Generated", f"{rates.tokens_generated:.2f}")

                layout["left"]["bottom"].update(rate_table)
                layout["right"].update(Panel(self.log_handler, title="Log"))

                live.refresh()
