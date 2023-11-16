import io
import os
import sys
import asyncio
import logging
import datetime
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
from rich.align import Align


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

        self.start_time = datetime.datetime.now()

    async def render_ui(self):
        layout = Layout()

        layout.split_row(
            Layout(Text(" "), name="left"),
            Layout(Text(" "), name="right"),
        )

        layout["right"].size = None
        layout["right"].ratio = 2

        layout["left"].split_column(
            Layout(Text(" "), name="top"),
            Layout(Text(" "), name="bottom"),
        )

        with Live(layout, auto_refresh=False, transient=True) as live:
            while True:
                await asyncio.sleep(1)
                rates = self.coordinator.aggregate_rates()

                elapsed_time = datetime.datetime.now() - self.start_time
                elapsed_time = datetime.timedelta(seconds=elapsed_time.seconds)

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
                stats_table.add_section()
                stats_table.add_row("Elapsed Time", f"{elapsed_time}")

                layout["left"]["top"].update(
                    Align.center(stats_table, vertical="middle")
                )

                rate_table = Table(title="Rates (Hz)")
                rate_table.add_column("Metric")
                rate_table.add_column("Current")
                rate_table.add_column("Peak")
                rate_table.add_row(
                    "States Processed",
                    f"{rates.states_processed:.2f}",
                    f"{self.coordinator.max_rates.states_processed:.2f}",
                )
                rate_table.add_row(
                    "Tokens Processed",
                    f"{rates.tokens_processed:.2f}",
                    f"{self.coordinator.max_rates.tokens_processed:.2f}",
                )
                rate_table.add_row(
                    "Tokens Generated",
                    f"{rates.tokens_generated:.2f}",
                    f"{self.coordinator.max_rates.tokens_generated:.2f}",
                )

                layout["left"]["bottom"].update(
                    Align.center(rate_table, vertical="middle")
                )
                layout["right"].update(Panel(self.log_handler, title="Log"))

                live.refresh()
