"""

Demonstrates the use of multiple Progress instances in a single Live display.    

"""

from time import sleep

from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table


class LiveBar:
    def __init__(self, jobdict: dict = None):
        self.progress_table = None
        self.overall_task = None
        self.overall_progress = None

        self.job_progress = Progress(
            "{task.description}",
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )

    def start(self):
        total = sum(task.total for task in self.job_progress.tasks)
        self.overall_progress = Progress()
        self.overall_task = self.overall_progress.add_task("All Jobs", total=int(total))
        self.progress_table = Table.grid()
        self.progress_table.add_row(
            Panel.fit(
                self.overall_progress, title="Overall Progress", border_style="green", padding=(2, 2)
            ),
            Panel.fit(self.job_progress, title="[b]Jobs", border_style="red", padding=(1, 2)),
        )
        with Live(self.progress_table, refresh_per_second=10):
            while not self.overall_progress.finished:
                sleep(0.1)
                for job in self.job_progress.tasks:
                    if not job.finished:
                        self.job_progress.advance(job.id)

                completed = sum(task.completed for task in self.job_progress.tasks)
                self.overall_progress.update(self.overall_task, completed=completed)


bar = LiveBar()
bar.job_progress.add_task("[green]Cooking", total=10)
bar.job_progress.add_task("[magenta]Baking", total=10)
bar.job_progress.add_task("[cyan]Mixing", total=30)
bar.start()
bar.start()
