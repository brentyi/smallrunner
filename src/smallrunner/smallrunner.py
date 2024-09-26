from __future__ import annotations

import atexit
import os
import re
import select
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, override

import pynvml
import tyro
from rich import print
from textual.app import App, ComposeResult
from textual.containers import Grid, ScrollableContainer
from textual.widgets import Log, Static, TabbedContent, TabPane


def main(
    shell_scripts: tuple[Path, ...],
    /,
    gpu_ids: Literal["all_free"] | tuple[int, ...] = "all_free",
    mem_ratio_threshold: float = 0.1,
    enforce_python: bool = True,
    jobs_per_gpu: int = 1,
) -> None:
    """The main entry point for the application.

    This function initializes the NVIDIA Management Library (NVML), determines available GPUs,
    parses the provided shell scripts into commands, and runs the SmallRunner app.

    Args:
        shell_scripts: Paths to shell scripts containing commands to run.
        gpu_ids: The GPU IDs to use, or "all_free" to use all available GPUs. Defaults to "all_free".
        mem_ratio_threshold: The memory usage threshold for considering a GPU as available. Defaults to 0.1.
        enforce_python: If True, ensures all commands start with 'python'. Defaults to True.
        jobs_per_gpu: Number of jobs to run simultaneously on each GPU. Defaults to 1.
    """
    pynvml.nvmlInit()
    if gpu_ids == "all_free":
        gpu_ids = get_available_gpus(mem_ratio_threshold)
    else:
        gpu_ids = tuple(
            gpu_id
            for gpu_id in gpu_ids
            if get_gpu_memory_usage(gpu_id) <= mem_ratio_threshold
        )
    commands = parse_commands(shell_scripts, enforce_python)

    app = SmallRunner(tuple(gpu_ids), tuple(commands), jobs_per_gpu)
    app.run()


@dataclass(frozen=True)
class Command:
    """Represents a command to be executed."""

    id: int  # Unique identifier for the command
    args: list[str]  # List of command arguments

    def __repr__(self) -> str:
        return f"[bold](id={self.id}, [cyan]{shlex.join(self.args)}[/cyan])[/bold]"


@dataclass
class FinishedJobInfo:
    """Represents information about a finished job."""

    command: Command
    gpu_id: int
    elapsed_time: float
    logdir: Path | None
    return_code: int


@dataclass(frozen=True)
class GlobalState:
    """Stores the global state of the application."""

    command_from_gpu_id: dict[
        int, list[str]
    ]  # Mapping of GPU IDs to their current commands
    start_time_from_gpu_id: dict[int, list[float]]
    logdir_from_gpu_id: dict[
        int, list[Path | None]
    ]  # Mapping of GPU IDs to their log directories
    finished_jobs: list[
        FinishedJobInfo
    ]  # List of dictionaries containing finished job information
    start_time: float  # Start time for all jobs
    show_on_exit: list[str]  # List of strings to show when smallrunner exits


class GpuOutputContainer(ScrollableContainer):
    """A widget that displays information about one GPU in the label.

    Args:
        gpu_id: The ID of the GPU to monitor.
        state: The global state object.
    """

    def __init__(self, gpu_id: int, job_index: int, state: GlobalState) -> None:
        super().__init__(classes="bordered-white")
        self._gpu_id = gpu_id
        self._job_index = job_index
        self._state = state

    # @override
    def on_mount(self) -> None:
        self._update_label()
        self.set_interval(1.0, self._update_label)

    def _update_label(self) -> None:
        """Update the GPU label with current utilization, memory information, and elapsed time."""
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_id)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = float(mem_info.used) / (1024**3)
        mem_total = float(mem_info.total) / (1024**3)

        label_parts = [
            f"GPU{self._gpu_id}-{self._job_index}",
            f"{gpu_util}%",
            f"{mem_used:.1f}/{mem_total:.1f}G",
        ]

        cmd = self._state.command_from_gpu_id[self._gpu_id][self._job_index]
        start_time = self._state.start_time_from_gpu_id[self._gpu_id][self._job_index]

        if cmd == "":
            label_parts.append("idle")
        else:
            elapsed_time = time.time() - start_time if start_time > 0 else 0
            elapsed_str = format_elapsed_time(elapsed_time)
            label_parts.append(elapsed_str)
            label_parts.append(cmd if not cmd.startswith("python ") else cmd[7:])

        self.border_title = (
            "[bold reverse] " + " • ".join(label_parts) + " [/bold reverse]"
        )


class SummaryDisplay(Static):
    def __init__(self, state: GlobalState, runner: SmallRunner) -> None:
        super().__init__()
        self._state = state
        self._runner = runner
        self._finish_time = None

    def on_mount(self) -> None:
        self.update_summary()
        self.set_interval(1.0, self.update_summary)

    def update_summary(self) -> None:
        running_jobs = sum(1 for cmd in self._state.command_from_gpu_id.values() if cmd)
        finished_jobs = len(self._state.finished_jobs)
        remaining_jobs = len(self._runner._commands_left)

        # Calculate average elapsed time for finished jobs
        if finished_jobs:
            elapsed_times = [job.elapsed_time for job in self._state.finished_jobs]
            avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
        else:
            avg_elapsed_time = 0
        avg_elapsed_str = format_elapsed_time(avg_elapsed_time)

        # Calculate total elapsed time
        if self._finish_time is None and remaining_jobs == 0 and running_jobs == 0:
            current_time = time.time()
            self._finish_time = current_time

        if self._finish_time is not None:
            # Total elapsed time is frozen at finish time
            total_elapsed_time = self._finish_time - self._state.start_time
        else:
            # Total elapsed time is still counting
            current_time = time.time()
            total_elapsed_time = current_time - self._state.start_time

        total_elapsed_str = format_elapsed_time(total_elapsed_time)

        # Calculate number of errors
        errors = sum(1 for job in self._state.finished_jobs if job.return_code != 0)

        summary = (
            " [dim]•[/dim] ".join(
                [
                    (
                        f"[dim]Finished:[/dim] {finished_jobs} [red]({errors} {'errors' if errors > 1 else 'error'})[/red]"
                        if errors > 0
                        else f"[dim]Finished:[/dim] {finished_jobs}"
                    ),
                    f"[dim]Running:[/dim] {running_jobs}",
                    f"[dim]Remaining:[/dim] {remaining_jobs}",
                    f"[dim]Avg time:[/dim] {avg_elapsed_str}",
                    f"[dim]Total time:[/dim] {total_elapsed_str}",
                ]
            )
            + "\n"
        )
        self.update(summary)


class SmallRunner(App):
    CSS = """
    .bordered-white {
        border: round white;
    }

    ScrollableContainer {
        scrollbar-color: white;
        scrollbar-size: 1 1;
        scrollbar-background: $surface;
    }
    """

    def __init__(
        self,
        cuda_device_ids: tuple[int, ...],
        commands: tuple[Command, ...],
        jobs_per_gpu: int = 1,
    ) -> None:
        super().__init__()

        if len(commands) < len(cuda_device_ids) * jobs_per_gpu:
            cuda_device_ids = cuda_device_ids[
                : len(commands) // jobs_per_gpu
                + (1 if len(commands) % jobs_per_gpu else 0)
            ]

        self._cuda_device_ids = cuda_device_ids
        self._commands = commands
        self._jobs_per_gpu = jobs_per_gpu
        self._state = GlobalState(
            command_from_gpu_id={id: [""] * jobs_per_gpu for id in cuda_device_ids},
            start_time_from_gpu_id={id: [0.0] * jobs_per_gpu for id in cuda_device_ids},
            logdir_from_gpu_id={id: [None] * jobs_per_gpu for id in cuda_device_ids},
            finished_jobs=[],
            start_time=time.time(),
            show_on_exit=[],
        )
        self._gpu_free_state = {id: [True] * jobs_per_gpu for id in cuda_device_ids}
        self._commands_left = list(reversed(commands))
        self._running_commands: dict[int, list[Command | None]] = {
            id: [None] * jobs_per_gpu for id in cuda_device_ids
        }
        self._commands_finished = []
        atexit.register(self._handle_exit)

    def _handle_exit(self):
        print("\n[bold]smallrunner[/bold] exiting. Showing on-exit information:")
        for item in self._state.show_on_exit:
            print("\t", item)

    def on_mount(self) -> None:
        self.set_interval(0.5, self._poll_update)
        self._poll_update()

    def _run_job(self, gpu_id: int, job_index: int, args: tuple[str, ...]) -> None:
        logdir = Path(
            f"/tmp/smallrunner_logs/{time.strftime('%Y%m%d_%H%M%S')}_{gpu_id}_{job_index}"
        )
        logdir.mkdir(parents=True, exist_ok=True)

        # Useful if smallrunner exits...
        self._state.show_on_exit.append(
            f"Started [cyan]{shlex.join(args)}[/cyan] on GPU [bold]{gpu_id}[/bold]-{job_index}, logging to [blue]{logdir}[/blue]"
        )

        self._state.logdir_from_gpu_id[gpu_id][job_index] = logdir
        self._state.command_from_gpu_id[gpu_id][job_index] = shlex.join(args)
        self._state.start_time_from_gpu_id[gpu_id][job_index] = time.time()
        log_display = self.query_one(f"#log-display-{gpu_id}-{job_index}", Log)
        log_display.clear()

        with open(logdir / "stdout.log", "w") as stdout_f, open(
            logdir / "stderr.log", "w"
        ) as stderr_f:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(
                    os.environ,
                    CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    CUDA_VISIBLE_DEVICES=str(gpu_id),
                ),
                bufsize=1,
                universal_newlines=True,
            )

            self._handle_process_output(process, stdout_f, stderr_f, log_display)

        self._gpu_free_state[gpu_id][job_index] = True
        self._state.command_from_gpu_id[gpu_id][job_index] = ""
        log_display.clear()

        finished_command = self._running_commands[gpu_id][job_index]
        self._running_commands[gpu_id][job_index] = None
        end_time = time.time()
        elapsed_time = end_time - self._state.start_time_from_gpu_id[gpu_id][job_index]
        elapsed_str = format_elapsed_time(elapsed_time)

        assert finished_command is not None
        assert isinstance(process.returncode, int)
        info = FinishedJobInfo(
            command=finished_command,
            gpu_id=gpu_id,
            elapsed_time=elapsed_time,
            logdir=self._state.logdir_from_gpu_id[gpu_id][job_index],
            return_code=process.returncode,
        )
        self._state.finished_jobs.append(info)

        status_part = (
            f"[red]failed with code {info.return_code}[/red]"
            if info.return_code != 0
            else "[green]completed[/green]"
        )
        time_part = f"in [bold]{elapsed_str}[/bold]"
        log_part = f"logs saved to [blue]{info.logdir}[/blue]"

        finished_message = f"{finished_command} {status_part} {time_part}, {log_part}"
        self._commands_finished.append(finished_message)
        self._state.show_on_exit.append(finished_message)

    def _handle_process_output(
        self,
        process: subprocess.Popen,
        stdout_f: IO,
        stderr_f: IO,
        log_display: Log,
    ) -> None:
        def process_stream(stream: IO, file: IO) -> None:
            line = stream.readline()
            if line:
                log_display.write(line)
                file.write(line)
                file.flush()

        assert process.stdout is not None
        assert process.stderr is not None
        while True:
            rlist, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

            for ready_stream in rlist:
                if ready_stream == process.stdout:
                    process_stream(process.stdout, stdout_f)
                elif ready_stream == process.stderr:
                    process_stream(process.stderr, stderr_f)

            if process.poll() is not None:
                break

        # Read any remaining output
        for stream, file in [
            (process.stdout, stdout_f),
            (process.stderr, stderr_f),
        ]:
            for _ in stream:
                process_stream(stream, file)

    def _poll_update(self) -> None:
        # Based on the terminal dimensions and number of GPUs, update `grid-size` for the output grid.
        # Assume minimum width of 60 characters per GPU.
        terminal_width, terminal_height = self.size
        num_gpus = len(self._cuda_device_ids)
        columns = max(1, min(num_gpus, terminal_width // 60))
        output_grid = self.query_one(".output-grid", Grid)
        output_grid.styles.grid_size_columns = columns

        for gpu_id in self._cuda_device_ids:
            for job_index in range(self._jobs_per_gpu):
                if not self._gpu_free_state[gpu_id][job_index]:
                    continue

                if self._commands_left:
                    command = self._commands_left.pop()
                    self._running_commands[gpu_id][job_index] = command
                    self._gpu_free_state[gpu_id][job_index] = False
                    threading.Thread(
                        target=self._run_job, args=(gpu_id, job_index, command.args)
                    ).start()

        self._update_list("#list-waiting", self._commands_left)
        self._update_list(
            "#list-running",
            [
                f"GPU {gpu_id}-{job_index}: {command}, logs to [blue]{self._state.logdir_from_gpu_id[gpu_id][job_index]}[/blue]"
                for gpu_id in self._cuda_device_ids
                for job_index, command in enumerate(self._running_commands[gpu_id])
                if command is not None
            ],
        )
        self._update_list("#list-finished", self._commands_finished)

    def _update_list(self, selector: str, items: list) -> None:
        list_widget = self.query_one(selector, Static)
        list_widget.update("\n".join(map(str, items)))

    @override
    def compose(self) -> ComposeResult:
        with TabbedContent():
            yield from self._create_log_tab("Outputs")
            yield from self._create_queue_tab()

    def _create_log_tab(self, title: str) -> ComposeResult:
        with TabPane(title):
            yield SummaryDisplay(self._state, self)
            with Grid(classes="output-grid"):
                for i in self._cuda_device_ids:
                    for j in range(self._jobs_per_gpu):
                        with GpuOutputContainer(i, j, self._state):
                            yield Log(
                                id=f"log-display-{i}-{j}",
                                max_lines=100,
                                auto_scroll=True,
                                classes="log-class",
                            )

    def _create_queue_tab(self) -> ComposeResult:
        with TabPane("Queue"):
            yield SummaryDisplay(self._state, self)
            with Grid():
                for title, id in [
                    ("Waiting", "waiting"),
                    ("Running", "running"),
                    ("Finished", "finished"),
                ]:
                    with ScrollableContainer(classes="bordered-white") as container:
                        container.border_title = (
                            f"[bold reverse] {title} [/bold reverse]"
                        )
                        yield Static(id=f"list-{id}")


def get_available_gpus(mem_ratio_threshold: float) -> tuple[int, ...]:
    """Determine the tuple of available GPU IDs based on memory usage and visibility.

    This function filters GPUs based on their memory usage and the CUDA_VISIBLE_DEVICES
    environment variable.

    Args:
        mem_ratio_threshold: The maximum memory usage ratio for a GPU to be considered available.

    Returns:
        A tuple of available GPU IDs.
    """
    gpu_ids = [
        i
        for i in range(pynvml.nvmlDeviceGetCount())
        if get_gpu_memory_usage(i) <= mem_ratio_threshold
    ]
    print(f"After filtering by memory threshold: {gpu_ids}")

    return tuple(sorted(gpu_ids))


def get_gpu_memory_usage(gpu_index: int) -> float:
    """Get the memory usage ratio for a specific GPU.

    Args:
        gpu_index: The index of the GPU to check.

    Returns:
        The ratio of used memory to total memory for the specified GPU.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return float(info.used) / float(info.total)


def parse_commands(
    shell_scripts: tuple[Path, ...], enforce_python: bool
) -> list[Command]:
    """Parse shell scripts into a list of Command objects.

    This function reads the contents of the provided shell scripts, splits them into lines,
    and creates Command objects for each non-empty, non-comment line.

    Args:
        shell_scripts: Paths to shell scripts containing commands.
        enforce_python: If True, ensures all commands start with 'python'.

    Returns:
        A list of Command objects representing the parsed commands.

    Raises:
        AssertionError: If enforce_python is True and a command doesn't start with 'python'.
    """
    script_contents = "\n".join(script.read_text() for script in shell_scripts)
    lines = re.split(r"(?<!\\)\n", script_contents)

    commands = []
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("#"):
            if enforce_python:
                assert line.startswith(
                    "python "
                ), f"Line {i+1} must start with 'python' when enforce_python is True"
            commands.append(Command(i, shlex.split(line)))

    return commands


def format_elapsed_time(elapsed_time: float) -> str:
    """
    Format elapsed time into a human-readable string.

    Args:
        elapsed_time: The elapsed time in seconds.

    Returns:
        A formatted string representing the elapsed time.
         - Less than 60 seconds: "Xs"
         - Less than 3600 seconds: "Xm Ys"
         - 3600 seconds or more: "Xh Ym Zs"
         Where X, Y, and Z are integer values.

    Examples:
        >>> format_elapsed_time(45)
        '45s'
        >>> format_elapsed_time(125)
        '2m 5s'
        >>> format_elapsed_time(3725)
        '1h 2m 5s'
    """
    if elapsed_time < 60:
        return f"{elapsed_time:.0f}s"
    elif elapsed_time < 3600:
        minutes, seconds = divmod(elapsed_time, 60)
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


entrypoint = lambda: tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
