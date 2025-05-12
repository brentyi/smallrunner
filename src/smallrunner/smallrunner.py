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
from typing import IO, Dict, List, Literal, Optional, override

import pynvml
import tyro
from rich import print
from textual.app import App, ComposeResult
from textual.command import Hit, Hits, Provider
from textual.containers import Grid, ScrollableContainer
from textual.widgets import Log, Static, TabbedContent, TabPane
from textual_fspicker import FileOpen


def main(
    shell_scripts: tuple[Path, ...],
    /,
    gpu_ids: Literal["all_free"] | tuple[int, ...] = "all_free",
    job_index_cond: str = "i>=0",
    mem_ratio_threshold: float = 0.1,
    enforce_python: bool = True,
    concurrent_jobs: int = 1,
    gpus_per_job: int = 1,
    use_topology: bool = True,
) -> None:
    """The main entry point for the application.

    This function initializes the NVIDIA Management Library (NVML), determines available GPUs,
    parses the provided shell scripts into commands, and runs the SmallRunner app.

    Args:
        shell_scripts: Paths to shell scripts containing commands to run.
        gpu_ids: The GPU IDs to use, or "all_free" to use all available GPUs. Defaults to "all_free".
        job_index_cond: The condition for running a job based on its index. Defaults to "i>=0" (all jobs).
        mem_ratio_threshold: The memory usage threshold for considering a GPU as available. Defaults to 0.1.
        enforce_python: If True, ensures all commands start with 'python'. Defaults to True.
        concurrent_jobs: Number of jobs to run concurrently on each primary GPU. Defaults to 1.
        gpus_per_job: Number of GPUs to allocate to each job. Defaults to 1.
        use_topology: If True, use GPU topology information to optimize GPU allocation. Defaults to True.
    """
    # Set up signal handling for cleaner shutdown
    import signal

    def handle_sigint(sig, frame):
        print("\nReceived interrupt signal, exiting gracefully...")
        # Just let the signal propagate to the app's exit handler
        # The app will set _is_shutting_down to True

    # Register the signal handler
    signal.signal(signal.SIGINT, handle_sigint)
    pynvml.nvmlInit()
    if gpu_ids == "all_free":
        gpu_ids = get_available_gpus(mem_ratio_threshold)
    else:
        gpu_ids = tuple(
            gpu_id
            for gpu_id in gpu_ids
            if get_gpu_memory_usage(gpu_id) <= mem_ratio_threshold
        )
    commands = [
        cmd
        for i, cmd in enumerate(parse_commands(shell_scripts, enforce_python))
        if eval(job_index_cond, {"i": i}, {})
    ]

    # Get GPU topology information if using multiple GPUs per job
    topology: Optional[Dict[int, List[int]]] = None
    if gpus_per_job > 1 and use_topology:
        try:
            topology = get_gpu_topology()
            print("Using GPU topology information for optimized GPU allocation")
        except Exception as e:
            print(
                f"[yellow]Warning: Could not get GPU topology information: {e}[/yellow]"
            )
            print("[yellow]Falling back to default GPU allocation[/yellow]")

    app = SmallRunner(
        tuple(gpu_ids), tuple(commands), concurrent_jobs, gpus_per_job, topology,
        enforce_python=enforce_python
    )
    app.run()


@dataclass
class Command:
    """Represents a command to be executed."""

    id: int  # Unique identifier for the command
    args: list[str]  # List of command arguments
    kill_flag: bool = False  # Flag to indicate if the command should be killed

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
    gpus_used_from_gpu_id: dict[
        int, list[list[int]]
    ]  # Mapping of primary GPU IDs to lists of all GPUs used for each job
    finished_jobs: list[
        FinishedJobInfo
    ]  # List of dictionaries containing finished job information
    start_time: float  # Start time for all jobs
    show_on_exit: list[str]  # List of strings to show when smallrunner exits
    log_file_path: Path = Path(".smallrunner_log")  # Path to the log file


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

        # Get information about all GPUs used by this job
        gpus_used = self._state.gpus_used_from_gpu_id[self._gpu_id][self._job_index]

        # Create GPU identifier
        if not gpus_used or len(gpus_used) <= 1:
            # Single GPU job
            gpu_label = f"GPU{self._gpu_id}-{self._job_index}"
        else:
            # Multi-GPU job - compact format showing all GPUs and job index
            gpu_label = f"GPU{','.join(str(g) for g in gpus_used)}-{self._job_index}"

        label_parts = [
            gpu_label,
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


class JobAdjustmentCommands(Provider):
    """A command provider for job management in the current working directory."""

    async def search(self, query: str) -> Hits:
        """Search for job management commands."""
        matcher = self.matcher(query)

        app = self.app
        assert isinstance(app, SmallRunner)

        # Add command to add new jobs
        add_jobs_command = "Add jobs from shell script"
        score = matcher.match(add_jobs_command)
        if score > 0:
            yield Hit(
                score,
                matcher.highlight(add_jobs_command),
                lambda: app._open_file_picker(),
                help="Select a shell script to add jobs to the queue.",
            )

        for command in app._commands_left:
            skip_command = "Skip " + str(command)
            score = matcher.match(skip_command)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(skip_command),
                    lambda id=command.id: app._skip_command(id),
                    help=f"Don't run this command with id={command.id}.",
                )

        skip_command = "Skip all waiting jobs"
        score = matcher.match(skip_command)
        if score > 0:
            yield Hit(
                score,
                matcher.highlight(skip_command),
                lambda: app._skip_command("all_remaining"),
                help="Don't run any more commands. Commands that are currently running will be unaffected.",
            )

        # Add commands for killing running jobs
        for gpu_id in app._cuda_device_ids:
            for job_index in range(app._jobs_per_gpu):
                command = app._running_commands[gpu_id][job_index]
                if command is not None:
                    kill_command = f"Kill job on GPU {gpu_id}-{job_index}: {command}"
                    score = matcher.match(kill_command)
                    if score > 0:
                        yield Hit(
                            score,
                            matcher.highlight(kill_command),
                            lambda id=command.id: app._kill_command(id),
                            help=f"Kill the running job with id={command.id} on GPU {gpu_id}-{job_index}.",
                        )

        kill_all_command = "Kill all running jobs"
        score = matcher.match(kill_all_command)
        if score > 0:
            yield Hit(
                score,
                matcher.highlight(kill_all_command),
                lambda: app._kill_command("all_running"),
                help="Kill all currently running jobs.",
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
        running_jobs = sum(
            sum(1 for cmd in cmds if cmd)
            for cmds in self._state.command_from_gpu_id.values()
        )
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


def write_to_log(state: GlobalState, message: str) -> None:
    """Write a message to the log file.

    Args:
        state: The global state containing the log file path
        message: The message to write to the log file
    """
    try:
        with open(state.log_file_path, "a") as f:
            f.write(f"{message}\n")
    except Exception as e:
        print(f"[yellow]Warning: Could not write to log file: {e}[/yellow]")


class SmallRunner(App):
    COMMANDS = App.COMMANDS | {JobAdjustmentCommands}
    CSS = """
    .bordered-white {
        border: round white;
    }

    ScrollableContainer {
        scrollbar-color: white;
        scrollbar-size: 1 1;
        scrollbar-background: $surface;
    }

    Grid {
        grid-size: 3;
        grid-gutter: 1;
    }

    #gpu-groups-grid {
        grid-size: 1;
        height: 10;
        margin-top: 1;
    }

    .gpu-groups {
        height: 9;
    }
    """

    # Flag to indicate if the app is shutting down
    _is_shutting_down = False

    def __init__(
        self,
        cuda_device_ids: tuple[int, ...],
        commands: tuple[Command, ...],
        concurrent_jobs: int = 1,
        gpus_per_job: int = 1,
        topology: Optional[Dict[int, List[int]]] = None,
        enforce_python: bool = True,
    ) -> None:
        super().__init__()

        # Reset the shutting down flag at initialization
        self.__class__._is_shutting_down = False

        self._gpus_per_job = gpus_per_job
        self._topology = topology
        self._gpu_groups = []  # Will store groups of GPUs for multi-GPU jobs
        self._job_threads = []  # Keep track of running job threads
        self._enforce_python = enforce_python  # For validating commands from shell scripts

        # If using multi-GPU jobs, limit number of primary GPUs based on --gpus_per_job
        if gpus_per_job > 1 and len(cuda_device_ids) > 1:
            # Calculate max number of jobs we can run with available GPUs
            max_jobs = len(cuda_device_ids) // gpus_per_job

            # Organize primary GPUs based on topology if available
            if topology is not None and gpus_per_job > 1:
                # Create groups of primary GPUs that have good connections between them
                self._gpu_groups = self._create_topology_aware_gpu_groups(
                    cuda_device_ids, gpus_per_job
                )

                # Sort groups by connection quality (best connections first)
                self._sort_gpu_groups_by_connection_quality()

                # Use the first GPU from each group as primary GPUs
                primary_gpus = [group[0] for group in self._gpu_groups[:max_jobs]]
                print(
                    f"Using {len(primary_gpus)} topology-optimized primary GPUs for {len(primary_gpus)} multi-GPU jobs"
                )
                cuda_device_ids = tuple(primary_gpus)
            else:
                # Without topology, just use the first n GPUs as primaries
                cuda_device_ids = cuda_device_ids[:max_jobs]
                print(
                    f"Using {len(cuda_device_ids)} primary GPUs for {max_jobs} multi-GPU jobs (each using {gpus_per_job} GPUs)"
                )

        # Limit GPU count if there are fewer commands than GPU slots
        if len(commands) < len(cuda_device_ids) * concurrent_jobs:
            needed_gpus = (len(commands) + concurrent_jobs - 1) // concurrent_jobs
            cuda_device_ids = cuda_device_ids[:needed_gpus]
            print(f"Limited to {len(cuda_device_ids)} GPUs based on command count")

        self._cuda_device_ids = cuda_device_ids
        self._commands = commands
        self._jobs_per_gpu = (
            concurrent_jobs  # Maintain compatibility with existing code
        )
        self._state = GlobalState(
            command_from_gpu_id={id: [""] * concurrent_jobs for id in cuda_device_ids},
            start_time_from_gpu_id={
                id: [0.0] * concurrent_jobs for id in cuda_device_ids
            },
            logdir_from_gpu_id={id: [None] * concurrent_jobs for id in cuda_device_ids},
            gpus_used_from_gpu_id={
                id: [[] for _ in range(concurrent_jobs)] for id in cuda_device_ids
            },
            finished_jobs=[],
            start_time=time.time(),
            show_on_exit=[],
        )
        self._gpu_free_state = {id: [True] * concurrent_jobs for id in cuda_device_ids}
        self._commands_left = list(reversed(commands))
        self._running_commands: dict[int, list[Command | None]] = {
            id: [None] * concurrent_jobs for id in cuda_device_ids
        }
        self._commands_finished = []
        atexit.register(self._handle_exit)

    def _create_topology_aware_gpu_groups(
        self, available_gpus: tuple[int, ...], gpus_per_group: int
    ) -> List[List[int]]:
        """Create groups of GPUs with optimal topology connections.

        Args:
            available_gpus: Tuple of available GPU IDs
            gpus_per_group: Number of GPUs per group

        Returns:
            List of GPU groups, where each group is a list of GPU IDs
        """
        if not self._topology or gpus_per_group <= 1:
            # Without topology information, just divide GPUs into groups
            return [
                list(available_gpus[i : i + gpus_per_group])
                for i in range(0, len(available_gpus), gpus_per_group)
            ]

        # Create groups based on topology
        gpu_groups = []
        remaining_gpus = set(available_gpus)

        # Process each GPU as a potential primary
        for primary_gpu in available_gpus:
            # Skip if this GPU is already assigned
            if primary_gpu not in remaining_gpus:
                continue

            # Start a new group with this GPU
            current_group = [primary_gpu]
            remaining_gpus.remove(primary_gpu)

            # Add the best-connected GPUs to this group
            for connected_gpu in self._topology[primary_gpu]:
                if (
                    connected_gpu in remaining_gpus
                    and len(current_group) < gpus_per_group
                ):
                    current_group.append(connected_gpu)
                    remaining_gpus.remove(connected_gpu)

            # If we couldn't find enough connected GPUs, add any available ones
            while len(current_group) < gpus_per_group and remaining_gpus:
                current_group.append(next(iter(remaining_gpus)))
                remaining_gpus.remove(current_group[-1])

            # Add the group to our list
            gpu_groups.append(current_group)

            # Stop if we have enough groups
            if len(gpu_groups) * gpus_per_group >= len(available_gpus):
                break

        return gpu_groups

    def _sort_gpu_groups_by_connection_quality(self) -> None:
        """Sort GPU groups by connection quality, with best connections first.

        This evaluates the overall connection quality of each group by calculating
        the average connection level between the primary GPU and other GPUs in the group.
        Lower connection levels are better (0=same device, 5=separate systems).
        """
        if not self._topology or not self._gpu_groups:
            return

        # Calculate connection quality score for each group
        group_scores = []
        for group in self._gpu_groups:
            if len(group) <= 1:
                # Single GPU groups have perfect score
                group_scores.append((group, 0))
                continue

            primary_gpu = group[0]
            other_gpus = group[1:]

            # Calculate average connection level (lower is better)
            connection_levels = []
            for other_gpu in other_gpus:
                try:
                    handle_i = pynvml.nvmlDeviceGetHandleByIndex(primary_gpu)
                    handle_j = pynvml.nvmlDeviceGetHandleByIndex(other_gpu)
                    level = pynvml.nvmlDeviceGetTopologyCommonAncestor(
                        handle_i, handle_j
                    )
                    connection_levels.append(level)
                except Exception:
                    # If we can't get topology info, assume worst connection
                    connection_levels.append(5)

            # Calculate average - lower scores are better
            avg_level = (
                sum(connection_levels) / len(connection_levels)
                if connection_levels
                else 5
            )
            group_scores.append((group, avg_level))

        # Sort groups by score (lowest/best first)
        group_scores.sort(key=lambda x: x[1])

        # Replace groups with sorted version
        self._gpu_groups = [group for group, _ in group_scores]

        # Print info about the sorted groups
        print("GPU groups sorted by connection quality (best first):")
        for i, (group, score) in enumerate(group_scores):
            connection_type = (
                "excellent"
                if score < 2
                else "good"
                if score < 3
                else "fair"
                if score < 4
                else "poor"
            )
            print(
                f"  Group {i + 1}: GPUs {group} - {connection_type} connection quality (score: {score:.1f})"
            )

    def _handle_exit(self):
        print("\n[bold]smallrunner[/bold] exiting. Showing on-exit information:")

        # First, write a header to the log file
        header = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] smallrunner session summary:"
        write_to_log(self._state, header)

        for item in self._state.show_on_exit:
            print("\t", item)
            # Don't duplicate items that were already logged during execution

    def on_mount(self) -> None:
        self.set_interval(0.5, self._poll_update)
        self._poll_update()

    def _open_file_picker(self) -> None:
        """Open a file picker dialog to select a shell script."""
        # Textual 3.x doesn't use the same work/worker system, so we use a simpler approach
        file_picker = FileOpen(
            title="Select a shell script",
            filters=[
                ("Shell Scripts", ["*.sh"])
            ]
        )

        # Show the file picker
        self.push_screen(file_picker, callback=self._add_jobs_from_script)

    def _add_jobs_from_script(self, selected_path: Path) -> None:
        """Add jobs from the selected shell script.

        Args:
            selected_path: The path to the selected shell script
        """
        try:
            # Parse commands from the shell script
            new_commands = parse_commands((selected_path,), self._enforce_python)

            if not new_commands:
                # Show a notification if no valid commands were found
                self.notify("No valid commands found in the selected shell script", severity="warning")
                return

            # Assign IDs to the new commands based on the last used ID
            # Get all existing command IDs to ensure uniqueness
            existing_ids = set()
            for cmd in self._commands:
                existing_ids.add(cmd.id)
            for cmd in self._commands_left:
                existing_ids.add(cmd.id)
            for gpu_id in self._cuda_device_ids:
                for job_index in range(self._jobs_per_gpu):
                    cmd = self._running_commands[gpu_id][job_index]
                    if cmd is not None:
                        existing_ids.add(cmd.id)

            next_id = max(existing_ids) + 1 if existing_ids else 1
            for i, cmd in enumerate(new_commands):
                cmd.id = next_id + i

            # Add the new commands to the waiting queue (prepend to _commands_left for LIFO order)
            for cmd in reversed(new_commands):
                self._commands_left.append(cmd)

            # Log the addition
            message = f"Added {len(new_commands)} jobs from {selected_path}"
            self._state.show_on_exit.append(message)
            write_to_log(self._state, message)

            # Show a notification
            self.notify(f"Added {len(new_commands)} jobs from {selected_path.name}", severity="information")

        except Exception as e:
            # Show error notification if something goes wrong
            self.notify(f"Error adding jobs: {e}", severity="error")
            print(f"[red]Error adding jobs: {e}[/red]")

    def exit(self, *args, **kwargs) -> None:
        """Override exit to mark the app as shutting down."""
        # Set the shutting down flag to prevent UI updates from threads
        self._is_shutting_down = True
        # Allow time for threads to notice the flag
        time.sleep(0.2)
        # Call the parent exit method
        super().exit(*args, **kwargs)

    def _kill_command(self, command_id: int | Literal["all_running"]) -> None:
        """Kill a command with the given command_id or all running commands."""
        if command_id == "all_running":
            for gpu_id in self._cuda_device_ids:
                for job_index in range(self._jobs_per_gpu):
                    command = self._running_commands[gpu_id][job_index]
                    if command is not None:
                        command.kill_flag = True
        else:
            for gpu_id in self._cuda_device_ids:
                for job_index in range(self._jobs_per_gpu):
                    command = self._running_commands[gpu_id][job_index]
                    if command is not None and command.id == command_id:
                        command.kill_flag = True
                        return

    def _skip_command(self, command_id: int | Literal["all_remaining"]) -> None:
        """Skip the command with the given command_id or all remaining commands."""
        if command_id == "all_remaining":
            for command in self._commands_left:
                finished_message = f"{command} [yellow]skipped[/yellow]"
                self._commands_finished.append(finished_message)
                self._state.show_on_exit.append(finished_message)
                write_to_log(self._state, finished_message)
            self._commands_left.clear()
        else:
            (command,) = [cmd for cmd in self._commands_left if cmd.id == command_id]
            self._commands_left = [
                cmd for cmd in self._commands_left if cmd.id != command_id
            ]
            finished_message = f"{command} [yellow]skipped[/yellow]"
            self._commands_finished.append(finished_message)
            self._state.show_on_exit.append(finished_message)
            write_to_log(self._state, finished_message)

    def _run_job(self, gpu_id: int, job_index: int, command: Command) -> None:
        logdir = Path(
            f"/tmp/smallrunner_logs/{time.strftime('%Y%m%d_%H%M%S')}_{gpu_id}_{job_index}"
        )
        logdir.mkdir(parents=True, exist_ok=True)

        # Allocate multiple GPUs if needed
        gpu_ids_for_job = [gpu_id]
        if self._gpus_per_job > 1:
            # Check if we have pre-computed GPU groups from topology analysis
            if self._gpu_groups:
                # Find the group that contains this primary GPU
                group_for_this_gpu = None
                for group in self._gpu_groups:
                    if group[0] == gpu_id:
                        group_for_this_gpu = group
                        break

                if group_for_this_gpu:
                    # Use the pre-computed group (skip first one as it's already the primary)
                    additional_gpus = group_for_this_gpu[1:]
                    if additional_gpus:
                        # Verify GPUs are still available
                        available_additional_gpus = []
                        for other_gpu_id in additional_gpus:
                            # Skip if memory is in use (by external processes)
                            if get_gpu_memory_usage(other_gpu_id) > 0.1:
                                continue

                            # Skip if it's a primary GPU and already in use
                            if other_gpu_id in self._cuda_device_ids and not all(
                                self._gpu_free_state[other_gpu_id]
                            ):
                                continue

                            available_additional_gpus.append(other_gpu_id)

                        if available_additional_gpus:
                            gpu_ids_for_job.extend(available_additional_gpus)
                            print(
                                f"Using topology-optimized GPU group for job: {gpu_ids_for_job}"
                            )

            # If we don't have enough GPUs yet from topology groups, find more
            if len(gpu_ids_for_job) < self._gpus_per_job:
                # Find additional available GPUs (from the full pool of GPUs, not just primary ones)
                all_gpu_ids = list(range(pynvml.nvmlDeviceGetCount()))
                available_gpus = []

                # Use topology information if available to sort by connection quality
                if self._topology is not None and gpu_id in self._topology:
                    # Sort all GPUs by connection quality to this GPU (with type assertion for mypy)
                    topology = self._topology  # Make a local non-optional reference
                    all_gpu_ids = sorted(
                        all_gpu_ids,
                        key=lambda x: topology[gpu_id].index(x)
                        if x in topology[gpu_id]
                        else 9999,
                    )

                # Find free GPUs
                for other_gpu_id in all_gpu_ids:
                    # Skip if already in our list
                    if other_gpu_id in gpu_ids_for_job:
                        continue

                    # Skip if it's a primary GPU and already in use
                    if other_gpu_id in self._cuda_device_ids and not all(
                        self._gpu_free_state[other_gpu_id]
                    ):
                        continue

                    # Skip if memory is in use (by external processes)
                    if (
                        get_gpu_memory_usage(other_gpu_id) > 0.1
                    ):  # Same threshold used for initial selection
                        continue

                    available_gpus.append(other_gpu_id)
                    if len(available_gpus) >= self._gpus_per_job - len(gpu_ids_for_job):
                        break

                # Add available GPUs to the job
                gpu_ids_for_job.extend(
                    available_gpus[: self._gpus_per_job - len(gpu_ids_for_job)]
                )

            # Print warning if we couldn't allocate enough GPUs
            if len(gpu_ids_for_job) < self._gpus_per_job:
                print(
                    f"[yellow]Warning: Could only allocate {len(gpu_ids_for_job)} GPUs for job instead of requested {self._gpus_per_job}[/yellow]"
                )

            # Mark the additional GPUs as in use (only needed for those that are primary GPUs)
            for other_gpu_id in gpu_ids_for_job[1:]:
                if other_gpu_id in self._gpu_free_state:
                    for other_job_index in range(self._jobs_per_gpu):
                        self._gpu_free_state[other_gpu_id][other_job_index] = False

            # Log topology information for debugging
            if len(gpu_ids_for_job) > 1:
                try:
                    primary_gpu = gpu_ids_for_job[0]
                    for other_gpu in gpu_ids_for_job[1:]:
                        handle_i = pynvml.nvmlDeviceGetHandleByIndex(primary_gpu)
                        handle_j = pynvml.nvmlDeviceGetHandleByIndex(other_gpu)
                        topo_level = pynvml.nvmlDeviceGetTopologyCommonAncestor(
                            handle_i, handle_j
                        )
                        topo_labels = {
                            0: "INTERNAL",
                            1: "SINGLE",
                            2: "MULTIPLE",
                            3: "HOSTBRIDGE",
                            4: "NODE",
                            5: "SYSTEM",
                        }
                        topo_name = topo_labels.get(topo_level, f"Level_{topo_level}")
                        print(
                            f"[dim]Debug: Connection between GPU{primary_gpu}-GPU{other_gpu}: {topo_name} (level: {topo_level})[/dim]"
                        )
                except Exception:
                    pass  # Ignore errors in topology debugging

        visible_devices = ",".join(str(g) for g in gpu_ids_for_job)

        # Useful if smallrunner exits...
        args = command.args
        if len(gpu_ids_for_job) <= 1:
            gpu_info = f"GPU [bold]{gpu_id}[/bold]-{job_index}"
        else:
            # More compact format for multi-GPU jobs
            gpu_info = f"GPU [bold]{','.join(str(g) for g in gpu_ids_for_job)}[/bold]-{job_index}"

        message = f"Started [cyan]{shlex.join(args)}[/cyan] on {gpu_info}, logging to [blue]{logdir}[/blue]"
        self._state.show_on_exit.append(message)
        write_to_log(self._state, message)

        self._state.logdir_from_gpu_id[gpu_id][job_index] = logdir
        self._state.command_from_gpu_id[gpu_id][job_index] = shlex.join(args)
        self._state.start_time_from_gpu_id[gpu_id][job_index] = time.time()
        self._state.gpus_used_from_gpu_id[gpu_id][job_index] = gpu_ids_for_job.copy()
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
                    CUDA_VISIBLE_DEVICES=visible_devices,
                ),
                bufsize=1,
                universal_newlines=True,
            )

            self._handle_process_output(
                command, process, stdout_f, stderr_f, log_display
            )

            if command.kill_flag:
                elapsed_time = (
                    time.time() - self._state.start_time_from_gpu_id[gpu_id][job_index]
                )
                elapsed_str = format_elapsed_time(elapsed_time)
                logdir = self._state.logdir_from_gpu_id[gpu_id][job_index]
                killed_message = f"{command} [red]killed[/red] after [bold]{elapsed_str}[/bold], logs saved to [blue]{logdir}[/blue]"
                self._commands_finished.append(killed_message)
                self._state.show_on_exit.append(killed_message)
                write_to_log(self._state, killed_message)

        # Free all GPUs used by this job
        self._gpu_free_state[gpu_id][job_index] = True

        # Get the list of all GPUs that were used for this job from our tracking variable
        additional_gpus = (
            self._state.gpus_used_from_gpu_id[gpu_id][job_index][1:]
            if self._state.gpus_used_from_gpu_id[gpu_id][job_index]
            else []
        )

        # Free additional GPUs if we have any
        if additional_gpus:
            for other_gpu_id in additional_gpus:
                if other_gpu_id in self._gpu_free_state:
                    for other_job_index in range(self._jobs_per_gpu):
                        self._gpu_free_state[other_gpu_id][other_job_index] = True

            # Clear the list of GPUs used for this job
            self._state.gpus_used_from_gpu_id[gpu_id][job_index] = []
        # If we don't have tracking info but are using multi-GPU, free all potentially used GPUs
        elif self._gpus_per_job > 1:
            for other_gpu_id in self._cuda_device_ids:
                if other_gpu_id != gpu_id and any(
                    not free for free in self._gpu_free_state[other_gpu_id]
                ):
                    for other_job_index in range(self._jobs_per_gpu):
                        self._gpu_free_state[other_gpu_id][other_job_index] = True

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
        write_to_log(self._state, finished_message)

    def _handle_process_output(
        self,
        command: Command,
        process: subprocess.Popen,
        stdout_f: IO,
        stderr_f: IO,
        log_display: Log,
    ) -> None:
        def process_stream(stream: IO, file: IO) -> None:
            line = stream.readline()
            if line:
                file.write(line)
                file.flush()
                # Only update the UI if the app is not shutting down
                if not self._is_shutting_down:
                    try:
                        log_display.write(line)
                    except Exception:
                        # Ignore errors when writing to the UI (app might be shutting down)
                        pass

        assert process.stdout is not None
        assert process.stderr is not None
        while True:
            # Exit the loop if the app is shutting down
            if self._is_shutting_down:
                break

            # Use a try-except block to handle possible errors during select
            try:
                rlist, _, _ = select.select(
                    [process.stdout, process.stderr], [], [], 0.1
                )

                for ready_stream in rlist:
                    if ready_stream == process.stdout:
                        process_stream(process.stdout, stdout_f)
                    elif ready_stream == process.stderr:
                        process_stream(process.stderr, stderr_f)
            except Exception:
                # An exception here likely means the app is shutting down
                break

            if command.kill_flag:
                process.terminate()
                break

            if process.poll() is not None:
                break

        # Read any remaining output if the app is not shutting down
        if not self._is_shutting_down:
            try:
                for stream, file in [
                    (process.stdout, stdout_f),
                    (process.stderr, stderr_f),
                ]:
                    for line in stream:
                        process_stream(stream, file)
            except Exception:
                # Ignore errors when reading remaining output
                pass

        if command.kill_flag and process.poll() is None:
            try:
                process.kill()  # Force kill if terminate didn't work
            except Exception:
                # Ignore errors when killing the process
                pass

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

                    # Create and start a new thread for the job
                    job_thread = threading.Thread(
                        target=self._run_job,
                        args=(gpu_id, job_index, command),
                        daemon=True,  # Make threads daemon so they exit when main thread exits
                    )
                    self._job_threads.append(job_thread)
                    job_thread.start()

        self._update_list("#list-waiting", self._commands_left)
        # Create running job list with all GPUs used by each job
        running_jobs = []
        for gpu_id in self._cuda_device_ids:
            for job_index, command in enumerate(self._running_commands[gpu_id]):
                if command is not None:
                    # Get all GPUs used by this job
                    gpus_used = self._state.gpus_used_from_gpu_id[gpu_id][job_index]

                    # Create GPU identifier based on whether multiple GPUs are used
                    if not gpus_used or len(gpus_used) <= 1:
                        # Single GPU job
                        gpu_label = f"GPU {gpu_id}-{job_index}"
                    else:
                        # Multi-GPU job - show all GPUs in the group
                        gpu_label = (
                            f"GPU {','.join(str(g) for g in gpus_used)}-{job_index}"
                        )

                    running_jobs.append(
                        f"{gpu_label}: {command}, logs to [blue]{self._state.logdir_from_gpu_id[gpu_id][job_index]}[/blue]"
                    )

        self._update_list("#list-running", running_jobs)
        self._update_list("#list-finished", self._commands_finished)

        # Update GPU groups display if using topology
        if (
            self._topology is not None
            and self._gpus_per_job > 1
            and hasattr(self, "_gpu_groups")
            and self._gpu_groups
        ):
            try:
                # Get widget if it exists
                gpu_groups_widget = self.query_one("#list-gpu-groups", Static)

                # Format GPU groups information
                groups_info = []
                for i, group in enumerate(self._gpu_groups):
                    # Show connection info for each group
                    if len(group) > 1:
                        connections = []
                        for j in range(1, len(group)):
                            # If available, get connection type between primary GPU and others
                            primary_gpu = group[0]
                            other_gpu = group[j]

                            # Get connection type between the GPUs
                            try:
                                handle_i = pynvml.nvmlDeviceGetHandleByIndex(
                                    primary_gpu
                                )
                                handle_j = pynvml.nvmlDeviceGetHandleByIndex(other_gpu)

                                # Get connection level (0=same device, 5=separate systems)
                                topo_level = pynvml.nvmlDeviceGetTopologyCommonAncestor(
                                    handle_i, handle_j
                                )

                                # Map topology level to connection type label
                                topo_labels = {
                                    0: "INTERNAL",  # Same GPU (shouldn't happen)
                                    1: "SINGLE",  # Same board
                                    2: "MULTIPLE",  # Multiple boards, same system
                                    3: "HOSTBRIDGE",  # Connected via host bridge
                                    4: "NODE",  # Connected via NUMA node
                                    5: "SYSTEM",  # Connected via system
                                }

                                # Add info to connections list
                                if (
                                    self._topology is not None
                                    and primary_gpu in self._topology
                                    and other_gpu in self._topology[primary_gpu]
                                ):
                                    rank = self._topology[primary_gpu].index(other_gpu)
                                    topo_name = topo_labels.get(
                                        topo_level, f"Level_{topo_level}"
                                    )
                                    connections.append(
                                        f"GPU{primary_gpu}→{other_gpu} ([magenta]{topo_name}[/magenta])"
                                    )
                            except Exception:
                                # If we can't get topology info, fall back to simple rank
                                if (
                                    self._topology is not None
                                    and primary_gpu in self._topology
                                    and other_gpu in self._topology[primary_gpu]
                                ):
                                    rank = self._topology[primary_gpu].index(other_gpu)
                                    connections.append(
                                        f"GPU{primary_gpu}→{other_gpu} (rank: {rank})"
                                    )

                        groups_info.append(
                            f"[bold]Group {i + 1}:[/bold] {','.join(str(g) for g in group)} "
                            + f"([green]{group[0]}[/green]→{', '.join(str(g) for g in group[1:])}): "
                            + f"{' | '.join(connections)}"
                        )
                    else:
                        groups_info.append(f"[bold]G{i + 1}:[/bold] GPU {group[0]}")

                # Update the widget
                gpu_groups_widget.update("\n".join(groups_info))
            except Exception:
                # If the widget doesn't exist, ignore the error
                pass

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
            # Main job lists
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

            # GPU groups info in a separate smaller grid
            if self._topology is not None and self._gpus_per_job > 1:
                with Grid(id="gpu-groups-grid"):
                    with ScrollableContainer(
                        classes="bordered-white gpu-groups"
                    ) as container:
                        container.border_title = (
                            "[bold reverse] GPU Groups [/bold reverse]"
                        )
                        yield Static(id="list-gpu-groups")


def get_gpu_topology() -> Dict[int, List[int]]:
    """Get GPU topology information showing which GPUs are physically connected.

    This function runs nvidia-smi topo -m and parses the output to find which GPUs
    have the closest connections (NVLink, PCIe, etc).

    Returns:
        Dict mapping GPU ID to list of connected GPU IDs (ordered by connection quality)
    """
    topology = {}
    gpu_count = pynvml.nvmlDeviceGetCount()

    # Connection types are ranked by NVML with values from 0 (closest) to 5 (farthest)
    # 0 = NVML_TOPOLOGY_INTERNAL (same device)
    # 1 = NVML_TOPOLOGY_SINGLE (same board)
    # 2 = NVML_TOPOLOGY_MULTIPLE (same system, different boards)
    # 3 = NVML_TOPOLOGY_HOSTBRIDGE (different systems, connected to same host bridge)
    # 4 = NVML_TOPOLOGY_NODE (different systems, connected to same NUMA node)
    # 5 = NVML_TOPOLOGY_SYSTEM (different systems)

    # Initialize empty connections for each GPU
    for i in range(gpu_count):
        topology[i] = []

    # For each GPU, get connection info to all other GPUs
    for i in range(gpu_count):
        handle_i = pynvml.nvmlDeviceGetHandleByIndex(i)
        connections = []

        # Check connection to each other GPU
        for j in range(gpu_count):
            if i == j:
                continue  # Skip self

            handle_j = pynvml.nvmlDeviceGetHandleByIndex(j)
            try:
                # Get PCIe topology information
                topo_level = pynvml.nvmlDeviceGetTopologyCommonAncestor(
                    handle_i, handle_j
                )

                # Convert the level to a rank based on our preference
                # NVML_TOPOLOGY_INTERNAL = 0 (same GPU), up to NVML_TOPOLOGY_SYSTEM = 5 (least connected)
                # Lower numbers are better, so we invert for consistent sorting
                rank = 5 - topo_level
                connections.append((j, rank))
            except pynvml.NVMLError:
                # If we can't get topology info, assume worst connection
                connections.append((j, 0))

        # Sort by connection quality (highest rank first)
        connections.sort(key=lambda x: x[1], reverse=True)

        # Store just the GPU IDs in order of connection quality
        topology[i] = [gpu_id for gpu_id, _ in connections]

    return topology


def get_available_gpus(mem_ratio_threshold: float) -> tuple[int, ...]:
    """Determine the tuple of available GPU IDs based on memory usage and visibility.

    This function filters GPUs based on their memory usage and the CUDA_VISIBLE_DEVICES
    environment variable.

    Args:
        mem_ratio_threshold: The maximum memory usage ratio for a GPU to be considered available.

    Returns:
        A tuple of available GPU IDs.
    """
    gpu_count = pynvml.nvmlDeviceGetCount()
    print(f"Total GPUs detected: {gpu_count}")

    gpu_ids = [
        i for i in range(gpu_count) if get_gpu_memory_usage(i) <= mem_ratio_threshold
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
                assert line.startswith("python "), (
                    f"Line {i + 1} must start with 'python' when enforce_python is True"
                )

            # Remove escaped newlines.
            line = line.replace("\\\n", "").strip()
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
