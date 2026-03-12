"""Callback to capture stdout/stderr and log as a logger artifact."""

import sys
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import lightning as L


class TeeStream:
    """Stream that writes to both original stream and a buffer."""

    def __init__(self, original_stream, buffer: StringIO):
        self.original_stream = original_stream
        self.buffer = buffer

    def write(self, data: str) -> int:
        self.original_stream.write(data)
        self.buffer.write(data)
        return len(data)

    def flush(self):
        self.original_stream.flush()

    def isatty(self) -> bool:
        return hasattr(self.original_stream, "isatty") and self.original_stream.isatty()

    def fileno(self):
        return self.original_stream.fileno()


class LogOutputCallback(L.Callback):
    """Capture stdout/stderr during training and log as a logger artifact.

    This callback captures all terminal output while still displaying it,
    then saves it as an artifact (MLFlow or W&B) for reproducibility.

    Args:
        artifact_name: Name of the output log file (default: "output.log")
        capture_stderr: Whether to also capture stderr (default: True)

    Example:
        Add to trainer callbacks in config:

        trainer:
          callbacks:
            - class_path: core.callbacks.LogOutputCallback.LogOutputCallback
              init_args:
                artifact_name: "training_output.log"
    """

    def __init__(
        self,
        artifact_name: str = "output.log",
        capture_stderr: bool = True,
    ):
        super().__init__()
        self.artifact_name = artifact_name
        self.capture_stderr = capture_stderr

        self._stdout_buffer: Optional[StringIO] = None
        self._stderr_buffer: Optional[StringIO] = None
        self._original_stdout = None
        self._original_stderr = None
        self._start_time: Optional[datetime] = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start capturing output at the beginning of training."""
        if not trainer.is_global_zero:
            return

        self._start_capture()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Stop capturing and log output at the end of training."""
        if not trainer.is_global_zero:
            return

        self._stop_and_log(trainer, stage="fit")

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start capturing output at the beginning of testing."""
        if not trainer.is_global_zero:
            return

        # Only start if not already capturing (e.g., from fit)
        if self._stdout_buffer is None:
            self._start_capture()

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Stop capturing and log output at the end of testing."""
        if not trainer.is_global_zero:
            return

        self._stop_and_log(trainer, stage="test")

    def on_exception(
        self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException
    ) -> None:
        """Ensure output is logged even if an exception occurs."""
        if not trainer.is_global_zero:
            return

        if self._stdout_buffer is not None:
            self._stop_and_log(trainer, stage="exception")

    def _start_capture(self) -> None:
        """Start capturing stdout and optionally stderr."""
        self._start_time = datetime.now()

        # Capture stdout
        self._stdout_buffer = StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = TeeStream(self._original_stdout, self._stdout_buffer)

        # Optionally capture stderr
        if self.capture_stderr:
            self._stderr_buffer = StringIO()
            self._original_stderr = sys.stderr
            sys.stderr = TeeStream(self._original_stderr, self._stderr_buffer)

    def _stop_and_log(self, trainer: L.Trainer, stage: str) -> None:
        """Stop capturing and log the output as an artifact."""
        # Restore original streams
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None

        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
            self._original_stderr = None

        # Build combined output
        output_lines = []

        # Add header
        end_time = datetime.now()
        output_lines.append(f"# Experiment Output Log")
        output_lines.append(f"# Stage: {stage}")
        if self._start_time:
            output_lines.append(f"# Started: {self._start_time.isoformat()}")
            output_lines.append(f"# Ended: {end_time.isoformat()}")
            duration = end_time - self._start_time
            output_lines.append(f"# Duration: {duration}")
        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("STDOUT")
        output_lines.append("=" * 80)
        output_lines.append("")

        # Add stdout content
        if self._stdout_buffer:
            output_lines.append(self._stdout_buffer.getvalue())

        # Add stderr content if captured
        if self._stderr_buffer and self.capture_stderr:
            stderr_content = self._stderr_buffer.getvalue()
            if stderr_content.strip():
                output_lines.append("")
                output_lines.append("=" * 80)
                output_lines.append("STDERR")
                output_lines.append("=" * 80)
                output_lines.append("")
                output_lines.append(stderr_content)

        # Log as artifact if logger supports it
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            logger_name = type(trainer.logger).__name__

            with tempfile.TemporaryDirectory() as tmp_dir:
                log_path = Path(tmp_dir) / self.artifact_name
                log_path.write_text("\n".join(output_lines))

                if "MLFlow" in logger_name and hasattr(trainer.logger, "run_id"):
                    trainer.logger.experiment.log_artifact(
                        local_path=str(log_path),
                        run_id=trainer.logger.run_id,
                    )
                    print(f"\nOutput log saved to artifact: {self.artifact_name}")
                elif "Wandb" in logger_name:
                    import wandb

                    artifact = wandb.Artifact("output_log", type="log")
                    artifact.add_file(str(log_path))
                    trainer.logger.experiment.log_artifact(artifact)
                    print(f"\nOutput log saved to artifact: {self.artifact_name}")

        # Clear buffers
        self._stdout_buffer = None
        self._stderr_buffer = None
        self._start_time = None
