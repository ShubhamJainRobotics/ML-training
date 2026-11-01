"""
Jaanch Real-Time Training Progress Monitor & Checkpoint Manager
Provides:
1. Live terminal progress bar with tqdm (current/total epoch, current/total batch, loss, elapsed, ETA).
2. End-of-epoch metrics display (train loss, val loss, val accuracy, precision, recall, F1-score).
3. Automatic CSV logging to reports/training_logs.csv.
4. Automatic best model checkpoint saving based on validation F1-score.
5. Resumption support from the last checkpoint if training is interrupted.
6. Hugging Face TrainerCallback integration for Transformer models.
7. Real-time JSON state export for live Streamlit dashboard updates.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TrainingMonitor:
    """
    Real-time training monitor with tqdm terminal progress, automatic CSV logging,
    checkpointing, and resumption capabilities.
    """

    def __init__(
        self,
        log_csv_path: Union[str, Path] = "ml/reports/training_logs.csv",
        checkpoint_dir: Union[str, Path] = "ml/models/checkpoints",
        total_epochs: int = 10,
        total_batches_per_epoch: int = 100,
        metric_for_best: str = "f1",
        higher_is_better: bool = True,
        resume: bool = False,
    ):
        self.log_csv_path = Path(log_csv_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.total_epochs = total_epochs
        self.total_batches_per_epoch = total_batches_per_epoch
        self.metric_for_best = metric_for_best
        self.higher_is_better = higher_is_better

        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / "checkpoint_state.json"

        # Tracking variables
        self.current_epoch = 0
        self.current_batch = 0
        self.global_step = 0
        self.total_steps = self.total_epochs * self.total_batches_per_epoch
        self.start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.running_batch_losses = []

        self.best_metric_value = -float("inf") if self.higher_is_better else float("inf")
        self.best_epoch = 0

        self.pbar: Optional[tqdm] = None
        self.is_resumed = False

        # Initialize or resume CSV and state
        self._init_or_resume_storage(resume=resume)

    def _init_or_resume_storage(self, resume: bool):
        """Initializes new log or resumes from existing state."""
        columns = [
            "timestamp",
            "epoch",
            "total_epochs",
            "batch",
            "total_batches",
            "global_step",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "precision",
            "recall",
            "f1",
            "elapsed_seconds",
            "eta_seconds",
            "status",
        ]

        if resume and self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.current_epoch = state.get("current_epoch", 0)
                self.best_metric_value = state.get("best_metric_value", self.best_metric_value)
                self.best_epoch = state.get("best_epoch", 0)
                self.global_step = state.get("global_step", 0)
                self.is_resumed = True
                print(f"\n[TrainingMonitor] Resuming from Checkpoint: Last completed Epoch {self.current_epoch}/{self.total_epochs} (Best {self.metric_for_best.upper()}: {self.best_metric_value:.4f} at Epoch {self.best_epoch})")
            except Exception as e:
                print(f"[TrainingMonitor] Warning: Could not read state file ({e}). Starting fresh.")
                self.is_resumed = False

        if not self.is_resumed or not self.log_csv_path.exists():
            # Create fresh CSV
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.log_csv_path, index=False)
            self._save_state(status="INITIALIZED")

    def start_training(self):
        """Marks training start timestamp."""
        if self.start_time is None:
            self.start_time = time.time()
        self._save_state(status="RUNNING")

    def start_epoch(self, epoch: int):
        """Initializes a new epoch and displays terminal progress bar."""
        self.current_epoch = epoch
        self.current_batch = 0
        self.running_batch_losses = []
        self.epoch_start_time = time.time()
        if self.start_time is None:
            self.start_time = time.time()

        desc = f"Epoch [{self.current_epoch}/{self.total_epochs}]"
        self.pbar = tqdm(
            total=self.total_batches_per_epoch,
            desc=desc,
            unit="batch",
            leave=True,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        self._save_state(status="TRAINING")

    def update_batch(self, batch_idx: int, batch_loss: float):
        """Updates live batch progress, loss, elapsed time, and ETA."""
        self.current_batch = batch_idx + 1
        self.global_step += 1
        self.running_batch_losses.append(batch_loss)

        avg_loss = float(np.mean(self.running_batch_losses[-20:]))  # Running mean of recent 20 batches
        elapsed = time.time() - self.start_time if self.start_time else 0.0

        # Estimate remaining time based on steps
        remaining_steps = max(0, self.total_steps - self.global_step)
        steps_per_sec = self.global_step / max(1e-5, elapsed)
        eta = remaining_steps / max(1e-5, steps_per_sec)

        if self.pbar:
            self.pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "elapsed": str(timedelta(seconds=int(elapsed))),
                "eta": str(timedelta(seconds=int(eta))),
            })
            self.pbar.update(1)

        # Update real-time state for Streamlit dashboard (every 5 batches or at end)
        if self.current_batch % 5 == 0 or self.current_batch == self.total_batches_per_epoch:
            self._save_state(
                status="TRAINING",
                train_loss=avg_loss,
                elapsed_seconds=round(elapsed, 1),
                eta_seconds=round(eta, 1),
            )

    def end_epoch(
        self,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        model_artifact: Any,
        is_torch_model: bool = False,
    ) -> bool:
        """
        Finalizes epoch: closes progress bar, prints formatted report, logs to CSV,
        and saves checkpoints (both latest and best).
        Returns True if new best model was achieved.
        """
        if self.pbar:
            self.pbar.close()

        elapsed = time.time() - self.start_time if self.start_time else 0.0
        remaining_epochs = max(0, self.total_epochs - self.current_epoch)
        sec_per_epoch = elapsed / max(1, self.current_epoch)
        eta = remaining_epochs * sec_per_epoch

        # Check if this epoch is the best performing
        eval_metric_val = {
            "f1": f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "precision": precision,
            "recall": recall,
        }.get(self.metric_for_best, f1)

        is_best = False
        if self.higher_is_better:
            if eval_metric_val > self.best_metric_value:
                is_best = True
        else:
            if eval_metric_val < self.best_metric_value:
                is_best = True

        old_best = self.best_metric_value
        if is_best:
            self.best_metric_value = eval_metric_val
            self.best_epoch = self.current_epoch

        # Terminal Print Box
        print("\n" + "=" * 70)
        print(f"  EPOCH {self.current_epoch}/{self.total_epochs} SUMMARY  |  Batches: {self.total_batches_per_epoch}/{self.total_batches_per_epoch}")
        print("=" * 70)
        print(f"  Training Loss:    {train_loss:.4f}     |  Elapsed Time: {timedelta(seconds=int(elapsed))}")
        print(f"  Validation Loss:  {val_loss:.4f}     |  Remaining ETA: {timedelta(seconds=int(eta))}")
        print(f"  Validation Acc:   {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"  Precision:        {precision:.4f}     |  Recall:   {recall:.4f}")
        print(f"  F1-Score:         {f1:.4f}     |  Target Metric: {self.metric_for_best.upper()}={eval_metric_val:.4f}")
        if is_best:
            print(f"  >>> CHECKPOINT: New Best Model Saved! ({self.metric_for_best.upper()}: {old_best:.4f} -> {eval_metric_val:.4f}) <<<")
        else:
            print(f"  Checkpoint: Saved latest (Best remains {self.metric_for_best.upper()}: {self.best_metric_value:.4f} at Epoch {self.best_epoch})")
        print("=" * 70 + "\n")

        # Save to CSV log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "batch": self.total_batches_per_epoch,
            "total_batches": self.total_batches_per_epoch,
            "global_step": self.global_step,
            "train_loss": round(float(train_loss), 5),
            "val_loss": round(float(val_loss), 5),
            "val_accuracy": round(float(val_accuracy), 5),
            "precision": round(float(precision), 5),
            "recall": round(float(recall), 5),
            "f1": round(float(f1), 5),
            "elapsed_seconds": round(float(elapsed), 1),
            "eta_seconds": round(float(eta), 1),
            "status": "COMPLETED" if self.current_epoch == self.total_epochs else "IN_PROGRESS",
        }
        df_entry = pd.DataFrame([log_entry])
        df_entry.to_csv(self.log_csv_path, mode="a", header=False, index=False)

        # Save Checkpoint Artifacts
        self._save_checkpoint_artifacts(
            model_artifact=model_artifact,
            is_best=is_best,
            is_torch_model=is_torch_model,
        )

        # Save Live State JSON
        self._save_state(
            status="COMPLETED" if self.current_epoch == self.total_epochs else "IN_PROGRESS",
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(eta, 1),
        )

        return is_best

    def _save_checkpoint_artifacts(self, model_artifact: Any, is_best: bool, is_torch_model: bool = False):
        """Saves both checkpoint_latest and checkpoint_best."""
        latest_path = self.checkpoint_dir / ("checkpoint_latest.pt" if is_torch_model else "checkpoint_latest.joblib")
        best_path = self.checkpoint_dir / ("checkpoint_best.pt" if is_torch_model else "checkpoint_best.joblib")

        try:
            if is_torch_model and HAS_TORCH:
                torch.save(model_artifact.state_dict() if hasattr(model_artifact, "state_dict") else model_artifact, latest_path)
                if is_best:
                    torch.save(model_artifact.state_dict() if hasattr(model_artifact, "state_dict") else model_artifact, best_path)
            else:
                joblib.dump(model_artifact, latest_path, compress=3)
                if is_best:
                    joblib.dump(model_artifact, best_path, compress=3)
        except Exception as e:
            print(f"[TrainingMonitor] Error saving model checkpoint: {e}")

    def _save_state(self, status: str = "IN_PROGRESS", **kwargs):
        """Exports JSON state for real-time Streamlit dashboard consumption."""
        state = {
            "status": status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches_per_epoch,
            "global_step": self.global_step,
            "total_steps": self.total_steps,
            "best_metric_name": self.metric_for_best,
            "best_metric_value": self.best_metric_value,
            "best_epoch": self.best_epoch,
            "last_updated": datetime.now().isoformat(),
        }
        state.update(kwargs)
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            pass

    def load_latest_checkpoint(self, is_torch_model: bool = False):
        """Loads latest saved model checkpoint for training resumption."""
        latest_path = self.checkpoint_dir / ("checkpoint_latest.pt" if is_torch_model else "checkpoint_latest.joblib")
        if not latest_path.exists():
            return None
        if is_torch_model and HAS_TORCH:
            return torch.load(latest_path)
        else:
            return joblib.load(latest_path)


# ==============================================================================
# Hugging Face TrainerCallback Implementation
# ==============================================================================
if HAS_TRANSFORMERS:
    class JaanchTrainerCallback(TrainerCallback):
        """
        Custom Hugging Face Trainer callback that connects Transformer training
        to TrainingMonitor for live terminal progress, CSV logging, and dashboard updates.
        """

        def __init__(self, monitor: TrainingMonitor):
            self.monitor = monitor
            self.current_train_loss = 0.0

        def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            self.monitor.start_training()
            print(f"\n[JaanchTrainerCallback] Transformer Training Started | Epochs: {args.num_train_epochs} | Logging to: {self.monitor.log_csv_path}")

        def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            epoch = int(state.epoch) + 1 if state.epoch is not None else 1
            self.monitor.start_epoch(epoch)

        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.log_history and "loss" in state.log_history[-1]:
                self.current_train_loss = float(state.log_history[-1]["loss"])
            batch_idx = (state.global_step - 1) % self.monitor.total_batches_per_epoch
            self.monitor.update_batch(batch_idx, self.current_train_loss)

        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, Any], **kwargs):
            val_loss = float(metrics.get("eval_loss", 0.0))
            val_acc = float(metrics.get("eval_accuracy", 0.0))
            prec = float(metrics.get("eval_precision", 0.0))
            rec = float(metrics.get("eval_recall", 0.0))
            f1 = float(metrics.get("eval_f1", 0.0))

            model = kwargs.get("model", None)
            self.monitor.end_epoch(
                train_loss=self.current_train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
                precision=prec,
                recall=rec,
                f1=f1,
                model_artifact=model,
                is_torch_model=True,
            )
else:
    class JaanchTrainerCallback:
        """Placeholder when transformers is not installed."""
        def __init__(self, *args, **kwargs):
            pass

