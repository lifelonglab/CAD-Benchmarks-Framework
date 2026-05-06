from __future__ import annotations

from collections.abc import Callable
import numpy as np
from pyclad.models.model import Model
from cadbench.concept import TemporalConcept, sliding_windows


class TemporalWindowAdapter(Model):
    """Wraps any pyclad sequence-aware ``Model`` to consume 2D row streams.

    The wrapped model is assumed to be seq-to-seq: given ``(n, seq_len, d)``
    input windows it returns per-timestep outputs of the same shape. Every row
    therefore has a native reconstruction error from every window that covers
    it - no need to broadcast a single window-level score onto its points.

    pyclad's ``Model.predict`` contract is ``(flags, scores)``:
    * ``scores`` (a.k.a. the inner model's ``err``) - continuous reconstruction
      error per timestep; higher = more anomalous. This is what our ROC-AUC /
      PR-AUC callbacks actually use.
    * ``flags`` (a.k.a. the inner model's ``binary``) - ``scores > threshold``
      applied post-hoc. The inner model thresholds once per window; we
      re-derive the flags from the *aggregated* per-row scores so the two
      outputs stay consistent (flagged iff aggregated score exceeds threshold).

    Args:
        build_inner: ``(input_features, seq_len) -> Model``. Called inside
            ``fit`` once ACF has chosen ``sequence_length`` - needed because
            pyclad's ``LSTMDecoder``/``GRUDecoder`` etc. require ``seq_len``
            at construction time.
        stride: Step between consecutive window start positions. Defaults to
            ``sequence_length`` (non-overlapping - each row in exactly one
            window). ``stride=1`` is fully overlapping (each row in up to
            ``seq_len`` windows). Predictions from overlapping windows are
            aggregated back to per-row so the adapter's output stays 1:1
            with the dataset's per-row ground-truth labels. Continuous
            reconstruction errors are aggregated via **median** across
            overlapping windows, following TadGAN (Geiger et al., IEEE
            BigData 2020), which reports the median outperforms the
            mean for this aggregation.
    """

    def __init__(
        self,
        build_inner: Callable[[int, int], Model],
        stride: int | None = None,
    ) -> None:
        self._build_model = build_inner
        self._stride = stride
        self.inner: Model | None = None
        self.seq_len: int | None = None

    def fit(self, data: np.ndarray) -> None:
        tc = TemporalConcept(features=data, train_ratio=1.0)
        self.seq_len = int(tc.sequence_length)

        if self._stride is not None and self._stride > self.seq_len:
            self._stride = self.seq_len

        self.inner = self._build_model(int(data.shape[1]), self.seq_len)
        self.inner.fit(tc.to_sequences(stride=self._stride))

    def name(self) -> str:
        return "LSTM-TemporalAutoencoder"

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.inner is None or self.seq_len is None:
            raise RuntimeError("Adapter has not been fitted yet.")
        T, d = data.shape
        if T < self.seq_len:
            raise ValueError(f"Test concept too short: T={T} < seq_len={self.seq_len}.")

        stride = self._stride if self._stride is not None else self.seq_len

        # Single batched forward pass through the inner seq-to-seq model.
        # ``anomaly_scores`` has shape ``(n, seq_len, 1)`` - one
        # reconstruction error per timestep per window.
        sequences = sliding_windows(data, self.seq_len, stride)
        n = sequences.shape[0]
        _, anomaly_scores = self.inner.predict(sequences)
        scores_per_window = np.asarray(anomaly_scores).reshape(n, self.seq_len)

        if stride == self.seq_len:
            # Non-overlapping: each row belongs to exactly one window, so the
            # per-timestep errors already line up 1:1 with rows.
            per_row_scores = scores_per_window.reshape(n * self.seq_len).astype(np.float64)
        else:
            # Overlapping (1 <= stride < seq_len): each row is covered by
            # multiple windows, so we get multiple per-row errors for the
            # same row. Aggregate them via **median** (TadGAN paper).

            # Collect all per-window errors for each row into a (T, max_k)
            # buffer (NaN-padded when a row is covered by fewer than max_k
            # windows), then take nanmedian along the window axis.
            max_k = (self.seq_len + stride - 1) // stride  # ceil(seq_len / stride)
            scores_buf = np.full((T, max_k), np.nan, dtype=np.float64)
            fill_idx = np.zeros(T, dtype=np.int64)
            for i in range(n):
                start = i * stride
                end = start + self.seq_len
                rows = np.arange(start, end)
                scores_buf[rows, fill_idx[rows]] = scores_per_window[i]
                fill_idx[rows] += 1
            per_row_scores = np.nanmedian(scores_buf, axis=1)
            per_row_scores = np.nan_to_num(per_row_scores, nan=0.0)

        # Tail rows uncovered by any window (when T - seq_len is not a
        # multiple of stride): repeat the last covered value so the output
        # length is T - keeps 1:1 alignment with per-row labels.
        covered_end = (n - 1) * stride + self.seq_len
        if T > covered_end:
            per_row_scores = np.concatenate([
                per_row_scores[:covered_end],
                np.full(T - covered_end, per_row_scores[covered_end - 1], dtype=np.float64),
            ])

        # Derive per-row flags from the aggregated scores using the inner
        # model's threshold. Single threshold application.
        threshold = getattr(self.inner, "threshold", 0.5)
        per_row_flags = (per_row_scores > threshold).astype(np.int64)

        return per_row_flags, per_row_scores
