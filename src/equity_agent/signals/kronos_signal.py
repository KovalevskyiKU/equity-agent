"""Kronos foundation-model signal: probabilistic direction from sample dispersion.

Kronos forecasts future OHLCV. Rather than trust its mean path as an oracle (the
weak ~59% directional read measured before), we draw ``sample_count`` sampled
future paths and summarise the *distribution* of the H-step-ahead return:

    k_p_up    - fraction of samples ending above the last close (a probability)
    k_exp_ret - mean sampled H-step return
    k_ret_std - dispersion of sampled returns (the model's own uncertainty)

These feed the same IC harness as the technical features, so Kronos is judged as
one signal among many — not the verdict.

``_sample_paths`` mirrors the vendored ``auto_regressive_inference``
(third_party/Kronos @ 67b630e) but skips the final sample-averaging, keeping the
efficient single-rollout batched sampling while retaining the full distribution.
If the vendored commit is bumped, re-check this function against it.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT

KRONOS_REPO = PROJECT_ROOT / "third_party" / "Kronos"
_CLOSE_IDX = 3  # column order: open, high, low, close, volume, amount


def _ensure_kronos_on_path() -> None:
    if not (KRONOS_REPO / "model").exists():
        raise RuntimeError(
            f"Kronos not found at {KRONOS_REPO}. Clone it:\n"
            f"  git clone https://github.com/shiyu-coder/Kronos third_party/Kronos"
        )
    p = str(KRONOS_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


def directional_signal(terminal_close_paths: np.ndarray, last_close: float) -> dict[str, float]:
    """Summarise H-step-ahead sampled closes into a probabilistic signal."""
    rets = terminal_close_paths / last_close - 1.0
    return {
        "k_p_up": float(np.mean(rets > 0.0)),
        "k_exp_ret": float(np.mean(rets)),
        "k_ret_std": float(np.std(rets)),
    }


class KronosForecaster:
    """Loads Kronos once (slow) and produces probabilistic signals (fast-ish)."""

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        max_context: int = 512,
        device: str | None = None,
    ) -> None:
        _ensure_kronos_on_path()
        from model import Kronos, KronosPredictor, KronosTokenizer

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)
        self._predictor = KronosPredictor(
            model, tokenizer, device=device, max_context=max_context
        )

    def _sample_paths(
        self,
        window: pd.DataFrame,
        x_ts: pd.Series,
        y_ts: pd.Series,
        pred_len: int,
        temperature: float,
        top_p: float,
        sample_count: int,
    ) -> np.ndarray:
        """Return per-sample denormalised OHLCV paths, shape (sample_count, pred_len, 6)."""
        import torch
        from model.kronos import calc_time_stamps, sample_from_logits

        pred: Any = self._predictor
        price_cols, vol, amt = pred.price_cols, pred.vol_col, pred.amt_vol
        df = window.copy()
        if vol not in df:
            df[vol] = 0.0
            df[amt] = 0.0
        if amt not in df:
            df[amt] = df[vol] * df[price_cols].mean(axis=1)

        feat_cols = [*price_cols, vol, amt]
        x = df[feat_cols].to_numpy(dtype=np.float32)
        x_mean, x_std = x.mean(axis=0), x.std(axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -pred.clip, pred.clip)[np.newaxis, :]
        x_stamp = calc_time_stamps(x_ts).to_numpy(dtype=np.float32)[np.newaxis, :]
        y_stamp = calc_time_stamps(y_ts).to_numpy(dtype=np.float32)[np.newaxis, :]

        device, tokenizer, model = pred.device, pred.tokenizer, pred.model
        max_context, clip = pred.max_context, pred.clip
        # `half` selects the BSQ two-part (s1/s2) token representation the inference
        # loop below relies on — it is NOT fp16 precision. Must stay True.
        half = True

        with torch.no_grad():
            xt = torch.from_numpy(x_norm).to(device)
            xs = torch.from_numpy(x_stamp).to(device)
            ys = torch.from_numpy(y_stamp).to(device)
            xt = torch.clip(xt, -clip, clip)
            xt = xt.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, xt.size(1), xt.size(2))
            xs = xs.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, xs.size(1), xs.size(2))
            ys = ys.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, ys.size(1), ys.size(2))

            x_token = tokenizer.encode(xt, half=half)
            initial_len = xt.size(1)
            batch = x_token[0].size(0)
            total_len = initial_len + pred_len
            full_stamp = torch.cat([xs, ys], dim=1)

            gen_pre = x_token[0].new_empty(batch, pred_len)
            gen_post = x_token[1].new_empty(batch, pred_len)
            pre_buf = x_token[0].new_zeros(batch, max_context)
            post_buf = x_token[1].new_zeros(batch, max_context)
            buf_len = min(initial_len, max_context)
            if buf_len > 0:
                start = max(0, initial_len - max_context)
                pre_buf[:, :buf_len] = x_token[0][:, start : start + buf_len]
                post_buf[:, :buf_len] = x_token[1][:, start : start + buf_len]

            for i in range(pred_len):
                cur = initial_len + i
                wlen = min(cur, max_context)
                if cur <= max_context:
                    tokens = [pre_buf[:, :wlen], post_buf[:, :wlen]]
                else:
                    tokens = [pre_buf, post_buf]
                cs = max(0, cur - max_context)
                stamp = full_stamp[:, cs:cur, :].contiguous()

                s1, ctx = model.decode_s1(tokens[0], tokens[1], stamp)
                pre = sample_from_logits(
                    s1[:, -1, :], temperature=temperature, top_k=0, top_p=top_p, sample_logits=True
                )
                s2 = model.decode_s2(ctx, pre)
                post = sample_from_logits(
                    s2[:, -1, :], temperature=temperature, top_k=0, top_p=top_p, sample_logits=True
                )

                gen_pre[:, i] = pre.squeeze(-1)
                gen_post[:, i] = post.squeeze(-1)
                if cur < max_context:
                    pre_buf[:, cur] = pre.squeeze(-1)
                    post_buf[:, cur] = post.squeeze(-1)
                else:
                    pre_buf.copy_(torch.roll(pre_buf, -1, 1))
                    post_buf.copy_(torch.roll(post_buf, -1, 1))
                    pre_buf[:, -1] = pre.squeeze(-1)
                    post_buf[:, -1] = post.squeeze(-1)

            full_pre = torch.cat([x_token[0], gen_pre], dim=1)
            full_post = torch.cat([x_token[1], gen_post], dim=1)
            cs = max(0, total_len - max_context)
            tokens = [
                full_pre[:, cs:total_len].contiguous(),
                full_post[:, cs:total_len].contiguous(),
            ]
            z = tokenizer.decode(tokens, half=half)
            z = z.reshape(-1, sample_count, z.size(1), z.size(2))  # (1, S, total_len, 6)
            paths = z[:, :, -pred_len:, :].cpu().numpy()[0]  # (S, pred_len, 6)

        return paths * (x_std + 1e-5) + x_mean

    def signal(
        self,
        window: pd.DataFrame,
        horizon: int,
        *,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 20,
    ) -> dict[str, float]:
        """Probabilistic signal at +horizon trading days, from a trailing OHLCV window."""
        last_ts = pd.Timestamp(window.index[-1])
        x_ts = pd.Series(pd.to_datetime(window.index))
        y_index = pd.bdate_range(start=last_ts, periods=horizon + 1)[1:]
        y_ts = pd.Series(y_index)

        paths = self._sample_paths(window, x_ts, y_ts, horizon, temperature, top_p, sample_count)
        terminal_close = paths[:, -1, _CLOSE_IDX]
        return directional_signal(terminal_close, float(window["close"].iloc[-1]))
