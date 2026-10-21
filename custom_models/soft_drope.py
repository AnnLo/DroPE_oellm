"""
Soft DroPE: remove RoPE gradually instead of in one step.

A single scalar alpha in [0, 1] (stored as config.rope_alpha) controls how much
RoPE is left. alpha = 1 is the original RoPE model, alpha = 0 is exactly the
DroPE (NoPE) model. A TrainerCallback moves alpha from 1 to 0 during training.

soft_mode:
  "angle"  - rotation angles scaled by alpha: theta_i * m * alpha
             (implemented as inv_freq <- inv_freq_orig * alpha; at alpha=0, cos=1, sin=0)
  "output" - two attention passes, outputs mixed:
             out = alpha * Attn_RoPE(x) + (1 - alpha) * Attn_NoPE(x)

qk_norm ("none" | "q" | "k" | "qk"):
  Adds RMSNorm on the query and/or key projections, the same way as the repo's
  qk_norm_nope variants: one RMSNorm over all heads (num_heads * head_dim), applied to the
  q_proj / k_proj output before RoPE. New weights, initialised to 1.

The final checkpoint is saved with rope_alpha=0.0 in its config, so loading it with
this class gives the NoPE model. Without qk_norm its weights can also be loaded into
custom_models.drope.DroPELlamaForCausalLM (attention_type: nope) for evaluation.
"""
import math
import logging

import torch
from transformers import AutoConfig, AutoModelForCausalLM, TrainerCallback
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaRMSNorm,
)

logger = logging.getLogger(__name__)

SOFT_MODES = ("angle", "output")
QK_NORMS = ("none", "q", "k", "qk")


# --------------------------------------------------------------------------- config
class SoftDroPELlamaConfig(LlamaConfig):
    model_type = "llama_soft_drope"

    def __init__(self, rope_alpha: float = 1.0, soft_mode: str = "angle",
                 qk_norm: str = "none", **kwargs):
        if soft_mode not in SOFT_MODES:
            raise ValueError(f"soft_mode must be one of {SOFT_MODES}, got {soft_mode!r}")
        if qk_norm not in QK_NORMS:
            raise ValueError(f"qk_norm must be one of {QK_NORMS}, got {qk_norm!r}")
        self.rope_alpha = float(rope_alpha)
        self.soft_mode = soft_mode
        self.qk_norm = qk_norm
        super().__init__(**kwargs)


# ------------------------------------------------------------------------ QK norm
def _norm_output_hook(norm):
    # A forward hook that returns a value replaces the module's output.
    def hook(module, inputs, output):
        return norm(output)
    return hook


def add_qk_norm(attn: LlamaAttention, which: str, config) -> None:
    """Add attn.q_norm / attn.k_norm and apply them to the q_proj / k_proj outputs.

    Uses forward hooks, so LlamaAttention.forward (and OutputMixLlamaAttention) stay
    unchanged and the original weight names (q_proj.weight, ...) still load. The norms
    are saved as self_attn.q_norm.weight / self_attn.k_norm.weight. The class name
    contains "RMSNorm", so HF initialises them to 1 when they're missing from a checkpoint.
    """
    if "q" in which:
        attn.q_norm = LlamaRMSNorm(config.num_attention_heads * attn.head_dim, eps=config.rms_norm_eps)
        attn.q_proj.register_forward_hook(_norm_output_hook(attn.q_norm))
    if "k" in which:
        attn.k_norm = LlamaRMSNorm(config.num_key_value_heads * attn.head_dim, eps=config.rms_norm_eps)
        attn.k_proj.register_forward_hook(_norm_output_hook(attn.k_norm))


# ------------------------------------------------------------- "output" attention
class OutputMixLlamaAttention(LlamaAttention):
    """Attention = alpha * (with RoPE) + (1 - alpha) * (without RoPE).

    Mixing after o_proj equals mixing before it (o_proj is linear, no bias in Llama).
    Simple version: calls the base forward twice, so the q/k/v/o projections are also
    computed twice. Training only (use_cache=False); with a KV cache the cache would
    be updated twice. At alpha in {0, 1} only one pass is run.
    """

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        alpha = float(getattr(self.config, "rope_alpha", 1.0))
        cos, sin = position_embeddings
        nope_pe = (torch.ones_like(cos), torch.zeros_like(sin))  # identity rotation

        if alpha >= 1.0:
            return super().forward(hidden_states, position_embeddings, attention_mask, **kwargs)
        if alpha <= 0.0:
            return super().forward(hidden_states, nope_pe, attention_mask, **kwargs)

        out_rope, _ = super().forward(hidden_states, position_embeddings, attention_mask, **kwargs)
        out_nope, _ = super().forward(hidden_states, nope_pe, attention_mask, **kwargs)
        return alpha * out_rope + (1.0 - alpha) * out_nope, None


# -------------------------------------------------------------------------- model
class SoftDroPELlamaForCausalLM(LlamaForCausalLM):
    config_class = SoftDroPELlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self._inv_freq_orig = None  # plain attribute (not a buffer): never saved/loaded
        if config.soft_mode == "output":
            # Swap the class in place: same attributes and weights, new forward.
            for layer in self.model.layers:
                layer.self_attn.__class__ = OutputMixLlamaAttention
        if config.soft_mode == "angle":
            # Runs right before the rotary module computes cos/sin (once per model forward).
            # Cheap: head_dim/2 elements.
            self.model.rotary_emb.register_forward_pre_hook(
                lambda module, args: self._apply_angle_alpha()
            )
        if config.qk_norm != "none":
            for layer in self.model.layers:
                add_qk_norm(layer.self_attn, config.qk_norm, config)
        logger.info(f"SoftDroPE: mode={config.soft_mode}, rope_alpha={config.rope_alpha}, "
                    f"qk_norm={config.qk_norm}")

    @torch.no_grad()
    def _apply_angle_alpha(self):
        """inv_freq <- inv_freq_orig * alpha. Recomputed from the RoPE init function,
        so it doesn't depend on buffer state after loading/moving the model."""
        rot = self.model.rotary_emb
        dev = rot.inv_freq.device
        if self._inv_freq_orig is None or self._inv_freq_orig.device != dev:
            inv_freq, _ = rot.rope_init_fn(rot.config, dev)
            self._inv_freq_orig = inv_freq.float()
        alpha = min(max(float(self.config.rope_alpha), 0.0), 1.0)
        # Assign a NEW tensor: rot.original_inv_freq aliases the old buffer, so no in-place ops.
        rot.inv_freq = self._inv_freq_orig * alpha

    # NOTE: no forward() override here. The HF Trainer inspects model.forward's signature to
    # decide which dataset columns to keep; a generic forward(*args, **kwargs) makes it drop
    # input_ids/labels ("The batch received was empty"). The alpha update is a pre-hook instead.


AutoConfig.register(SoftDroPELlamaConfig.model_type, SoftDroPELlamaConfig)
AutoModelForCausalLM.register(SoftDroPELlamaConfig, SoftDroPELlamaForCausalLM)


# ----------------------------------------------------------------------- schedule
class RopeAlphaSchedule(TrainerCallback):
    """Moves config.rope_alpha from 1 to 0.

    alpha = 1                          for step <  start_frac * max_steps
    alpha = 1 -> 0 (linear | cosine)   between start_frac and end_frac
    alpha = 0                          for step >= end_frac * max_steps  (pure NoPE phase)

    Depends only on global_step, so it is correct after resuming from a checkpoint.
    """

    def __init__(self, start_frac: float = 0.0, end_frac: float = 0.5, shape: str = "linear"):
        assert 0.0 <= start_frac < end_frac <= 1.0
        assert shape in ("linear", "cosine")
        self.start_frac, self.end_frac, self.shape = start_frac, end_frac, shape

    def alpha(self, step: int, max_steps: int) -> float:
        start, end = self.start_frac * max_steps, self.end_frac * max_steps
        t = min(max((step - start) / max(end - start, 1.0), 0.0), 1.0)
        if self.shape == "linear":
            return 1.0 - t
        return 0.5 * (1.0 + math.cos(math.pi * t))

    def _set(self, state, model):
        a = self.alpha(state.global_step, state.max_steps)
        if model is not None:
            getattr(model, "module", model).config.rope_alpha = a
        return a

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        a = self._set(state, model)
        logger.info(f"RopeAlphaSchedule: step {state.global_step}/{state.max_steps}, alpha={a:.4f}")

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._set(state, model)

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"train/rope_alpha": self.alpha(state.global_step, state.max_steps),
                           "train/global_step": state.global_step})
        except ImportError:
            pass

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # The final model is saved after this. Its config must hold the alpha it ended at
        # (0.0 when the schedule finished), so it reloads as the NoPE model.
        a = self._set(state, model)
        logger.info(f"RopeAlphaSchedule: training ended at step {state.global_step}, alpha={a:.4f}")


# ============================================================================ Qwen3 (prelude)
# Angle mode only. Qwen3 already has trained q_norm/k_norm, so there is no qk_norm option.
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM


class SoftDroPEQwen3Config(Qwen3Config):
    model_type = "qwen3_soft_drope"

    def __init__(self, rope_alpha: float = 1.0, soft_mode: str = "angle", **kwargs):
        if soft_mode != "angle":
            raise ValueError("SoftDroPEQwen3 supports soft_mode='angle' only")
        self.rope_alpha = float(rope_alpha)
        self.soft_mode = soft_mode
        super().__init__(**kwargs)


class SoftDroPEQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = SoftDroPEQwen3Config

    def __init__(self, config):
        super().__init__(config)
        self._inv_freq_orig = None
        self.model.rotary_emb.register_forward_pre_hook(lambda module, args: self._apply_angle_alpha())
        logger.info(f"SoftDroPE (Qwen3): mode=angle, rope_alpha={config.rope_alpha}")

    # same as the Llama version: inv_freq <- inv_freq_orig * alpha (alpha=0 -> cos=1, sin=0)
    _apply_angle_alpha = SoftDroPELlamaForCausalLM._apply_angle_alpha


AutoConfig.register(SoftDroPEQwen3Config.model_type, SoftDroPEQwen3Config)
AutoModelForCausalLM.register(SoftDroPEQwen3Config, SoftDroPEQwen3ForCausalLM)