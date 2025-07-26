from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BasicsTransformerLM(nn.Module):
    """A Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner-layer (section 3.3).
        attn_pdrop: Optional[Float], default is None.
            If given, drop-out the attention probabilities with this rate.
        residual_pdrop: Optional[Float], default is None.
            If given, apply dropout to output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v
            for k, v in locals().items()
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.residual_pdrop = residual_pdrop
        # report number of parameters
        logger.info(
            "number of non-embedding parameters: %.2fM" % (self.get_num_params() / 1e6,)
        )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the token and position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embeddings.weight.numel()
            n_params -= self.token_embeddings.weight.numel()
        return n_params

    def forward(self, x: torch.LongTensor):
        """
        Args:
            x: LongTensor of shape `(batch_size, sequence_length)`.
                Input IDs for language modeling.

        Returns: A FloatTensor of shape
            (batch size, sequence_length, vocab_size) with the predicted next-word
            distribution for each token).
        """
        _, sequence_length = x.size()
        # (batch size, sequence_length, d_model)
        # NOTE: paper mentions "In the embedding layers, we multiply those
        # weights by sqrt(d_model)", but we aren't doing that here.
        embedded_tokens = self.token_embeddings(x)

        # Shape: (1, sequence_length)
        positions = torch.arange(
            0, sequence_length, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        # (1, sequence_length, d_model)
        embedded_positions = self.position_embeddings(positions)
        # (batch size, sequence_length, d_model)
        x = embedded_tokens + embedded_positions
        if self.residual_pdrop:
            # (batch size, sequence_length, d_model)
            x = F.dropout(x, self.residual_pdrop)
        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)
        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)
        # (batch size, sequence_length, vocab_size)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            # Get the logits from the model
            logits = self.forward(x)
            # Take the logits for the next token
            next_token_logits = logits[:, -1]
            # apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # If top-k is provided, take the tokens with the highest score
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # Get the score of the kth item that we kept---items with lower scores should be masked.
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(
                    topk_mask, float("-inf")
                )
            next_token_probabilities = F.softmax(
                temperature_scaled_next_token_logits, dim=-1
            )
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # End generation if we see the EOS token ID
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model


class TransformerBlock(nn.Module):
    """A single Transformer layer.

    This implements a single layer of the Transformer, as described in section 3.1
    of the paper.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner-layer (section 3.3).
        attn_pdrop: Optional[Float], default is None.
            If given, drop-out the attention probabilities with this rate.
        residual_pdrop: Optional[Float], default is None.
            If given, apply dropout to output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_pdrop if attn_pdrop else 0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model, bias=False)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.ln2 = nn.LayerNorm(d_model, bias=False)
        self.residual_pdrop = residual_pdrop

    def forward(self, x: torch.FloatTensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        # NOTE: this is a pre-norm Transformer, and differs from the original
        # description in the paper.
        # Apply the multi-head self-attention sublayer
        x_ln = self.ln1(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        x_attn = self.attn(
            x_ln, x_ln, x_ln, need_weights=False, attn_mask=causal_mask, is_causal=True
        )[0]
        if self.residual_pdrop is not None:
            x_attn = F.dropout(x_attn, self.residual_pdrop)
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        if self.residual_pdrop is not None:
            x_ffn = F.dropout(x_ffn, self.residual_pdrop)
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        return x
