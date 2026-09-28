from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("rqvae_quantization")
class RQVAEQuantizationSkill(BaseSkill):
    """Small residual-quantization-style autoencoder block for RQVAE genomes."""

    skill_spec = SkillSpec(
        name="rqvae_quantization",
        category="generative",
        description="Encode dense item features, quantize against a codebook, and reconstruct them.",
        input_specs=[TensorSpec("dense_input", "[batch_size, input_dim]")],
        output_specs=[
            TensorSpec("quantized", "[batch_size, latent_dim]"),
            TensorSpec("reconstruction", "[batch_size, input_dim]"),
            TensorSpec("rqvae_loss", "[]"),
        ],
        task_types=["generative", "matching"],
        inductive_bias=["discrete_semantic_id_quantization"],
        failure_signatures=["input_dim_mismatch", "codebook_collapse"],
        hyperparameters={"input_dim": "int", "latent_dim": "int", "num_codes": "int"},
        paper_origin="Residual Quantized VAE / TIGER semantic IDs",
    )

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_codes: int,
        input_key: str = "dense_input",
        output_key: str = "quantized",
        reconstruction_key: str = "reconstruction",
        loss_key: str = "rqvae_loss",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_key = input_key
        self.output_key = output_key
        self.reconstruction_key = reconstruction_key
        self.loss_key = loss_key
        self.encoder = torch.nn.Linear(input_dim, latent_dim)
        self.codebook = torch.nn.Embedding(num_codes, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, input_dim)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key).float()
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        latent = self.encoder(x)
        distances = torch.cdist(latent.unsqueeze(1), self.codebook.weight.unsqueeze(0)).squeeze(1)
        code_ids = torch.argmin(distances, dim=-1)
        quantized = self.codebook(code_ids)
        reconstruction = self.decoder(quantized)
        recon_loss = torch.nn.functional.mse_loss(reconstruction, x)
        commit_loss = torch.nn.functional.mse_loss(quantized.detach(), latent)
        ctx.put("latent", latent)
        ctx.put("code_ids", code_ids)
        ctx.put(self.output_key, quantized)
        ctx.put(self.reconstruction_key, reconstruction)
        return ctx.put(self.loss_key, recon_loss + 0.25 * commit_loss)
