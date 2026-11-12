# 0006. Model components and composition

Status: Proposed
Date: 2026-11-09

## Context
`BacteriaModel` is one monolithic class: tokenization, encoder, latent bottleneck, decoder and
output layer are hardcoded together, with several hyperparameters fixed inside the class.
We plan to benchmark different models and run ablations (other layers, depth, modalities),
which today means rewriting or duplicating the class (#18).

## Decision

Two levels of the interface for benchmarking:

1. Model level, which matters most for benchmarks. One small contract that the Lightning system,
   loss, metrics and prediction writer rely on:
   `forward(inputs) -> ModelOutput(reconstructions, latent, aux)`.
    - `inputs` and `reconstructions` are dicts keyed by modality name, so models can use more,
      fewer or different modalities (e.g. for ablations) without changing the interface.
    - `aux` holds extra tensors a loss may need (e.g. mean and log-variance of a VAE);
      it is empty for a plain autoencoder.
    - Any model that meets it can be trained and evaluated with the same code,
      even one that isn't a transformer or isn't built from the components below.


2. Component level, for ablations inside one family.
   Tokenizer → Encoder → Latent → Decoder → Head, each with its own small base class,
   plugged into a generic container. Each base class fixes only what goes in and out,
   so one implementation can be swapped for another without touching the rest of the model.
    - Encoder and Decoder take dicts keyed by modality, so modalities can be processed
      either fused from the start or separately and mixed later.
    - Head is the final output layer (hidden state → one reconstructed value per feature),
      not the attention heads, which are internal to the Encoder and Decoder.

E.g. `ComposableAutoEncoder` (`gut_former.models.autoencoder`) has the architecture
Tokenizer → Encoder → Latent → Decoder → Head; it is not tied to transformers.

Both interfaces are kept minimal, because we do not have a concrete plan yet for the models
and architectures to test. We extend them when a second model needs it,
so we don't design for imagined needs.

The tensor shapes of each interface are documented in the code
(`gut_former.models.base`, `gut_former.models.components`), not here.

The original model is rebuilt from these components as GutFormer
(`gut_former.models.autoencoder.gut_former`) and frozen: its classes never change behaviour;
a change means a new class. We do not keep the monolithic `BacteriaModel` next to it.
A test checks that GutFormer reproduces the outputs saved from the original model
(`tests/data/gut_former_reference.pt`).

## Consequences
- New models and ablations plug into the same training and evaluation code;
  components can be unit-tested on their own.
- GutFormer is frozen in code with all its hyperparameters (number of heads, dropout, layer
  depth, feed-forward width, embedding and latent sizes); its config only names the class.
  To try other values, a component is rewritten under a descriptive name with normal config
  parameters. Flexible settings for benchmarks belong to the components of new models.
- Models are configured as a composition of components (`conf/model/*.yaml`), and the
  checkpoint stores the resolved composition, so it records which classes and settings were used.
- Parameter names in the `state_dict` change, so checkpoints of the original `BacteriaModel`
  do not load into GutFormer. We do not provide a name mapping, because no original trained
  weights need to be reused; the legacy scripts load only `.pt` files trained after this change.
- Weight initialization consumes the random number generator in a different order,
  so a fresh training run with the same seed is not bit-for-bit identical to an old one,
  although the architecture is the same.
- The interfaces may need to grow, e.g. a per-modality Head (different output forms
  for taxonomy and pathways) or a Latent that returns `aux` for a VAE.
