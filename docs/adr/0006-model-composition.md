# 0006. Model components and composition

Status: Accepted

Date: 2026-11-09

## Context
`BacteriaModel` is one monolithic class: tokenization, encoder, latent bottleneck, decoder and
output layer are hardcoded together, with several hyperparameters fixed inside the class.
We plan to benchmark different models and run ablations (other layers, depth, modalities),
which today means rewriting or duplicating this class.

## Decision

Two levels of the interface for benchmarking:

1. Model level, the interface every model implements to plug into our pipeline:
   `MicrobiomeModel.forward(inputs) -> ModelOutput(latent)` (`gut_former.models.base`).
   It matters most for benchmarks: every model, ours or an external baseline, learns a
   representation of a microbiome sample, so anything that only needs representations
   (inference outputs, downstream evaluation, comparisons) works with every model.
    - `inputs` is a dict keyed by modality name, so models can use more, fewer or different
      modalities (e.g. for ablations) without changing the interface.
    - `latent` is the deterministic sample representation used downstream
      (e.g. the mean for a VAE, the pooled embedding for a masked encoder).
    - Each learning method extends `ModelOutput` with what its loss needs, and has a system
      that consumes it. Autoencoders return `AutoEncoderOutput`, which adds `reconstructions`
      (keyed by modality); `AutoEncoderSystem` trains `AutoEncoderModel` subclasses
      (`gut_former.models.autoencoder.base`). A VAE would add explicit mean and log-variance
      fields. Classification is a downstream evaluation of the latent, not a model here.


2. Component level, for ablations inside one family of models.
   E.g. `ComposableAutoEncoder` (`gut_former.models.autoencoder`) has the architecture
    Tokenizer → Encoder → Latent → Decoder → Head; it is not tied to transformers.
   Each base class fixes only what goes in and out, so one implementation can be swapped
    for another without touching the rest of the model.
    - Encoder and Decoder take dicts keyed by modality, so modalities can be processed
      either fused from the start or separately and mixed later.
    - Head is the final output layer.

Both interfaces are kept minimal, because we do not have a concrete plan yet for the models
and architectures to test. They are extended when a model needs it, not in advance for
imagined needs.

The tensor shapes of each interface are documented in the docstrings
(`gut_former.models.base`, `gut_former.models.autoencoder.base`, `gut_former.models.components`).

The original, legacy model is rebuilt from these components as GutFormer
(`gut_former.models.autoencoder.gut_former`) and frozen: its classes never change behaviour.
The monolithic `BacteriaModel` is removed rather than kept next to it.
A test checks that GutFormer reproduces the outputs saved from the original model
(`tests/data/gut_former_reference.pt`).

## Consequences
- New models and ablations plug into the same training and evaluation code;
  components can be unit-tested on their own.
- A new learning method (e.g. masked modelling) needs its own output, model base class and
  system; a new architecture with an existing learning method needs only new components.
- GutFormer is frozen in code with all its hyperparameters (number of heads, dropout, layer
  depth, feed-forward width, embedding and latent sizes); its config only names the class.
  To try other values, a component should be rewritten under a descriptive name with normal config
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
  for taxonomy and pathways), a Latent that also returns mean and log-variance for a VAE,
  or a mask passed to the model for masked modelling.
