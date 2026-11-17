# 0006. Models share one interface and are built from composable components

Status: Accepted
Date: 2026-11-09

## Context
`BacteriaModel` is one monolithic class: tokenization, encoder, latent bottleneck, decoder and
output layer are hardcoded together, with several hyperparameters fixed inside the class.
We plan to benchmark different models and run ablations (other layers, depth, modalities),
which today means rewriting or duplicating this class. The models we compare all learn
representations of microbiome samples in a latent space; their architectures and learning
methods differ. This repository benchmarks foundation models for the human gut microbiome, so
we shape the interfaces around these models and extend them as our research requires.

## Decision

We define two levels of interface:

1. Model level defines the interface any model must implement to plug into our pipeline.
   It is a single small contract (`gut_former.models.base`):
   `MicrobiomeModel.forward(inputs: dict[str, Tensor]) -> RepresentationOutput`.
   It matters most for benchmarks because any model that satisfies it, ours or an external
   baseline, returns a representation of each microbiome sample (`latent`), so it can be
   evaluated and compared with the same code. Training also relies on the output of the model's
   learning method (see below): the Lightning system, loss, metrics and prediction writer of
   that method, e.g. `AutoEncoderSystem`.
    - `inputs` is a dict keyed by modality name, so models can use more, fewer or different
      modalities (e.g. for ablations) without changing the interface.
    - `latent` is the deterministic sample representation used downstream
      (e.g. the mean for a VAE, the pooled embedding for a masked encoder).
    - Each learning method extends `RepresentationOutput` with what its loss needs, and has a
      system that consumes it. Autoencoders return `AutoEncoderOutput`, which adds `reconstructions`
      (keyed by modality); `AutoEncoderSystem` trains `AutoEncoderModel` subclasses
      (`gut_former.models.autoencoder.base`). Classification is a downstream evaluation of
      the latent, not a model here.

2. Component level, for ablations and architectural tests within one learning method.
   E.g. `ComposableAutoEncoder` (`gut_former.models.autoencoder`) has the architecture
    Tokenizer → Encoder → Latent → Decoder → Head; it is not tied to transformers.
   Each base class fixes only what goes in and out, so one implementation can be swapped
    for another without touching the rest of the model.
    - Encoder and Decoder take dicts keyed by modality, so modalities can be processed
      either fused from the start or separately and mixed later.
    - Head is the final output layer.

We keep both interfaces minimal, because we do not have a concrete plan yet for the models
and architectures to test. We extend them when a model needs it, not in advance for
imagined needs.

Examples of extensions, planned or possible:
- Architectures, as new components of an existing learning method. Planned in #73: a
  configurable transformer autoencoder that can mirror GutFormer, composed of
  `FeatureEmbeddingTokenizer`, `JointTransformerEncoder`, `PooledLinearLatent` (with a
  `Pooling` such as `MeanPooling`), `SharedTransformerDecoder` and `LinearHead`.
  Possible later: a `LateFusionEncoder` (modalities encoded separately and mixed later),
  a `VariationalLatent`, a `PerModalityHead` (different output forms for taxonomy and
  pathways), or whatever we come up with during research.
- Learning methods, each with its own output, model base class and system (none designed
  yet): `VariationalAutoEncoderOutput` (adds `mean` and `log_variance`),
  `MaskedModellingOutput` (predictions for masked features; the mask must also reach the
  model), `ContrastiveOutput` (projections of two augmented views of each sample).

The tensor shapes of each interface are documented in the docstrings
(`gut_former.models.base`, `gut_former.models.autoencoder.base`, `gut_former.models.components`).

We rebuild the original model from these components as GutFormer
(`gut_former.models.autoencoder.gut_former`) and freeze it: its classes never change behaviour.
We remove the monolithic `BacteriaModel` rather than keep it next to GutFormer.
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
- The interfaces may need to grow when the extensions listed in the Decision arrive, e.g. the
  model input for masked modelling, or the Latent interface for a VAE.
