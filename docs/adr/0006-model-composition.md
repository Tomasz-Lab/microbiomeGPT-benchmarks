# 0006. Model components and composition

Status: Proposed
Date: 2026-11-09



## Context
What problem we're solving and why now. 2–4 sentences.

## Decision
What we do. Concrete, in active voice: "We save…", "We don't use…".

Two levels of the interface for benchmarking:
1. Model level, which matters most for benchmarks. One small contract that the Lightning system, 
loss, metrics and prediction writer rely on, for example 
forward(inputs) -> ModelOutput(reconstructions, latent). 
Any model that meets it can be trained and evaluated with the same code, 
even one that isn’t a transformer or isn’t split into parts.
2. Component level, for ablations inside one family.
Tokenizer → Encoder → Latent → Decoder → Head, each with its own small base class, 
plugged into a generic container.

As we do not have a concrete plan for models and architecture to test,
both interfaces should be kept minimal for now until there’s a second model,
so we don’t design for imagined needs.

## Consequences
What we gain, what we lose, what we now need to watch. Include the downsides.