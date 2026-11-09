# 0006. Model components and composition

Status: Proposed
Date: 2026-11-09



## Context
What problem we're solving and why now. 2–4 sentences.

## Decision
What we do. Concrete, in active voice: "We save…", "We don't use…".

forward(inputs) -> ModelOutput(reconstructions, latent).
Any model that meets it can be trained and evaluated with the same code,
even one that isn’t a transformer or isn’t split into parts.

## Consequences
What we gain, what we lose, what we now need to watch. Include the downsides.