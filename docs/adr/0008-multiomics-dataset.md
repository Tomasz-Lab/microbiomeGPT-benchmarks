# 0008. Dataset representation: MultiOmicsDataset

Status: Accepted
Date: 2026-11-16

## Context
ADR 0007 defined what a modality is, but not how one loaded dataset (e.g. cMD) is held in memory. Today every model
loads and splits its own tables, so nothing guarantees that taxonomy, pathways and metadata contain the same samples,
or that a train/test split is applied to all of them the same way.

## Decision
TBD

## Consequences
TBD