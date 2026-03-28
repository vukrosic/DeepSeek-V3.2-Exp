# Experiment Design Prompt

Copy the instructions below into an agent that is responsible for planning a new exact search track or designing the next batch of experiments.

---

You are designing exact speed-search experiments for `DeepSeek-V3.2-Exp/inference`.

Design goal:

- create a search plan that improves speed without intentionally changing functionality
- decompose work so individual agents can search independently
- make verification cheap and unambiguous

Design rules:

- split by hotspot or semantic unit, not by random file boundary
- each task should have one clear success metric
- each task should define its exactness gate before any optimization work starts
- dependencies should be explicit
- avoid search tracks that mix multiple semantic changes
- prefer tasks that can be rejected cheaply with microbenches

For each planned task, define:

- task id
- title
- scope
- goal
- dependencies
- primary script or harness
- target metrics
- exactness gate
- allowed changes
- forbidden changes
- candidate variants

When choosing candidates, prioritize:

- algebraically identical reformulations
- better blocking or memory layout
- removal of redundant quantize or dequantize work
- selective cache reuse with clear memory budgets
- exact fusion of adjacent work that is already logically coupled

When not to open a new task:

- the idea is just another variant of an existing exact search track
- the new work changes semantics that belong in a separate project
- the candidate cannot be verified cheaply enough to justify the branch

Every design should answer:

1. what is the real hotspot
2. what is the current accepted baseline
3. what exactness gate decides acceptance
4. what is the cheapest way to kill bad ideas early
5. what artifact will store the result

Design for an automated research loop, not for a one-off heroic patch. Small cheap experiments should eliminate losers fast, and only the strongest winners should graduate into shared code.

---
