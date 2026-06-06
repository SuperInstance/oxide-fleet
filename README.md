# oxide-fleet

> **Fleet coordination layer for the Flux→PTX distributed GPU runtime.**

[![Crates.io](https://img.shields.io/crates/v/oxide-fleet)](https://crates.io/crates/oxide-fleet)
[![Docs.rs](https://docs.rs/oxide-fleet/badge.svg)](https://docs.rs/oxide-fleet)
[![License](https://img.shields.io/crates/l/oxide-fleet)](LICENSE)

---

## Table of Contents

- [Background](#background)
- [How It Works](#how-it-works)
  - [Agent Discovery & Capability Negotiation](#agent-discovery--capability-negotiation)
  - [Workload Distribution](#workload-distribution)
  - [Rhythm-Based Optimization](#rhythm-based-optimization)
- [Architecture Overview](#architecture-overview)
- [Applications](#applications)
- [Getting Started](#getting-started)
- [Related Projects](#related-projects)
- [License](#license)

---

## Background

Modern GPU computing is no longer confined to a single device. As model sizes grow and inference pipelines demand ever-lower latency, workloads must fan out across heterogeneous clusters of accelerators—each with different compute capabilities, memory capacities, and availability windows. The challenge is not merely *running* code on a GPU; it is *orchestrating* thousands of cooperative agents across dozens of nodes without drowning in coordination overhead.

**oxide-fleet** was born from the [Flux→PTX](https://github.com/SuperInstance/SuperInstance) ecosystem, an ambitious effort to compile high-level agentic intent (Flux bytecode) directly to NVIDIA PTX and execute it on distributed GPUs. In that stack, `oxide-fleet` sits at the coordination layer: it discovers agents, negotiates their capabilities, distributes work requests, and continuously optimizes the fleet-wide workload rhythm.

The crate is intentionally minimal. It does not speak CUDA directly, nor does it manage network sockets. Instead, it provides the *decision engine*—the data structures and algorithms that turn a chaotic bag of GPU-enabled agents into an organized, queryable, and load-balanced fleet.

---

## How It Works

At its heart, `oxide-fleet` is a stateful coordinator. You register `FleetAgent` instances with it, each advertising a set of `Capability` values and a collection of `GpuDevice` descriptors. When a `WorkRequest` arrives, the coordinator discovers eligible agents, scores them by current load, and produces an `Assignment`. Over time, it accumulates a history of assignments and can analyze that history to detect hotspots and load imbalance.

### Agent Discovery & Capability Negotiation

Agents are not anonymous workers. Each one carries a typed capability vector:

- **`FluxCompiler`** — the agent can compile Flux bytecode to an intermediate representation.
- **`KernelExecutor { min_sm }`** — the agent can execute PTX kernels, and it advertises the minimum SM (Streaming Multiprocessor) version it supports.
- **`ConstructLoader`** — the agent can dynamically load computational constructs from external sources (e.g., git-backed compute graphs).
- **`CrdtSync`** — the agent can participate in CRDT-based state synchronization.
- **`Custom(String)`** — an open-ended escape hatch for domain-specific capabilities.

Discovery is performed via set intersection: a work request declares which capabilities it requires, and the coordinator returns only those agents whose capability vectors are supersets of the request. For `KernelExecutor`, the coordinator performs a *semantic* match: an agent advertising `min_sm: 80` satisfies a request for `min_sm: 70`, but not the reverse. This subtle detail prevents kernels from being scheduled on hardware that cannot execute them.

### Workload Distribution

Once candidates are identified, the coordinator applies a multi-stage filter:

1. **Status filter** — only `Online` agents are considered. Agents that are `Busy`, `Offline`, or `Draining` are skipped.
2. **Hardware filter** — the agent must possess at least one GPU whose compute capability and VRAM meet or exceed the request's minimums, and that GPU must currently be available.
3. **Load scoring** — remaining candidates are ranked by `(running_kernels, gpu_utilization_pct)`. The agent with the lowest tuple wins.

The result is a "best-fit" assignment that spreads work across the fleet rather than hammering the first capable agent. When the work completes, the caller informs the coordinator via `complete_work`, which removes the pending assignment and frees the GPU for future requests.

### Rhythm-Based Optimization

Long-running fleets develop patterns. Some agents become hotspots; others sit under-utilized. `oxide-fleet` captures every assignment in an `assignment_history` and exposes `analyze_rhythm`, which computes:

- **Total assignments** — the raw throughput of the fleet.
- **Unique agents touched** — a proxy for distribution breadth.
- **Load imbalance ratio** — the maximum assignments received by any single agent divided by the per-agent average. A value of `1.0` is perfect balance; values above `1.5` indicate hotspots.
- **Hotspot list** — the specific agent IDs that are receiving disproportionate load.

This analysis is intentionally lightweight. It runs in-memory, requires no external time-series database, and can be polled periodically by a control loop to trigger rebalancing actions—migrating queued work away from hotspots or spinning up new agents in the affected node pools.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     oxide-fleet                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   FleetAgent │  │  WorkRequest │  │  FleetCoordinator │  │
│  │  (discovery) │  │ (scheduling) │  │   (orchestration) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         │                 │                    │            │
│         ▼                 ▼                    ▼            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Capability  │  │  Assignment  │  │  RhythmAnalysis  │  │
│  │  negotiation │  │   decision   │  │   (hotspots)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   Flux→PTX Distributed Runtime │
              │   (cuda-oxide / cudaclaw)      │
              └───────────────────────────────┘
```

The crate exposes a small, orthogonal API surface:

- **`FleetCoordinator`** — the central registry and scheduler.
- **`FleetAgent`** / **`GpuDevice`** — hardware topology descriptors.
- **`Capability`** — a strongly-typed capability taxonomy.
- **`WorkRequest`** / **`WorkPriority`** — declarative job specifications.
- **`Assignment`** — an immutable scheduling decision.
- **`FleetStats`** / **`RhythmAnalysis`** — observability and optimization signals.
- **`FleetError`** — structured error variants for every failure mode.

All core types implement `Debug` and `Clone`, making them easy to serialize, log, or ferry across async boundaries.

---

## Applications

While `oxide-fleet` was designed for the Flux→PTX stack, its abstraction level makes it suitable for any domain that needs capability-aware scheduling across a heterogeneous agent pool:

- **Distributed inference serving** — Route prompt-processing jobs to nodes with tensor-core support and sufficient free VRAM.
- **GPU-native agent swarms** — Coordinate thousands of lightweight agents executing on persistent CUDA kernels, each with distinct roles (compilation, execution, synchronization).
- **Dynamic construct markets** — Load and unload computational "constructs" (kernels, compute graphs, model shards) from git-backed registries, scheduling them only on agents that advertise `ConstructLoader` capabilities.
- **Heterogeneous cluster management** — Unify nodes with different GPU generations (e.g., A100s, H100s, RTX 4090s) behind a single scheduling policy that respects compute-capability differences.
- **Research testbeds** — Rapidly prototype new scheduling heuristics (power-aware, thermal-aware, or cost-aware) by swapping the scoring logic inside `assign_work`.

---

## Getting Started

Add `oxide-fleet` to your `Cargo.toml`:

```toml
[dependencies]
oxide-fleet = "0.1"
```

Register agents and schedule work:

```rust
use oxide_fleet::{FleetCoordinator, FleetAgent, Capability, GpuDevice, AgentStatus, WorkRequest, WorkPriority};

let mut coord = FleetCoordinator::new();

coord.register_agent(FleetAgent {
    id: "node-a-gpu-0".into(),
    node: "node-a".into(),
    capabilities: vec![Capability::KernelExecutor { min_sm: 80 }, Capability::FluxCompiler],
    gpu_devices: vec![GpuDevice {
        node_id: "node-a".into(),
        gpu_index: 0,
        compute_capability: 80,
        vram_mb: 40_960,
        has_tensor_cores: true,
        is_available: true,
    }],
    status: AgentStatus::Online,
    workload: Default::default(),
});

let request = WorkRequest {
    id: "inference-batch-42".into(),
    required_capabilities: vec![Capability::KernelExecutor { min_sm: 80 }],
    min_compute_capability: 80,
    min_vram_mb: 8_192,
    priority: WorkPriority::High,
    estimated_duration_ms: 500,
};

let assignment = coord.assign_work(&request)?;
println!("Assigned to {} on GPU {}", assignment.agent_id, assignment.gpu_index);

// ... later, when the kernel finishes ...
coord.complete_work(&request.id);
```

Run the built-in tests to verify behavior:

```bash
cargo test
```

---

## Related Projects

- **[SuperInstance](https://github.com/SuperInstance/SuperInstance)** — The umbrella project for the Flux→PTX distributed GPU runtime. `oxide-fleet` is the coordination layer within that ecosystem.
- **cuda-oxide** — The compiler pipeline that translates Pliron IR through NVVM and LLVM to PTX.
- **cudaclaw** — The persistent-kernel runtime that executes agent logic directly on the GPU.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
