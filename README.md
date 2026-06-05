# oxide-fleet

Fleet coordination layer for the Flux→PTX distributed GPU runtime.

`oxide-fleet` is the control plane that binds heterogeneous GPU agents into a single, addressable compute surface. It handles agent discovery by capability, work distribution with priority awareness, real-time status tracking, and rhythm-based workload optimization. If you are running a multi-node GPU cluster and need something between "manual SSH scripts" and "full Kubernetes," this is your layer.

---

## What This Crate Does

In a distributed GPU system, you have nodes with different hardware generations, varying VRAM capacities, and heterogeneous capabilities—some agents compile Flux IR to PTX, others execute kernels, others synchronize state via CRDTs. `oxide-fleet` gives you a single `FleetCoordinator` that:

- Maintains a live registry of every agent and its GPU inventory
- Matches work requests to agents based on **capabilities**, not just labels
- Tracks agent lifecycle from `Online` → `Busy` → `Draining` → `Offline`
- Prioritizes work across four explicit levels from `Low` to `Critical`
- Detects load imbalance and hotspots through **rhythm analysis**

This is not a job queue. It is a placement engine with awareness of what your agents can actually do.

---

## Agent Discovery by Capabilities

Agents are not identified by static hostnames. They advertise **capabilities**—structured descriptors of what they can provide.

```rust
use oxide_fleet::{Capability, FleetAgent, GpuDevice, AgentStatus, WorkloadInfo};

let agent = FleetAgent {
    id: "node-c-01".to_string(),
    node: "cuda-rack-3".to_string(),
    capabilities: vec![
        Capability::FluxCompiler,
        Capability::KernelExecutor { min_sm: 80 },  // SM 8.0+
        Capability::CrdtSync,
        Capability::Custom("nvlink-mesh".to_string()),
    ],
    gpu_devices: vec![
        GpuDevice {
            node_id: "cuda-rack-3".to_string(),
            gpu_index: 0,
            compute_capability: 80,
            vram_mb: 24_576,        // 24 GB
            has_tensor_cores: true,
            is_available: true,
        },
    ],
    status: AgentStatus::Online,
    workload: WorkloadInfo::default(),
};
```

Capabilities are typed:

| Capability | Meaning |
|---|---|
| `FluxCompiler` | Can compile Flux IR down to PTX |
| `KernelExecutor { min_sm }` | Can launch PTX kernels; requires at least `min_sm` streaming multiprocessors |
| `ConstructLoader` | Can load constructs from git-backed storage |
| `CrdtSync` | Can participate in CRDT-based state synchronization |
| `Custom(s)` | Arbitrary domain-specific capability |

Discovery uses **subsumption matching**: a `KernelExecutor { min_sm: 80 }` request will match an agent advertising `min_sm: 90`, but not one advertising `min_sm: 70`.

```rust
use oxide_fleet::{FleetCoordinator, Capability};

let mut fleet = FleetCoordinator::new();
fleet.register_agent(agent);

// Find every agent that can execute kernels on SM 8.0+ hardware
let capable = fleet.discover(&[
    Capability::KernelExecutor { min_sm: 80 },
]);
```

---

## Work Request System with Priorities

Work is submitted as a `WorkRequest`, not a raw binary blob. The request carries both **functional requirements** (capabilities, minimum compute capability, minimum VRAM) and **operational metadata** (priority, estimated duration).

```rust
use oxide_fleet::{WorkRequest, WorkPriority, Capability};

let req = WorkRequest {
    id: "inference-batch-7721".to_string(),
    required_capabilities: vec![
        Capability::KernelExecutor { min_sm: 80 },
        Capability::CrdtSync,
    ],
    min_compute_capability: 80,
    min_vram_mb: 12_288,          // 12 GB
    priority: WorkPriority::High,
    estimated_duration_ms: 45_000,
};
```

Priority levels are ordered and comparable:

| Level | Ordinal | Typical Use |
|---|---|---|
| `Low` | 0 | Backfill, batch preprocessing, cache warming |
| `Normal` | 1 | Standard training iterations, routine inference |
| `High` | 2 | Latency-sensitive inference, checkpoint commits |
| `Critical` | 3 | Failure recovery, control-plane heartbeats, emergency checkpointing |

The coordinator currently uses priority to inform placement decisions in the caller layer. The `WorkPriority` type implements `Ord`, so you can build your own priority-queue wrapper on top without friction.

---

## Assigning Work

`assign_work` is greedy but workload-aware. It filters by capability, then by status (`Online` only), then by GPU fit, and finally selects the agent with the lowest `(running_kernels, gpu_utilization_pct)` tuple.

```rust
let assignment = fleet.assign_work(&req)?;
println!(
    "Assigned {} to agent {} on GPU {}. Reason: {}",
    assignment.work_id,
    assignment.agent_id,
    assignment.gpu_index,
    assignment.reason
);
// "Assigned inference-batch-7721 to agent node-c-01 on GPU 0. \
//  Reason: best fit: 2 kernels running, 34% util"
```

When work completes, call `complete_work` to remove the assignment from the pending set:

```rust
let finished = fleet.complete_work("inference-batch-7721");
assert_eq!(finished.unwrap().agent_id, "node-c-01");
```

If no agent satisfies the request, `assign_work` returns `FleetError::NoAvailableAgent` immediately. There is no hidden queuing—failures are explicit and observable.

---

## Agent Status Tracking

Agents move through a finite state of statuses:

| Status | Meaning |
|---|---|
| `Online` | Agent is healthy and accepting work |
| `Busy { task }` | Agent is currently occupied with a named task |
| `Offline` | Agent is unreachable—no work will be assigned |
| `Draining` | Agent is finishing in-flight work; new assignments are rejected |

`Draining` is the graceful off-ramp. Use it when you need to reboot a node, update drivers, or evacuate a rack. The coordinator still counts a draining agent as present, but `assign_work` will skip it because its status is no longer `Online`.

```rust
use oxide_fleet::{FleetAgent, AgentStatus};

let mut agent = make_agent("node-c-01", 80, 24_576);
agent.status = AgentStatus::Draining;
fleet.register_agent(agent);  // Existing registrations are overwritten
```

The `WorkloadInfo` struct carries runtime telemetry:

```rust
WorkloadInfo {
    running_kernels: 4,
    pending_tasks: 1,
    gpu_utilization_pct: 67,
    memory_used_mb: 19_200,
}
```

Feed this from your node-level metrics daemon. The coordinator does not poll—push updates into `FleetAgent` and re-register.

---

## Rhythm Analysis

After work has been flowing for a while, you need to know whether your fleet is actually balanced or whether one agent is eating all the traffic. `analyze_rhythm` returns a `RhythmAnalysis` that surfaces:

- **Total assignments** — how many placements have been recorded
- **Unique agents touched** — coverage of the fleet
- **Load imbalance ratio** — max assignments per agent divided by the mean; 1.0 is perfectly uniform
- **Hotspots** — agents receiving >1.5× the average assignment count

```rust
let rhythm = fleet.analyze_rhythm();

println!("Assignments: {} across {} agents", rhythm.total_assignments, rhythm.unique_agents);
println!("Load imbalance: {:.2}", rhythm.load_imbalance);
println!("Hotspots: {:?}", rhythm.hotspots);
```

Example output after 10,000 inference requests across 8 nodes:

```text
Assignments: 10000 across 8 agents
Load imbalance: 2.34
Hotspots: ["node-c-01", "node-c-04"]
```

A ratio above ~1.5 is a signal to investigate: tensor-core affinity, NVLink topology, or stale workload telemetry causing the scheduler to over-prefer certain nodes. Rhythm analysis gives you the data to act, not just observe.

---

## FleetStats Overview

Call `stats()` at any time for a point-in-time snapshot of the fleet:

```rust
let s = fleet.stats();
println!("Agents: {}/{} online", s.online_agents, s.total_agents);
println!("GPUs: {}/{} available", s.available_gpus, s.total_gpus);
println!("Kernels running: {}", s.running_kernels);
println!("Pending assignments: {}", s.pending_assignments);
```

| Field | Description |
|---|---|
| `total_agents` | Registered agents |
| `online_agents` | Agents currently in `Online` status |
| `total_gpus` | GPUs across all registered agents |
| `available_gpus` | GPUs marked `is_available == true` |
| `running_kernels` | Sum of `workload.running_kernels` fleet-wide |
| `pending_assignments` | Active assignments not yet completed |

Use `FleetStats` for dashboards, health-check endpoints, and autoscaling triggers.

---

## Relationship to the Agent Ecosystem

`oxide-fleet` does not exist in a vacuum. It sits at the center of three companion crates:

- **agent-handshake** — Capability negotiation and secure joining. When a new node boots, it runs the handshake protocol to prove it can provide the capabilities it claims. `oxide-fleet` trusts the resulting `FleetAgent` descriptor and inserts it into the coordinator.
- **agent-manifest** — Declarative agent configuration. The manifest specifies which capabilities an agent should advertise, which GPUs to expose, and node metadata. Fleet operators edit manifests; the agent runtime applies them; `oxide-fleet` consumes the resulting agent records.
- **agent-rhythm** — Workload-pattern optimization. While `oxide-fleet` provides the `analyze_rhythm` diagnostic, `agent-rhythm` consumes that output and generates concrete rebalancing recommendations—migrating kernels, adjusting affinity masks, or suggesting fleet topology changes.

In short: **handshake gets you in, manifest tells you what you are, fleet moves work to you, and rhythm tells you whether the movement is healthy.**

---

## Error Handling

All placement failures are explicit:

```rust
pub enum FleetError {
    NoAvailableAgent,                        // No agent matched constraints
    AgentNotFound(String),                   // Dereference of unknown agent ID
    CapabilityMismatch { agent, required },  // Capability subsumption failed
    GpuUnavailable { agent, gpu_index },     // Target GPU marked unavailable
}
```

`FleetError` implements `std::error::Error` and `Display`. Propagate it with `?` or match on it to decide whether to retry, relax constraints, or page an operator.

---

## Quick Start

```rust
use oxide_fleet::*;

fn main() -> Result<(), FleetError> {
    let mut fleet = FleetCoordinator::new();

    // Register two nodes
    fleet.register_agent(FleetAgent {
        id: "gpu-01".into(),
        node: "rack-a".into(),
        capabilities: vec![
            Capability::KernelExecutor { min_sm: 80 },
            Capability::FluxCompiler,
        ],
        gpu_devices: vec![GpuDevice {
            node_id: "rack-a".into(),
            gpu_index: 0,
            compute_capability: 80,
            vram_mb: 24_576,
            has_tensor_cores: true,
            is_available: true,
        }],
        status: AgentStatus::Online,
        workload: WorkloadInfo::default(),
    });

    // Find compilers
    let compilers = fleet.discover(&[Capability::FluxCompiler]);
    println!("Compilers available: {}", compilers.len());

    // Submit critical work
    let job = WorkRequest {
        id: "compile-pass-7".into(),
        required_capabilities: vec![Capability::FluxCompiler],
        min_compute_capability: 80,
        min_vram_mb: 4096,
        priority: WorkPriority::Critical,
        estimated_duration_ms: 120_000,
    };

    let assignment = fleet.assign_work(&job)?;
    println!("Assigned to {} (GPU {})", assignment.agent_id, assignment.gpu_index);

    // Later...
    fleet.complete_work(&job.id);

    // Check fleet health
    let stats = fleet.stats();
    let rhythm = fleet.analyze_rhythm();
    println!("Fleet: {:?}", stats);
    println!("Rhythm: {:?}", rhythm);

    Ok(())
}
```

---

## License

Apache-2.0
