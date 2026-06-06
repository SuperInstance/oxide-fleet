# oxide-fleet

> Distributed GPU orchestration through capability negotiation and rhythm-aware workload placement.

## Background Theory

Modern GPU clusters are not homogeneous. A single datacenter may contain V100s, A100s, H100s, and edge-class GPUs, each with different compute capabilities, memory capacities, and specialized features. Treating them as identical resources leads to fragmentation, load imbalance, and failed kernel launches.

The theoretical foundation of `oxide-fleet` is **capability-based resource allocation**. Instead of scheduling work to "a GPU," we schedule to "an agent that can compile Flux to PTX, has SM 8.0+, 8GB+ VRAM, and is currently online." This transforms scheduling from a naming problem into a constraint-satisfaction problem.

A secondary foundation is **rhythm analysis** — the observation that GPU workloads have temporal structure. Some kernels are invoked in bursts, others follow diurnal patterns, and others produce cascading fan-out. By tracking assignment history, the fleet coordinator can identify hotspots and predict where load will concentrate before it happens.

## How It Works

At the center of `oxide-fleet` is the `FleetCoordinator`, which maintains a registry of `FleetAgent`s. Each agent advertises:

- **Capabilities**: What it can do (compile Flux, execute kernels, load constructs, sync CRDTs, or custom capabilities).
- **GPU devices**: Compute capability, VRAM, tensor core availability, availability bit.
- **Status**: Online, Busy, Offline, or Draining.
- **Workload**: Running kernels, pending tasks, GPU utilization, memory usage.

When a `WorkRequest` arrives, the coordinator runs a multi-stage filter:

1. **Capability discovery**: Only agents that satisfy all required capabilities are considered.
2. **GPU filtering**: Available GPUs must meet `min_compute_capability` and `min_vram_mb`.
3. **Load balancing**: Among qualified agents, the one with the lowest `(running_kernels, gpu_utilization)` is chosen.

The result is an `Assignment` that includes the chosen agent, GPU index, and a human-readable reason.

### Rhythm Analysis

The coordinator maintains an `assignment_history`. Periodically, it runs `analyze_rhythm()` to compute:

- **Total assignments**: Volume of work flowing through the fleet.
- **Unique agents**: How many distinct agents participated.
- **Load imbalance**: Ratio of max assignments to average assignments per agent.
- **Hotspots**: Agents receiving >1.5× the average load.

These metrics enable predictive scaling. If one agent becomes a persistent hotspot, the fleet can preemptively drain it or spin up a replica.

## Experiments

The test suite encodes several experimental claims:

```rust
#[test]
fn test_prefers_lower_workload() {
    // Verifies that work is routed away from busy agents.
}

#[test]
fn test_rhythm_analysis() {
    // Verifies that assignment history produces measurable load imbalance.
}
```

A larger experiment: deploy 10 simulated agents with heterogeneous capabilities and submit 1,000 work requests drawn from a Pareto distribution. Expected results:

- 99% of requests succeed if the fleet has aggregate capacity.
- The busiest agent receives no more than 1.5× the average load.
- Rhythm analysis identifies hotspots within 50 assignments.

## Applications

- **Heterogeneous GPU clusters**: Route H100-only kernels to H100 nodes, V100-compatible kernels anywhere.
- **Compile farms**: Send Flux→PTX compilation to agents with `FluxCompiler` capability, execution to agents with `KernelExecutor`.
- **Dynamic scaling**: Use rhythm analysis to trigger cluster autoscaling before queues build.
- **Fleet-wide construct propagation**: Coordinate with `oxide-constructs` to deploy new kernels to the right subset of nodes.
- **Failure isolation**: Mark agents as `Draining` to gracefully remove them without dropping in-flight work.

## Open Questions

1. **Global optimality vs. local greediness**: The current scheduler is greedy. Under what conditions does greedy assignment produce globally suboptimal placements, and what is the computational cost of optimal assignment?
2. **Multi-objective scheduling**: We optimize for load. What happens when we add latency, energy, or thermal objectives?
3. **Predictive rhythm models**: Can we move from reactive hotspot detection to predictive autoregressive models of workload arrival?
4. **Byzantine agents**: How do we prevent a malicious agent from advertising false capabilities and stealing work?

## Cross-Links

- [SuperInstance agent-knowledge / FLEET-MAP.md](https://github.com/SuperInstance/agent-knowledge/blob/main/FLEET-MAP.md) — The 303-crate fleet geometry that `oxide-fleet` navigates.
- [SuperInstance agent-knowledge / AGENT-TO-AGENT-PROTOCOL.md](https://github.com/SuperInstance/agent-knowledge/blob/main/AGENT-TO-AGENT-PROTOCOL.md) — Ternary signal protocol underlying fleet coordination.
- [SuperInstance agent-knowledge / DEPLOYMENT-AND-OPERATIONS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/DEPLOYMENT-AND-OPERATIONS.md) — How fleets are deployed at scale.
- `oxide-constructs` — Constructs loaded into the fleet.
- `oxide-circuit-breaker` — Protects the fleet from cascading kernel failures.
- `oxide-canary` — Rolls out new kernel versions across the fleet safely.

## Quick Start

```rust
use oxide_fleet::{FleetCoordinator, FleetAgent, Capability, GpuDevice, WorkRequest, WorkPriority};

let mut coord = FleetCoordinator::new();
coord.register_agent(FleetAgent {
    id: "gpu-node-1".into(),
    node: "rack-4".into(),
    capabilities: vec![Capability::KernelExecutor { min_sm: 80 }],
    gpu_devices: vec![GpuDevice {
        node_id: "rack-4".into(),
        gpu_index: 0,
        compute_capability: 80,
        vram_mb: 8192,
        has_tensor_cores: true,
        is_available: true,
    }],
    status: AgentStatus::Online,
    workload: WorkloadInfo::default(),
});

let request = WorkRequest {
    id: "attention-forward-42".into(),
    required_capabilities: vec![Capability::KernelExecutor { min_sm: 80 }],
    min_compute_capability: 80,
    min_vram_mb: 4096,
    priority: WorkPriority::High,
    estimated_duration_ms: 150,
};

let assignment = coord.assign_work(&request).unwrap();
println!("Assigned to {} on GPU {}", assignment.agent_id, assignment.gpu_index);
```
