//! # oxide-fleet
//!
//! Fleet coordination layer for the Flux→PTX distributed GPU runtime.
//!
//! Coordinates GPU agents across multiple nodes using capability negotiation
//! (agent-handshake), declarative manifests (agent-manifest), and rhythm-based
//! workload optimization (agent-rhythm patterns).

use std::collections::HashMap;

/// Unique agent identifier.
pub type AgentId = String;

/// GPU device descriptor.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub node_id: String,
    pub gpu_index: u32,
    pub compute_capability: u32,
    pub vram_mb: u32,
    pub has_tensor_cores: bool,
    pub is_available: bool,
}

/// A capability that an agent can provide.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Can compile Flux to PTX.
    FluxCompiler,
    /// Can execute PTX kernels.
    KernelExecutor { min_sm: u32 },
    /// Can load constructs from git.
    ConstructLoader,
    /// Can synchronize state via CRDTs.
    CrdtSync,
    /// Custom capability.
    Custom(String),
}

/// An agent in the fleet.
#[derive(Debug, Clone)]
pub struct FleetAgent {
    pub id: AgentId,
    pub node: String,
    pub capabilities: Vec<Capability>,
    pub gpu_devices: Vec<GpuDevice>,
    pub status: AgentStatus,
    pub workload: WorkloadInfo,
}

/// Agent status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStatus {
    Online,
    Busy { task: String },
    Offline,
    Draining,
}

/// Current workload information.
#[derive(Debug, Clone, Default)]
pub struct WorkloadInfo {
    pub running_kernels: u32,
    pub pending_tasks: u32,
    pub gpu_utilization_pct: u8,
    pub memory_used_mb: u32,
}

/// A work request to distribute across the fleet.
#[derive(Debug, Clone)]
pub struct WorkRequest {
    pub id: String,
    pub required_capabilities: Vec<Capability>,
    pub min_compute_capability: u32,
    pub min_vram_mb: u32,
    pub priority: WorkPriority,
    pub estimated_duration_ms: u64,
}

/// Work priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Assignment decision.
#[derive(Debug, Clone)]
pub struct Assignment {
    pub work_id: String,
    pub agent_id: AgentId,
    pub gpu_index: u32,
    pub reason: String,
}

/// The fleet coordinator — manages agent discovery and workload distribution.
#[derive(Debug, Clone)]
pub struct FleetCoordinator {
    agents: HashMap<AgentId, FleetAgent>,
    assignments: HashMap<String, Assignment>,
    /// History for rhythm analysis.
    assignment_history: Vec<Assignment>,
}

impl FleetCoordinator {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            assignments: HashMap::new(),
            assignment_history: Vec::new(),
        }
    }

    /// Register a new agent in the fleet.
    pub fn register_agent(&mut self, agent: FleetAgent) {
        self.agents.insert(agent.id.clone(), agent);
    }

    /// Remove an agent from the fleet.
    pub fn deregister_agent(&mut self, id: &str) -> Option<FleetAgent> {
        self.agents.remove(id)
    }

    /// Discover agents that can satisfy given capabilities.
    pub fn discover(&self, required: &[Capability]) -> Vec<&FleetAgent> {
        self.agents.values()
            .filter(|a| {
                required.iter().all(|cap| {
                    a.capabilities.iter().any(|c| match (c, cap) {
                        (Capability::KernelExecutor { min_sm }, Capability::KernelExecutor { min_sm: req_sm }) => min_sm >= req_sm,
                        (a, b) => a == b,
                    })
                })
            })
            .collect()
    }

    /// Assign work to the best available agent.
    pub fn assign_work(&mut self, request: &WorkRequest) -> Result<Assignment, FleetError> {
        let candidates = self.discover(&request.required_capabilities);

        let best = candidates.into_iter()
            .filter(|a| a.status == AgentStatus::Online)
            .filter(|a| {
                a.gpu_devices.iter().any(|gpu| {
                    gpu.compute_capability >= request.min_compute_capability
                        && gpu.vram_mb >= request.min_vram_mb
                        && gpu.is_available
                })
            })
            .min_by_key(|a| {
                // Prefer agents with lower workload
                (a.workload.running_kernels, a.workload.gpu_utilization_pct)
            })
            .ok_or(FleetError::NoAvailableAgent)?;

        let gpu = best.gpu_devices.iter()
            .find(|g| g.compute_capability >= request.min_compute_capability && g.vram_mb >= request.min_vram_mb && g.is_available)
            .unwrap();

        let assignment = Assignment {
            work_id: request.id.clone(),
            agent_id: best.id.clone(),
            gpu_index: gpu.gpu_index,
            reason: format!("best fit: {} kernels running, {}% util",
                best.workload.running_kernels, best.workload.gpu_utilization_pct),
        };

        self.assignment_history.push(assignment.clone());
        self.assignments.insert(request.id.clone(), assignment.clone());
        Ok(assignment)
    }

    /// Complete a work assignment.
    pub fn complete_work(&mut self, work_id: &str) -> Option<Assignment> {
        self.assignments.remove(work_id)
    }

    /// Get fleet statistics.
    pub fn stats(&self) -> FleetStats {
        let total = self.agents.len();
        let online = self.agents.values().filter(|a| a.status == AgentStatus::Online).count();
        let total_gpus: usize = self.agents.values().map(|a| a.gpu_devices.len()).sum();
        let available_gpus: usize = self.agents.values()
            .flat_map(|a| a.gpu_devices.iter())
            .filter(|g| g.is_available)
            .count();
        let total_kernels: u32 = self.agents.values().map(|a| a.workload.running_kernels).sum();

        FleetStats {
            total_agents: total,
            online_agents: online,
            total_gpus,
            available_gpus,
            running_kernels: total_kernels,
            pending_assignments: self.assignments.len(),
        }
    }

    /// Analyze workload patterns (rhythm analysis).
    pub fn analyze_rhythm(&self) -> RhythmAnalysis {
        if self.assignment_history.is_empty() {
            return RhythmAnalysis::default();
        }

        let total = self.assignment_history.len();
        let mut agent_counts: HashMap<String, usize> = HashMap::new();
        for a in &self.assignment_history {
            *agent_counts.entry(a.agent_id.clone()).or_insert(0) += 1;
        }

        let max_agent = agent_counts.iter().max_by_key(|(_, c)| *c);
        let hotspots: Vec<String> = agent_counts.iter()
            .filter(|(_, &c)| {
                let avg = total as f64 / agent_counts.len() as f64;
                c as f64 > avg * 1.5
            })
            .map(|(id, _)| id.clone())
            .collect();

        RhythmAnalysis {
            total_assignments: total,
            unique_agents: agent_counts.len(),
            load_imbalance: if agent_counts.is_empty() { 0.0 } else {
                let max = *agent_counts.values().max().unwrap_or(&0) as f64;
                let avg = total as f64 / agent_counts.len() as f64;
                if avg == 0.0 { 0.0 } else { max / avg }
            },
            hotspots,
            recommendation: String::new(),
        }
    }
}

impl Default for FleetCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Fleet-wide statistics.
#[derive(Debug, Clone)]
pub struct FleetStats {
    pub total_agents: usize,
    pub online_agents: usize,
    pub total_gpus: usize,
    pub available_gpus: usize,
    pub running_kernels: u32,
    pub pending_assignments: usize,
}

/// Rhythm analysis results.
#[derive(Debug, Clone, Default)]
pub struct RhythmAnalysis {
    pub total_assignments: usize,
    pub unique_agents: usize,
    pub load_imbalance: f64,
    pub hotspots: Vec<String>,
    pub recommendation: String,
}

/// Fleet errors.
#[derive(Debug, Clone)]
pub enum FleetError {
    NoAvailableAgent,
    AgentNotFound(String),
    CapabilityMismatch { agent: String, required: String },
    GpuUnavailable { agent: String, gpu_index: u32 },
}

impl std::fmt::Display for FleetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAvailableAgent => write!(f, "no available agent satisfies requirements"),
            Self::AgentNotFound(id) => write!(f, "agent not found: {}", id),
            Self::CapabilityMismatch { agent, required } => {
                write!(f, "agent {} lacks capability: {}", agent, required)
            }
            Self::GpuUnavailable { agent, gpu_index } => {
                write!(f, "GPU {} on agent {} unavailable", gpu_index, agent)
            }
        }
    }
}

impl std::error::Error for FleetError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent(id: &str, sm: u32, vram: u32) -> FleetAgent {
        FleetAgent {
            id: id.to_string(),
            node: format!("node-{}", id),
            capabilities: vec![
                Capability::KernelExecutor { min_sm: sm },
                Capability::ConstructLoader,
            ],
            gpu_devices: vec![GpuDevice {
                node_id: format!("node-{}", id),
                gpu_index: 0,
                compute_capability: sm,
                vram_mb: vram,
                has_tensor_cores: sm >= 80,
                is_available: true,
            }],
            status: AgentStatus::Online,
            workload: WorkloadInfo::default(),
        }
    }

    #[test]
    fn test_register_discover() {
        let mut coord = FleetCoordinator::new();
        coord.register_agent(make_agent("a1", 80, 8192));
        coord.register_agent(make_agent("a2", 70, 4096));

        let found = coord.discover(&[Capability::KernelExecutor { min_sm: 80 }]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "a1");
    }

    #[test]
    fn test_assign_work() {
        let mut coord = FleetCoordinator::new();
        coord.register_agent(make_agent("a1", 80, 8192));

        let req = WorkRequest {
            id: "job-1".to_string(),
            required_capabilities: vec![Capability::KernelExecutor { min_sm: 80 }],
            min_compute_capability: 80,
            min_vram_mb: 4096,
            priority: WorkPriority::Normal,
            estimated_duration_ms: 100,
        };

        let assignment = coord.assign_work(&req).unwrap();
        assert_eq!(assignment.agent_id, "a1");
    }

    #[test]
    fn test_no_available_agent() {
        let mut coord = FleetCoordinator::new();
        coord.register_agent(make_agent("a1", 70, 2048));

        let req = WorkRequest {
            id: "job-1".to_string(),
            required_capabilities: vec![Capability::KernelExecutor { min_sm: 90 }],
            min_compute_capability: 90,
            min_vram_mb: 16384,
            priority: WorkPriority::High,
            estimated_duration_ms: 1000,
        };

        assert!(coord.assign_work(&req).is_err());
    }

    #[test]
    fn test_prefers_lower_workload() {
        let mut coord = FleetCoordinator::new();

        let mut busy = make_agent("busy", 80, 8192);
        busy.workload.running_kernels = 5;

        let mut idle = make_agent("idle", 80, 8192);

        coord.register_agent(busy);
        coord.register_agent(idle);

        let req = WorkRequest {
            id: "job-1".to_string(),
            required_capabilities: vec![Capability::KernelExecutor { min_sm: 80 }],
            min_compute_capability: 80,
            min_vram_mb: 1024,
            priority: WorkPriority::Normal,
            estimated_duration_ms: 100,
        };

        let assignment = coord.assign_work(&req).unwrap();
        assert_eq!(assignment.agent_id, "idle");
    }

    #[test]
    fn test_stats() {
        let mut coord = FleetCoordinator::new();
        coord.register_agent(make_agent("a1", 80, 8192));
        coord.register_agent(make_agent("a2", 80, 8192));

        let stats = coord.stats();
        assert_eq!(stats.total_agents, 2);
        assert_eq!(stats.online_agents, 2);
        assert_eq!(stats.total_gpus, 2);
        assert_eq!(stats.available_gpus, 2);
    }

    #[test]
    fn test_rhythm_analysis() {
        let mut coord = FleetCoordinator::new();
        coord.register_agent(make_agent("a1", 80, 8192));
        coord.register_agent(make_agent("a2", 80, 8192));

        for i in 0..10 {
            let req = WorkRequest {
                id: format!("job-{}", i),
                required_capabilities: vec![Capability::KernelExecutor { min_sm: 80 }],
                min_compute_capability: 80,
                min_vram_mb: 1024,
                priority: WorkPriority::Normal,
                estimated_duration_ms: 100,
            };
            coord.assign_work(&req).unwrap();
        }

        let rhythm = coord.analyze_rhythm();
        assert_eq!(rhythm.total_assignments, 10);
        assert!(rhythm.unique_agents >= 1);
    }

    #[test]
    fn test_complete_work() {
        let mut coord = FleetCoordinator::new();
        coord.register_agent(make_agent("a1", 80, 8192));

        let req = WorkRequest {
            id: "job-1".to_string(),
            required_capabilities: vec![Capability::KernelExecutor { min_sm: 80 }],
            min_compute_capability: 80,
            min_vram_mb: 1024,
            priority: WorkPriority::Normal,
            estimated_duration_ms: 100,
        };
        coord.assign_work(&req).unwrap();
        assert_eq!(coord.stats().pending_assignments, 1);

        let completed = coord.complete_work("job-1").unwrap();
        assert_eq!(completed.agent_id, "a1");
        assert_eq!(coord.stats().pending_assignments, 0);
    }
}
