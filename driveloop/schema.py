from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DriveLoopConfig:
    max_iterations: int = 3
    target_score: float = 0.8
    output_dir: Path = Path("outputs/driveloop")
    keep_all_generations: bool = True
    use_task_utility: bool = False
    utility_weights: Optional[Dict[str, float]] = None
    intent_epsilon: float = 0.05
    use_postprocess: bool = False
    acceptance_policy: str = "score_and_diagnosis"
    perception_acceptance_threshold: float = 0.3
    stop_on_acceptance: bool = True


@dataclass(frozen=True)
class DriveLoopRequest:
    prompt: str
    scenario_id: Optional[str] = None
    condition: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneObject:
    category: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneSpecification:
    prompt: str
    objects: List[SceneObject] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: List[str] = field(default_factory=list)
    motion_primitives: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LongTailConditionPlan:
    tags: List[str] = field(default_factory=list)
    prompt_suffixes: List[str] = field(default_factory=list)
    postprocess_effects: List[str] = field(default_factory=list)
    executable_controls: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Generation:
    iteration: int
    prompt: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Diagnosis:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Evaluation:
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    diagnosis: Diagnosis = field(default_factory=lambda: Diagnosis(passed=True))


@dataclass(frozen=True)
class Refinement:
    prompt: str
    condition: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DriveLoopAttempt:
    iteration: int
    request: DriveLoopRequest
    scene_specification: SceneSpecification
    long_tail_condition_plan: LongTailConditionPlan
    dd2_condition: Dict[str, Any]
    condition_package: Dict[str, Any]
    source_binding: Dict[str, Any]
    generation: Generation
    evaluation: Evaluation
    refinement: Optional[Refinement]
    status: str
    source_selection: Dict[str, Any] = field(default_factory=dict)
    claim_boundary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriveLoopResult:
    request: DriveLoopRequest
    best_generation: Generation
    best_evaluation: Evaluation
    history: List[Tuple[Generation, Evaluation]]
    attempt_history: List[DriveLoopAttempt] = field(default_factory=list)
    # best_* is the unconstrained rank maximum; returned_* obeys acceptance.
    returned_generation: Optional[Generation] = None
    returned_evaluation: Optional[Evaluation] = None
    return_status: str = "unknown"
    selection_metric: str = "evaluation_score"
    acceptance_policy: str = "unknown"
