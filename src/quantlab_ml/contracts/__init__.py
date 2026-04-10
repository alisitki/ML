from quantlab_ml.contracts.common import (
    InvalidActionMaskSemantics,
    LineagePointer,
    NumericBand,
    TimeRange,
)
from quantlab_ml.contracts.dataset import (
    DatasetSpec,
    NormalizedMarketEvent,
    REQUIRED_FIELDS_BY_STREAM,
    StreamFieldCatalog,
)
from quantlab_ml.contracts.evaluation import EvaluationBoundary, EvaluationReport, PolicyScore
from quantlab_ml.contracts.learning_surface import (
    ActionChoice,
    ActionFeasibilitySurface,
    ActionSpaceSpec,
    DerivedChannel,
    DerivedSurface,
    FeasibilityCell,
    ObservationContext,
    ObservationSchema,
    PolicyState,
    RawScaleTensor,
    ScaleSpec,
    TrajectoryBundle,
    TrajectoryRecord,
    TrajectorySpec,
    TrajectoryStep,
)
from quantlab_ml.contracts.policies import (
    ExecutorMetadata,
    ExecutorPolicyExport,
    OpaquePolicyPayload,
    PolicyArtifact,
)
from quantlab_ml.contracts.registry import CoverageStats, RegistryIndex, RegistryRecord, ScoreSnapshot
from quantlab_ml.contracts.rewards import (
    ActionReward,
    RewardContext,
    RewardEventSpec,
    RewardSnapshot,
    RewardTimeline,
    VenueExecutionRef,
)

# NOTE: contracts.compat intentionally excluded from public API.
# Import it directly from quantlab_ml.contracts.compat when needed.

__all__ = [
    # common
    "InvalidActionMaskSemantics",
    "LineagePointer",
    "NumericBand",
    "TimeRange",
    # dataset
    "DatasetSpec",
    "NormalizedMarketEvent",
    "REQUIRED_FIELDS_BY_STREAM",
    "StreamFieldCatalog",
    # evaluation
    "EvaluationBoundary",
    "EvaluationReport",
    "PolicyScore",
    # learning_surface
    "ActionChoice",
    "ActionFeasibilitySurface",
    "ActionSpaceSpec",
    "DerivedChannel",
    "DerivedSurface",
    "FeasibilityCell",
    "ObservationContext",
    "ObservationSchema",
    "PolicyState",
    "RawScaleTensor",
    "ScaleSpec",
    "TrajectoryBundle",
    "TrajectoryRecord",
    "TrajectorySpec",
    "TrajectoryStep",
    # policies
    "ExecutorMetadata",
    "ExecutorPolicyExport",
    "OpaquePolicyPayload",
    "PolicyArtifact",
    # registry
    "CoverageStats",
    "RegistryIndex",
    "RegistryRecord",
    "ScoreSnapshot",
    # rewards
    "ActionReward",
    "RewardContext",
    "RewardEventSpec",
    "RewardSnapshot",
    "RewardTimeline",
    "VenueExecutionRef",
]
