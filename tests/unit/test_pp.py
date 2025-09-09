"""Unit tests for pipeline parallel components."""

import pytest
from src.titanax.parallel.pp import (
    Stage,
    StageProtocol,
    PipelineSchedule,
    create_simple_stage,
    create_1f1b_schedule,
)
from src.titanax.parallel.plan import PP, Plan
from src.titanax.runtime.mesh import MeshSpec
from src.titanax.exceptions import PlanError


def dummy_forward(inputs, training=True):
    """Dummy forward function for testing."""
    return inputs, inputs  # Return same as outputs and saved activations


def dummy_backward(grad_outputs, activations):
    """Dummy backward function for testing."""
    return grad_outputs  # Just pass through


class TestStage:
    """Test Stage dataclass and validation."""

    def test_stage_creation(self):
        """Test basic stage creation."""
        stage = Stage(forward_fn=dummy_forward, stage_id=0)

        assert stage.forward_fn == dummy_forward
        assert stage.backward_fn is None
        assert stage.stage_id == 0
        assert stage.remat_policy == "none"
        assert stage.stage_name is None

    def test_stage_with_options(self):
        """Test stage creation with all options."""
        stage = Stage(
            forward_fn=dummy_forward,
            backward_fn=dummy_backward,
            stage_id=2,
            remat_policy="full",
            stage_name="encoder",
        )

        assert stage.backward_fn == dummy_backward
        assert stage.stage_id == 2
        assert stage.remat_policy == "full"
        assert stage.stage_name == "encoder"

    def test_stage_validation_invalid_forward_fn(self):
        """Test stage validation with invalid forward function."""
        with pytest.raises(PlanError, match="forward_fn must be callable"):
            Stage(forward_fn="not_callable", stage_id=0)

    def test_stage_validation_invalid_backward_fn(self):
        """Test stage validation with invalid backward function."""
        with pytest.raises(PlanError, match="backward_fn must be callable or None"):
            Stage(forward_fn=dummy_forward, backward_fn="not_callable", stage_id=0)

    def test_stage_validation_negative_stage_id(self):
        """Test stage validation with negative stage ID."""
        with pytest.raises(PlanError, match="stage_id must be non-negative"):
            Stage(forward_fn=dummy_forward, stage_id=-1)

    def test_stage_validation_invalid_remat_policy(self):
        """Test stage validation with invalid remat policy."""
        with pytest.raises(PlanError, match="Invalid remat_policy"):
            Stage(forward_fn=dummy_forward, stage_id=0, remat_policy="invalid")

    def test_stage_forward(self):
        """Test stage forward pass."""
        stage = Stage(forward_fn=dummy_forward, stage_id=0)
        inputs = {"x": [1, 2, 3]}

        outputs, activations = stage.forward(inputs, training=True)
        assert outputs == inputs
        assert activations == inputs

    def test_stage_backward_custom(self):
        """Test stage backward pass with custom function."""
        stage = Stage(forward_fn=dummy_forward, backward_fn=dummy_backward, stage_id=0)

        grad_outputs = {"grad": [1, 2, 3]}
        activations = {"x": [4, 5, 6]}

        grad_inputs = stage.backward(grad_outputs, activations)
        assert grad_inputs == grad_outputs

    def test_stage_backward_default_not_implemented(self):
        """Test stage backward pass without custom function raises NotImplementedError."""
        stage = Stage(forward_fn=dummy_forward, stage_id=0)

        with pytest.raises(NotImplementedError, match="Default JAX autodiff backward"):
            stage.backward({"grad": [1, 2, 3]}, {"x": [4, 5, 6]})

    def test_stage_describe(self):
        """Test stage description generation."""
        stage = Stage(forward_fn=dummy_forward, stage_id=1)
        desc = stage.describe()
        assert "Stage1" in desc
        assert "(ID: 1)" in desc

        named_stage = Stage(
            forward_fn=dummy_forward,
            stage_id=2,
            stage_name="decoder",
            remat_policy="selective",
        )
        desc = named_stage.describe()
        assert "decoder" in desc
        assert "(ID: 2)" in desc
        assert "remat=selective" in desc


class TestPipelineSchedule:
    """Test PipelineSchedule configuration and validation."""

    def test_schedule_creation_default(self):
        """Test default schedule creation."""
        schedule = PipelineSchedule()

        assert schedule.strategy == "1F1B"
        assert schedule.num_microbatches == 4
        assert schedule.warmup_steps is None
        assert schedule.cooldown_steps is None

    def test_schedule_creation_custom(self):
        """Test custom schedule creation."""
        schedule = PipelineSchedule(
            strategy="interleaved", num_microbatches=8, warmup_steps=2, cooldown_steps=2
        )

        assert schedule.strategy == "interleaved"
        assert schedule.num_microbatches == 8
        assert schedule.warmup_steps == 2
        assert schedule.cooldown_steps == 2

    def test_schedule_validation_invalid_strategy(self):
        """Test schedule validation with invalid strategy."""
        with pytest.raises(PlanError, match="Invalid strategy"):
            PipelineSchedule(strategy="invalid")

    def test_schedule_validation_invalid_num_microbatches(self):
        """Test schedule validation with invalid num_microbatches."""
        with pytest.raises(PlanError, match="num_microbatches must be >= 1"):
            PipelineSchedule(num_microbatches=0)

    def test_schedule_validation_negative_warmup(self):
        """Test schedule validation with negative warmup steps."""
        with pytest.raises(PlanError, match="warmup_steps must be non-negative"):
            PipelineSchedule(warmup_steps=-1)

    def test_schedule_validation_negative_cooldown(self):
        """Test schedule validation with negative cooldown steps."""
        with pytest.raises(PlanError, match="cooldown_steps must be non-negative"):
            PipelineSchedule(cooldown_steps=-1)

    def test_schedule_validate_with_pipeline(self):
        """Test schedule validation against pipeline configuration."""
        stages = [
            Stage(forward_fn=dummy_forward, stage_id=0),
            Stage(forward_fn=dummy_forward, stage_id=1),
        ]
        schedule = PipelineSchedule(num_microbatches=4)

        # Should pass validation
        schedule.validate_with_pipeline(stages, microbatch_size=32)

    def test_schedule_validate_empty_pipeline(self):
        """Test schedule validation with empty pipeline."""
        schedule = PipelineSchedule()

        with pytest.raises(PlanError, match="Pipeline must have at least one stage"):
            schedule.validate_with_pipeline([], microbatch_size=32)

    def test_schedule_validate_insufficient_microbatches(self):
        """Test schedule validation with too few microbatches."""
        stages = [
            Stage(forward_fn=dummy_forward, stage_id=0),
            Stage(forward_fn=dummy_forward, stage_id=1),
            Stage(forward_fn=dummy_forward, stage_id=2),
        ]
        schedule = PipelineSchedule(num_microbatches=2)  # Less than num_stages

        with pytest.raises(PlanError, match="should be >= num_stages"):
            schedule.validate_with_pipeline(stages, microbatch_size=32)

    def test_schedule_validate_invalid_stage_ids(self):
        """Test schedule validation with non-consecutive stage IDs."""
        stages = [
            Stage(forward_fn=dummy_forward, stage_id=0),
            Stage(forward_fn=dummy_forward, stage_id=2),  # Skip stage_id=1
        ]
        schedule = PipelineSchedule(num_microbatches=4)

        with pytest.raises(PlanError, match="Stage IDs .* should be consecutive"):
            schedule.validate_with_pipeline(stages, microbatch_size=32)

    def test_schedule_describe(self):
        """Test schedule description generation."""
        schedule = PipelineSchedule(strategy="1F1B", num_microbatches=6)
        desc = schedule.describe()
        assert "1F1B schedule" in desc
        assert "6 microbatches" in desc

        schedule_with_steps = PipelineSchedule(
            strategy="interleaved", num_microbatches=8, warmup_steps=3, cooldown_steps=3
        )
        desc = schedule_with_steps.describe()
        assert "interleaved schedule" in desc
        assert "warmup=3" in desc
        assert "cooldown=3" in desc


class TestPPIntegration:
    """Test PP integration with Plan and validation."""

    def test_pp_plan_with_stages(self):
        """Test PP plan creation with Stage objects."""
        stages = [
            Stage(forward_fn=dummy_forward, stage_id=0, stage_name="encoder"),
            Stage(forward_fn=dummy_forward, stage_id=1, stage_name="decoder"),
        ]

        pp = PP(axis="pipe", stages=stages, microbatch_size=16)
        assert pp.axis == "pipe"
        assert len(pp.stages) == 2
        assert pp.microbatch_size == 16

    def test_pp_plan_validation_with_mesh(self):
        """Test PP plan validation against mesh."""
        stages = [Stage(forward_fn=dummy_forward, stage_id=0)]
        pp = PP(axis="pipe", stages=stages, microbatch_size=8)

        mesh_spec = MeshSpec(axes=("pipe",))
        pp.validate_with_mesh(mesh_spec)  # Should pass

        mesh_spec_invalid = MeshSpec(axes=("data",))
        with pytest.raises(PlanError, match="PP axis 'pipe' not found"):
            pp.validate_with_mesh(mesh_spec_invalid)

    def test_plan_with_pp_stages_validation(self):
        """Test Plan validation with PP stages and microbatch validation."""
        # Create stages with proper stage IDs
        stages = [
            Stage(forward_fn=dummy_forward, stage_id=0),
            Stage(forward_fn=dummy_forward, stage_id=1),
        ]

        pp = PP(axis="pipe", stages=stages, microbatch_size=4)
        Plan(pipeline_parallel=pp)  # Should create successfully

        # Create schedule and validate compatibility
        schedule = PipelineSchedule(num_microbatches=8)  # >= num_stages
        schedule.validate_with_pipeline(stages, microbatch_size=4)  # Should pass


class TestHelperFunctions:
    """Test helper functions for creating stages and schedules."""

    def test_create_simple_stage(self):
        """Test simple stage creation helper."""
        stage = create_simple_stage(
            forward_fn=dummy_forward,
            stage_id=3,
            stage_name="attention",
            remat_policy="selective",
        )

        assert stage.forward_fn == dummy_forward
        assert stage.stage_id == 3
        assert stage.stage_name == "attention"
        assert stage.remat_policy == "selective"
        assert stage.backward_fn is None

    def test_create_1f1b_schedule(self):
        """Test 1F1B schedule creation helper."""
        schedule = create_1f1b_schedule(
            num_stages=4, microbatch_size=8, global_batch_size=64
        )

        assert schedule.strategy == "1F1B"
        assert schedule.num_microbatches == 8  # 64 / 8
        assert schedule.warmup_steps == 3  # num_stages - 1
        assert schedule.cooldown_steps == 3  # num_stages - 1

    def test_create_1f1b_schedule_invalid_batch_size(self):
        """Test 1F1B schedule creation with invalid batch size."""
        with pytest.raises(PlanError, match="global_batch_size .* must be divisible"):
            create_1f1b_schedule(
                num_stages=2,
                microbatch_size=7,  # 64 is not divisible by 7
                global_batch_size=64,
            )


class TestStageProtocol:
    """Test StageProtocol compliance."""

    def test_stage_implements_protocol(self):
        """Test that Stage class implements StageProtocol."""
        stage = Stage(forward_fn=dummy_forward, stage_id=0)

        # Check that Stage is recognized as implementing the protocol
        assert isinstance(stage, StageProtocol)

        # Check that it has the required methods
        assert hasattr(stage, "forward")
        assert hasattr(stage, "backward")
        assert callable(stage.forward)
        assert callable(stage.backward)
