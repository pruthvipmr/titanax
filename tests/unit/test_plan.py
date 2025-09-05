"""Unit tests for parallel plan components."""

import pytest
from src.titanax.parallel.plan import DP, TP, PP, Plan
from src.titanax.runtime.mesh import MeshSpec
from src.titanax.exceptions import PlanError


class TestDP:
    """Test Data Parallel plan."""
    
    def test_dp_creation(self):
        """Test basic DP creation."""
        dp = DP(axis="data")
        assert dp.axis == "data"
        assert dp.accumulate_steps == 1
        assert dp.sync_metrics is True
    
    def test_dp_with_microbatching(self):
        """Test DP with microbatch accumulation."""
        dp = DP(axis="data", accumulate_steps=4, sync_metrics=False)
        assert dp.accumulate_steps == 4
        assert dp.sync_metrics is False
    
    def test_dp_validation_empty_axis(self):
        """Test DP validation with empty axis."""
        with pytest.raises(PlanError, match="DP axis cannot be empty"):
            DP(axis="")
    
    def test_dp_validation_invalid_accumulate_steps(self):
        """Test DP validation with invalid accumulate_steps."""
        with pytest.raises(PlanError, match="accumulate_steps must be >= 1"):
            DP(axis="data", accumulate_steps=0)
    
    def test_dp_mesh_validation(self):
        """Test DP validation against mesh."""
        dp = DP(axis="data")
        mesh_spec = MeshSpec(axes=("data", "model"))
        dp.validate_with_mesh(mesh_spec)  # Should not raise
        
        mesh_spec_invalid = MeshSpec(axes=("model",))
        with pytest.raises(PlanError, match="DP axis 'data' not found"):
            dp.validate_with_mesh(mesh_spec_invalid)
    
    def test_dp_describe(self):
        """Test DP description generation."""
        dp = DP(axis="data")
        assert "Data Parallel on axis 'data'" in dp.describe()
        
        dp_micro = DP(axis="data", accumulate_steps=4)
        desc = dp_micro.describe()
        assert "microbatch accumulation" in desc
        
        dp_no_sync = DP(axis="data", sync_metrics=False)
        desc = dp_no_sync.describe()
        assert "metrics not synchronized" in desc


class TestTP:
    """Test Tensor Parallel plan."""
    
    def test_tp_creation(self):
        """Test basic TP creation."""
        rules = {"transformer/attn/qkv/kernel": ("model", None)}
        tp = TP(axis="model", rules=rules)
        assert tp.axis == "model"
        assert tp.rules == rules
        assert tp.prefer_reduce_scatter is True
    
    def test_tp_validation_empty_axis(self):
        """Test TP validation with empty axis."""
        with pytest.raises(PlanError, match="TP axis cannot be empty"):
            TP(axis="", rules={"param": ("model",)})
    
    def test_tp_validation_empty_rules(self):
        """Test TP validation with empty rules."""
        with pytest.raises(PlanError, match="TP rules cannot be empty"):
            TP(axis="model", rules={})
    
    def test_tp_validation_invalid_rule_key(self):
        """Test TP validation with invalid rule key."""
        with pytest.raises(PlanError, match="Rule key must be string"):
            TP(axis="model", rules={123: ("model",)})
    
    def test_tp_validation_invalid_rule_value(self):
        """Test TP validation with invalid rule value."""
        with pytest.raises(PlanError, match="Rule value must be tuple/list"):
            TP(axis="model", rules={"param": "invalid"})
    
    def test_tp_mesh_validation(self):
        """Test TP validation against mesh."""
        rules = {"transformer/attn/qkv/kernel": ("model", None)}
        tp = TP(axis="model", rules=rules)
        
        mesh_spec = MeshSpec(axes=("data", "model"))
        tp.validate_with_mesh(mesh_spec)  # Should not raise
        
        mesh_spec_invalid = MeshSpec(axes=("data",))
        with pytest.raises(PlanError, match="TP axis 'model' not found"):
            tp.validate_with_mesh(mesh_spec_invalid)
        
        # Test rule with unknown axis
        rules_invalid = {"param": ("unknown_axis", None)}
        tp_invalid = TP(axis="model", rules=rules_invalid)
        with pytest.raises(PlanError, match="references unknown axis 'unknown_axis'"):
            tp_invalid.validate_with_mesh(mesh_spec)
    
    def test_tp_describe(self):
        """Test TP description generation."""
        rules = {"transformer/attn/qkv/kernel": ("model", None)}
        tp = TP(axis="model", rules=rules)
        desc = tp.describe()
        assert "Tensor Parallel on axis 'model'" in desc
        assert "1 rules" in desc
        assert "prefer reduce_scatter" in desc


class TestPP:
    """Test Pipeline Parallel plan."""
    
    def test_pp_creation(self):
        """Test basic PP creation."""
        pp = PP(axis="pipe", stages=[], microbatch_size=4)
        assert pp.axis == "pipe"
        assert pp.stages == []
        assert pp.microbatch_size == 4
        assert pp.checkpoint_ratio == 0.0
    
    def test_pp_validation_empty_axis(self):
        """Test PP validation with empty axis."""
        with pytest.raises(PlanError, match="PP axis cannot be empty"):
            PP(axis="", stages=[], microbatch_size=4)
    
    def test_pp_validation_invalid_microbatch_size(self):
        """Test PP validation with invalid microbatch_size."""
        with pytest.raises(PlanError, match="microbatch_size must be >= 1"):
            PP(axis="pipe", stages=[], microbatch_size=0)
    
    def test_pp_validation_invalid_checkpoint_ratio(self):
        """Test PP validation with invalid checkpoint_ratio."""
        with pytest.raises(PlanError, match="checkpoint_ratio must be in"):
            PP(axis="pipe", stages=[], microbatch_size=4, checkpoint_ratio=1.5)
    
    def test_pp_describe(self):
        """Test PP description generation."""
        pp = PP(axis="pipe", stages=[], microbatch_size=4)
        desc = pp.describe()
        assert "Pipeline Parallel on axis 'pipe'" in desc
        assert "microbatch_size=4" in desc


class TestPlan:
    """Test composite Plan."""
    
    def test_plan_creation_dp_only(self):
        """Test plan with only DP."""
        dp = DP(axis="data")
        plan = Plan(data_parallel=dp)
        assert plan.data_parallel == dp
        assert plan.tensor_parallel is None
        assert plan.pipeline_parallel is None
    
    def test_plan_creation_dp_tp(self):
        """Test plan with DP×TP composition."""
        dp = DP(axis="data")
        tp = TP(axis="model", rules={"param": ("model", None)})
        plan = Plan(data_parallel=dp, tensor_parallel=tp)
        assert plan.data_parallel == dp
        assert plan.tensor_parallel == tp
    
    def test_plan_validation_empty(self):
        """Test plan validation with no strategies."""
        with pytest.raises(PlanError, match="must specify at least one"):
            Plan()
    
    def test_plan_validation_axis_conflict(self):
        """Test plan validation with axis conflicts."""
        dp = DP(axis="data")
        tp = TP(axis="data", rules={"param": ("data", None)})  # Same axis
        with pytest.raises(PlanError, match="Axis 'data' is used by multiple"):
            Plan(data_parallel=dp, tensor_parallel=tp)
    
    def test_plan_mesh_validation(self):
        """Test plan validation against mesh."""
        dp = DP(axis="data")
        tp = TP(axis="model", rules={"param": ("model", None)})
        plan = Plan(data_parallel=dp, tensor_parallel=tp)
        
        mesh_spec = MeshSpec(axes=("data", "model"))
        plan.validate(mesh_spec)  # Should not raise
        
        mesh_spec_invalid = MeshSpec(axes=("data",))  # Missing model axis
        with pytest.raises(PlanError, match="TP axis 'model' not found"):
            plan.validate(mesh_spec_invalid)
    
    def test_plan_describe(self):
        """Test plan description generation."""
        dp = DP(axis="data")
        plan = Plan(data_parallel=dp)
        assert "Data Parallel" in plan.describe()
        
        tp = TP(axis="model", rules={"param": ("model", None)})
        plan_composite = Plan(data_parallel=dp, tensor_parallel=tp)
        desc = plan_composite.describe()
        assert "Data Parallel" in desc
        assert "Tensor Parallel" in desc
        assert " × " in desc
    
    def test_plan_utility_methods(self):
        """Test plan utility methods."""
        dp = DP(axis="data")
        tp = TP(axis="model", rules={"param": ("model", None)})
        plan = Plan(data_parallel=dp, tensor_parallel=tp)
        
        assert plan.get_all_axes() == ("data", "model")
        assert not plan.is_data_parallel_only()
        
        dp_only_plan = Plan(data_parallel=dp)
        assert dp_only_plan.is_data_parallel_only()
        
        # Test microbatching detection
        assert not plan.has_microbatching()
        
        dp_micro = DP(axis="data", accumulate_steps=4)
        plan_micro = Plan(data_parallel=dp_micro)
        assert plan_micro.has_microbatching()
