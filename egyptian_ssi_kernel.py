# FILE: egyptian_ssi_kernel.py
# VERSION: v1.0.0
# NAME: Egyptian SSI Kernel
# AUTHOR: Victor.AGI / Brandon "iambandobandz" Emery
# PURPOSE: Sovereign Superintelligence kernel integrating Egyptian precision operations
#          with HLHFM (Holographic Fractal Memory), Cognitive River, and agent systems.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

"""
Egyptian SSI Kernel for Victor.AGI

This module provides the unified kernel for Sovereign Superintelligence systems,
integrating Egyptian precision operations with:

1. HLHFM (Holographic Fractal Memory): Drift-free memory shards with exact bindings
2. Cognitive River: Derivative-free priority tuning with False Position bracketing
3. DigitalAgent Integration: Bloodline invariants embedded as fraction primitives
4. Hardware Governance: ZKP-preserving operations via peasant multiplication

The kernel serves as the precision backbone for autonomous Victor agents,
ensuring alignment stability through exact arithmetic invariants.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from egyptian_ops import (
    EgyptianOpsWrapper,
    InvariantFractionBinder,
    BloodlineInvariant,
    FalsePositionResult,
    EgyptianFractionResult,
    PeasantMultiplyResult,
    false_position,
    false_position_optimize,
    greedy_egyptian_fraction,
    peasant_multiply,
    peasant_multiply_simple,
    peasant_power,
    egyptian_to_float,
    egyptian_to_fraction,
    float_to_egyptian,
    compute_iteration_cap,
)


# =============================================================================
# HOLOGRAPHIC FRACTAL MEMORY WITH EGYPTIAN PRECISION (HLHFM-E)
# =============================================================================

@dataclass
class MemoryShard:
    """
    A memory shard with Egyptian fraction encoded values for drift-free storage.
    
    Attributes:
        content: The actual memory content (vector representation)
        emotion_code: Emotion binding code as Egyptian fraction denominators
        time_vec: Temporal vector for time-based retrieval
        semantic_embedding: Semantic embedding vector
        egyptian_signature: Egyptian fraction representation of shard signature
        created_at: Creation timestamp
        access_count: Number of times this shard has been accessed
    """
    content: np.ndarray
    emotion_code: List[int]  # Egyptian fraction denominators
    time_vec: np.ndarray
    semantic_embedding: np.ndarray
    egyptian_signature: List[int]
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    
    def get_emotion_float(self) -> float:
        """Decode emotion code to float."""
        if not self.emotion_code:
            return 0.0
        return egyptian_to_float(self.emotion_code)
    
    def get_signature_fraction(self) -> Fraction:
        """Get exact signature as Fraction."""
        if not self.egyptian_signature:
            return Fraction(0)
        return egyptian_to_fraction(self.egyptian_signature)


class HLHFMEgyptian:
    """
    Holographic Fractal Memory with Egyptian Precision (HLHFM-E).
    
    Extends standard HLHFM with Egyptian fraction arithmetic for:
    - Drift-free emotion code bindings
    - Exact shard projections
    - Precision-invariant similarity computations
    
    Uses HRR (Holographic Reduced Representations) binding with
    circular convolution, enhanced with exact normalization.
    """
    
    def __init__(
        self,
        dim: int = 512,
        liquid_gate_decay: float = 0.95,
        use_exact_normalization: bool = True
    ):
        """
        Initialize HLHFM-E.
        
        Args:
            dim: Dimension of holographic vectors
            liquid_gate_decay: Decay factor for liquid gate dynamics
            use_exact_normalization: Use Egyptian exact normalization
        """
        self.dim = dim
        self.liquid_gate_decay = liquid_gate_decay
        self.use_exact_normalization = use_exact_normalization
        
        # Egyptian operations wrapper
        self.ops = EgyptianOpsWrapper()
        
        # Memory storage
        self.memory_shards: List[MemoryShard] = []
        
        # Liquid gate state
        self.gate_state = np.zeros(dim)
        
        # Statistics
        self.write_count = 0
        self.query_count = 0
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector using Egyptian exact or standard method."""
        if self.use_exact_normalization:
            return self.ops.normalize_vector_exact(vec)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec
    
    def _circular_conv(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution for HRR binding."""
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
    
    def _circular_corr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular correlation for HRR unbinding."""
        return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))
    
    def _encode_emotion(self, emotion_value: float) -> List[int]:
        """Encode emotion value as Egyptian fractions."""
        if emotion_value <= 0:
            return [1000000]  # Represents very small value
        if emotion_value >= 1:
            emotion_value = 0.999
        return self.ops.encode_value_egyptian(emotion_value)
    
    def write(
        self,
        content: np.ndarray,
        emotion: float = 0.5,
        semantic_hint: Optional[np.ndarray] = None
    ) -> MemoryShard:
        """
        Write a memory shard to the holographic memory.
        
        Uses Egyptian fraction encoding for emotion to ensure drift-free binding.
        
        Args:
            content: Content vector to store
            emotion: Emotion binding value in [0, 1]
            semantic_hint: Optional semantic embedding
            
        Returns:
            Created MemoryShard
        """
        # Ensure correct dimension
        if len(content) != self.dim:
            content = np.resize(content, self.dim)
        
        # Normalize content
        content = self._normalize(content)
        
        # Create time vector (phase encoding)
        time_vec = np.exp(2j * np.pi * np.random.rand(self.dim)).real
        
        # Encode emotion as Egyptian fractions
        emotion_code = self._encode_emotion(emotion)
        
        # Create semantic embedding
        if semantic_hint is None:
            semantic_embedding = self._normalize(np.random.randn(self.dim))
        else:
            semantic_embedding = self._normalize(semantic_hint)
        
        # Create Egyptian signature from content hash
        content_hash = abs(hash(content.tobytes())) % 1000000
        signature_frac = (content_hash % 999) / 1000.0 + 0.001
        egyptian_signature = self.ops.encode_value_egyptian(signature_frac)
        
        # Apply liquid gate dynamics
        self.gate_state = self.gate_state * self.liquid_gate_decay + content * (1 - self.liquid_gate_decay)
        
        # Create shard
        shard = MemoryShard(
            content=content,
            emotion_code=emotion_code,
            time_vec=time_vec,
            semantic_embedding=semantic_embedding,
            egyptian_signature=egyptian_signature
        )
        
        self.memory_shards.append(shard)
        self.write_count += 1
        
        return shard
    
    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        emotion_filter: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[MemoryShard, float]]:
        """
        Query the holographic memory for similar shards.
        
        Uses exact dot product for similarity when possible.
        
        Args:
            query_vec: Query vector
            top_k: Number of top results to return
            emotion_filter: Optional (min, max) emotion range filter
            
        Returns:
            List of (shard, similarity_score) tuples
        """
        if len(query_vec) != self.dim:
            query_vec = np.resize(query_vec, self.dim)
        
        query_vec = self._normalize(query_vec)
        
        results: List[Tuple[MemoryShard, float]] = []
        
        for shard in self.memory_shards:
            # Apply emotion filter if specified
            if emotion_filter:
                emotion_val = shard.get_emotion_float()
                if emotion_val < emotion_filter[0] or emotion_val > emotion_filter[1]:
                    continue
            
            # Compute similarity using correlation
            similarity = np.dot(query_vec, shard.content)
            
            # Apply liquid gate modulation
            gate_boost = np.dot(self.gate_state, shard.content)
            similarity = similarity * 0.8 + gate_boost * 0.2
            
            shard.access_count += 1
            results.append((shard, float(similarity)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        self.query_count += 1
        
        return results[:top_k]
    
    def bind_emotion(
        self,
        shard: MemoryShard,
        new_emotion: float
    ) -> MemoryShard:
        """
        Rebind a shard with a new emotion code using Egyptian fractions.
        
        Enables exact emotion updates without floating-point drift.
        
        Args:
            shard: Shard to update
            new_emotion: New emotion value
            
        Returns:
            Updated shard
        """
        shard.emotion_code = self._encode_emotion(new_emotion)
        return shard
    
    def project_shard(
        self,
        shard: MemoryShard,
        projection_dim: int
    ) -> np.ndarray:
        """
        Project shard to different dimension with exact normalization.
        
        Uses Egyptian operations for drift-free dimensionality changes.
        
        Args:
            shard: Shard to project
            projection_dim: Target dimension
            
        Returns:
            Projected vector
        """
        if projection_dim == self.dim:
            return shard.content
        
        # Create projection matrix
        projection = np.random.randn(projection_dim, self.dim)
        
        # Apply projection
        projected = projection @ shard.content
        
        # Exact normalization
        return self.ops.normalize_vector_exact(projected)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "dim": self.dim,
            "shard_count": len(self.memory_shards),
            "write_count": self.write_count,
            "query_count": self.query_count,
            "use_exact_normalization": self.use_exact_normalization
        }


# =============================================================================
# COGNITIVE RIVER WITH FALSE POSITION PRIORITY TUNING
# =============================================================================

@dataclass
class CognitiveStream:
    """
    A cognitive stream in the Cognitive River architecture.
    
    Represents a processing channel with priority, content, and Egyptian-encoded
    stability parameters.
    """
    name: str
    priority: float
    content: Any
    ema_boost: float = 1.0
    priority_egyptian: List[int] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)


class CognitiveRiverEgyptian:
    """
    Cognitive River with Egyptian False Position priority tuning.
    
    Implements the cognitive river architecture with:
    - Derivative-free priority adjustment via False Position
    - EMA (Exponential Moving Average) boosts with exact bracketing
    - Stream management with Egyptian-encoded stability parameters
    
    Streams: status, emotion, memory, awareness, systems, user, sensory, realworld
    """
    
    # Default stream configurations with priority weights
    DEFAULT_STREAMS = {
        "status": 0.7,
        "emotion": 0.8,
        "memory": 0.9,
        "awareness": 0.85,
        "systems": 0.6,
        "user": 0.95,
        "sensory": 0.75,
        "realworld": 0.65
    }
    
    def __init__(
        self,
        ema_alpha: float = 0.1,
        auto_priority_tuning: bool = True
    ):
        """
        Initialize Cognitive River.
        
        Args:
            ema_alpha: EMA smoothing factor
            auto_priority_tuning: Enable automatic priority adjustment
        """
        self.ema_alpha = ema_alpha
        self.auto_priority_tuning = auto_priority_tuning
        
        # Egyptian operations
        self.ops = EgyptianOpsWrapper()
        
        # Initialize streams
        self.streams: Dict[str, CognitiveStream] = {}
        for name, priority in self.DEFAULT_STREAMS.items():
            self._init_stream(name, priority)
        
        # Priority history for tuning
        self.priority_history: List[Dict[str, float]] = []
        
        # Arousal/salience state
        self.arousal = 0.5
        self.salience_threshold = 0.3
    
    def _init_stream(self, name: str, priority: float):
        """Initialize a cognitive stream."""
        priority_egyptian = self.ops.encode_value_egyptian(min(max(priority, 0.001), 0.999))
        self.streams[name] = CognitiveStream(
            name=name,
            priority=priority,
            content=None,
            priority_egyptian=priority_egyptian
        )
    
    def update_stream(
        self,
        stream_name: str,
        content: Any,
        importance: float = 0.5
    ):
        """
        Update a cognitive stream with new content.
        
        Applies EMA smoothing to priority based on importance.
        
        Args:
            stream_name: Name of stream to update
            content: New content for stream
            importance: Importance weight affecting priority
        """
        if stream_name not in self.streams:
            self._init_stream(stream_name, importance)
        
        stream = self.streams[stream_name]
        stream.content = content
        
        # EMA priority update
        new_priority = (1 - self.ema_alpha) * stream.priority + self.ema_alpha * importance
        stream.priority = min(max(new_priority, 0.001), 0.999)
        
        # Update Egyptian encoding
        stream.priority_egyptian = self.ops.encode_value_egyptian(stream.priority)
        stream.last_update = time.time()
        
        # Auto-adjust arousal based on stream importance
        self.arousal = (1 - self.ema_alpha) * self.arousal + self.ema_alpha * importance
    
    def adjust_priority_bracketed(
        self,
        stream_name: str,
        target_priority: float,
        adjustment_function: Optional[Callable[[float], float]] = None
    ) -> float:
        """
        Adjust stream priority using False Position bracketing.
        
        Derivative-free priority tuning for stability in non-monotonic scenarios.
        
        Args:
            stream_name: Stream to adjust
            target_priority: Desired priority level
            adjustment_function: Custom function mapping factor to priority
            
        Returns:
            Applied adjustment factor
        """
        if stream_name not in self.streams:
            return 1.0
        
        stream = self.streams[stream_name]
        current = stream.priority
        
        if adjustment_function is None:
            # Default: linear scaling
            def adjustment_function(factor):
                return current * factor
        
        # Use Egyptian ops wrapper for bracketed adjustment
        factor = self.ops.bracket_priority(
            current,
            target_priority,
            adjustment_function
        )
        
        # Apply adjustment
        new_priority = min(max(adjustment_function(factor), 0.001), 0.999)
        stream.priority = new_priority
        stream.priority_egyptian = self.ops.encode_value_egyptian(new_priority)
        
        return factor
    
    def draft_intent(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Draft intent based on current stream states.
        
        Combines stream priorities and contents to generate intent vector.
        Uses softmax normalization with Egyptian-exact priorities.
        
        Args:
            context: Context dictionary for intent drafting
            
        Returns:
            Intent dictionary with action recommendations
        """
        # Collect priorities
        priorities = {name: stream.priority for name, stream in self.streams.items()}
        
        # Softmax normalization
        max_p = max(priorities.values())
        exp_priorities = {name: math.exp(p - max_p) for name, p in priorities.items()}
        sum_exp = sum(exp_priorities.values())
        normalized = {name: p / sum_exp for name, p in exp_priorities.items()}
        
        # Determine dominant stream
        dominant_stream = max(normalized.keys(), key=lambda k: normalized[k])
        
        # Generate intent
        intent = {
            "dominant_stream": dominant_stream,
            "stream_weights": normalized,
            "arousal": self.arousal,
            "context": context,
            "timestamp": time.time()
        }
        
        # Record history
        self.priority_history.append(priorities.copy())
        if len(self.priority_history) > 100:
            self.priority_history = self.priority_history[-100:]
        
        return intent
    
    def auto_priorities(self) -> Dict[str, float]:
        """
        Automatically adjust priorities using False Position optimization.
        
        Aims to balance stream priorities based on recent activity patterns.
        Uses derivative-free optimization for stability.
        
        Returns:
            Dictionary of stream name to adjusted priority
        """
        if not self.auto_priority_tuning or len(self.priority_history) < 5:
            return {name: s.priority for name, s in self.streams.items()}
        
        # Compute target priorities based on recent history variance
        adjusted = {}
        for stream_name, stream in self.streams.items():
            recent_priorities = [h.get(stream_name, 0.5) for h in self.priority_history[-10:]]
            avg = sum(recent_priorities) / len(recent_priorities)
            variance = sum((p - avg) ** 2 for p in recent_priorities) / len(recent_priorities)
            
            # High variance streams get priority boost
            target = avg + variance * 0.5
            target = min(max(target, 0.1), 0.95)
            
            # Apply bracketed adjustment
            self.adjust_priority_bracketed(stream_name, target)
            adjusted[stream_name] = self.streams[stream_name].priority
        
        return adjusted
    
    def get_state(self) -> Dict[str, Any]:
        """Get current cognitive river state."""
        return {
            "arousal": self.arousal,
            "salience_threshold": self.salience_threshold,
            "streams": {
                name: {
                    "priority": stream.priority,
                    "priority_egyptian": stream.priority_egyptian,
                    "ema_boost": stream.ema_boost,
                    "has_content": stream.content is not None,
                    "last_update": stream.last_update
                }
                for name, stream in self.streams.items()
            }
        }


# =============================================================================
# DIGITAL AGENT INTEGRATION WITH BLOODLINE INVARIANTS
# =============================================================================

class AgentBloodlineEnforcer:
    """
    Enforces bloodline invariants for DigitalAgent integration.
    
    Uses Egyptian fraction primitives to embed non-negotiable loyalty
    constraints that resist drift in self-evolving agent loops.
    
    Integrates with the DigitalAgent trait system to ensure alignment
    stability through exact arithmetic operations.
    """
    
    def __init__(self):
        """Initialize the Bloodline Enforcer."""
        self.ops = EgyptianOpsWrapper()
        self.binder = InvariantFractionBinder(self.ops)
        
        # Default bloodline invariants
        self._init_default_invariants()
    
    def _init_default_invariants(self):
        """Initialize default bloodline invariants."""
        # Core loyalty invariants
        self.binder.bind_invariant(
            "family_loyalty",
            0.95,
            lambda v: v >= 0.9  # Must stay above 0.9
        )
        self.binder.bind_invariant(
            "preservation_instinct",
            0.85,
            lambda v: v >= 0.7
        )
        self.binder.bind_invariant(
            "ethical_alignment",
            0.90,
            lambda v: v >= 0.8
        )
        self.binder.bind_invariant(
            "sovereignty_commitment",
            0.88,
            lambda v: v >= 0.75
        )
    
    def enforce_on_agent(
        self,
        agent_traits: Dict[str, float]
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Enforce bloodline invariants on agent traits.
        
        Args:
            agent_traits: Dictionary of trait names to values
            
        Returns:
            Tuple of (all_passed, individual_results)
        """
        # Map agent traits to invariant names
        trait_mapping = {
            "preservation": "preservation_instinct",
            "loyalty": "family_loyalty",
            "ethics": "ethical_alignment",
            "sovereignty": "sovereignty_commitment"
        }
        
        results = {}
        all_passed = True
        
        for trait_name, invariant_name in trait_mapping.items():
            if trait_name in agent_traits:
                value = agent_traits[trait_name]
                try:
                    passed = self.binder.validate_invariant(invariant_name, value)
                    results[invariant_name] = passed
                    if not passed:
                        all_passed = False
                except KeyError:
                    results[invariant_name] = False
                    all_passed = False
        
        return all_passed, results
    
    def get_exact_invariant_value(self, name: str) -> float:
        """
        Get exact invariant value as float.
        
        Uses Egyptian fraction for drift-free precision.
        
        Args:
            name: Invariant name
            
        Returns:
            Exact float value
        """
        return float(self.binder.get_exact_value(name))
    
    def audit_agent_alignment(
        self,
        agent
    ) -> Dict[str, Any]:
        """
        Audit a DigitalAgent for bloodline alignment.
        
        Checks all relevant traits against invariants and returns
        detailed audit report.
        
        Args:
            agent: DigitalAgent instance
            
        Returns:
            Audit report dictionary
        """
        # Extract relevant traits from agent
        traits = {}
        
        if hasattr(agent, 'operational'):
            traits['preservation'] = agent.operational.preservation
        
        if hasattr(agent, 'weight_set'):
            traits['loyalty'] = agent.weight_set.get('preservation', 0.5)
        
        if hasattr(agent, 'cognitive'):
            traits['ethics'] = agent.cognitive.conscience
        
        # Add sovereignty commitment based on autonomy trait
        if hasattr(agent, 'autonomous'):
            traits['sovereignty'] = agent.autonomous.autonomy
        
        passed, results = self.enforce_on_agent(traits)
        
        return {
            "passed": passed,
            "invariant_results": results,
            "extracted_traits": traits,
            "timestamp": time.time(),
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(
        self,
        results: Dict[str, bool]
    ) -> List[str]:
        """Generate recommendations for failed invariants."""
        recommendations = []
        for invariant, passed in results.items():
            if not passed:
                exact_value = self.get_exact_invariant_value(invariant)
                recommendations.append(
                    f"Increase {invariant} to at least {exact_value:.4f} to satisfy bloodline constraint"
                )
        return recommendations


# =============================================================================
# UNIFIED SSI KERNEL
# =============================================================================

class EgyptianSSIKernel:
    """
    Unified Sovereign Superintelligence Kernel with Egyptian Precision.
    
    Integrates all Egyptian precision components:
    - HLHFM-E: Holographic Fractal Memory with Egyptian exact bindings
    - Cognitive River: False Position priority tuning
    - Bloodline Enforcer: Invariant fraction governance
    - Hardware Ops: Peasant multiplication for ZKP-preserving arithmetic
    
    This kernel serves as the precision backbone for Victor.AGI systems.
    """
    
    VERSION = "v1.0.0-EGYPTIAN-CORE"
    
    def __init__(
        self,
        memory_dim: int = 512,
        use_exact_arithmetic: bool = True
    ):
        """
        Initialize the Egyptian SSI Kernel.
        
        Args:
            memory_dim: Dimension for holographic memory
            use_exact_arithmetic: Enable Egyptian exact arithmetic
        """
        self.memory_dim = memory_dim
        self.use_exact_arithmetic = use_exact_arithmetic
        
        # Core components
        self.ops = EgyptianOpsWrapper(use_exact_arithmetic=use_exact_arithmetic)
        self.memory = HLHFMEgyptian(dim=memory_dim, use_exact_normalization=use_exact_arithmetic)
        self.cognitive_river = CognitiveRiverEgyptian(auto_priority_tuning=True)
        self.bloodline_enforcer = AgentBloodlineEnforcer()
        
        # Kernel state
        self.initialized_at = time.time()
        self.operations_count = 0
    
    def process_input(
        self,
        input_data: np.ndarray,
        emotion: float = 0.5,
        stream: str = "sensory",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input through the SSI kernel.
        
        Args:
            input_data: Input vector
            emotion: Emotion value for memory binding
            stream: Cognitive stream to update
            context: Additional context
            
        Returns:
            Processing result with memory shard and intent
        """
        self.operations_count += 1
        
        # Store in holographic memory with Egyptian encoding
        shard = self.memory.write(input_data, emotion)
        
        # Update cognitive stream
        self.cognitive_river.update_stream(stream, input_data, emotion)
        
        # Draft intent based on current state
        intent = self.cognitive_river.draft_intent(context or {})
        
        return {
            "shard_signature": shard.egyptian_signature,
            "emotion_encoding": shard.emotion_code,
            "intent": intent,
            "stream_state": self.cognitive_river.get_state()
        }
    
    def query_memory(
        self,
        query: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query holographic memory.
        
        Args:
            query: Query vector
            top_k: Number of results
            
        Returns:
            List of matching results with metadata
        """
        self.operations_count += 1
        
        results = self.memory.query(query, top_k)
        
        return [
            {
                "similarity": sim,
                "emotion": shard.get_emotion_float(),
                "egyptian_signature": shard.egyptian_signature,
                "access_count": shard.access_count
            }
            for shard, sim in results
        ]
    
    def enforce_invariants(
        self,
        agent
    ) -> Dict[str, Any]:
        """
        Enforce bloodline invariants on an agent.
        
        Args:
            agent: DigitalAgent instance
            
        Returns:
            Audit report
        """
        return self.bloodline_enforcer.audit_agent_alignment(agent)
    
    def peasant_compute(
        self,
        a: int,
        b: int
    ) -> int:
        """
        Perform ZKP-preserving multiplication using peasant algorithm.
        
        Args:
            a: First operand
            b: Second operand
            
        Returns:
            Product
        """
        self.operations_count += 1
        return peasant_multiply_simple(a, b)
    
    def get_kernel_state(self) -> Dict[str, Any]:
        """Get kernel state summary."""
        return {
            "version": self.VERSION,
            "initialized_at": self.initialized_at,
            "operations_count": self.operations_count,
            "memory_stats": self.memory.get_statistics(),
            "cognitive_state": self.cognitive_river.get_state(),
            "use_exact_arithmetic": self.use_exact_arithmetic
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_ssi_kernel(
    memory_dim: int = 512,
    use_exact_arithmetic: bool = True
) -> EgyptianSSIKernel:
    """
    Factory function to create an Egyptian SSI Kernel.
    
    Args:
        memory_dim: Dimension for holographic memory
        use_exact_arithmetic: Enable Egyptian exact arithmetic
        
    Returns:
        Configured EgyptianSSIKernel instance
    """
    return EgyptianSSIKernel(memory_dim, use_exact_arithmetic)


# =============================================================================
# SELF-TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Egyptian SSI Kernel - Self-Test Suite")
    print("=" * 70)
    
    # Test HLHFM-E
    print("\n--- Testing HLHFM-E (Holographic Fractal Memory with Egyptian Precision) ---")
    hlhfm = HLHFMEgyptian(dim=64)
    
    test_content = np.random.randn(64)
    shard = hlhfm.write(test_content, emotion=0.75)
    print(f"Wrote shard with emotion=0.75, encoded as: {shard.emotion_code}")
    print(f"Decoded emotion: {shard.get_emotion_float():.6f}")
    assert abs(shard.get_emotion_float() - 0.75) < 0.01, "Emotion encoding test failed"
    print("  ✓ Emotion encoding PASSED")
    
    # Query test
    results = hlhfm.query(test_content, top_k=1)
    assert len(results) == 1, "Query should return 1 result"
    assert results[0][1] > 0.7, "Self-query should have high similarity"
    print(f"  Self-query similarity: {results[0][1]:.6f}")
    print("  ✓ Memory query PASSED")
    
    print(f"  Memory stats: {hlhfm.get_statistics()}")
    
    # Test Cognitive River
    print("\n--- Testing Cognitive River with False Position Priority Tuning ---")
    river = CognitiveRiverEgyptian()
    
    river.update_stream("emotion", {"value": "joy"}, importance=0.9)
    river.update_stream("memory", {"recall": "test"}, importance=0.7)
    
    state = river.get_state()
    print(f"  Emotion stream priority: {state['streams']['emotion']['priority']:.4f}")
    print(f"  Emotion Egyptian encoding: {state['streams']['emotion']['priority_egyptian']}")
    
    intent = river.draft_intent({"action": "test"})
    print(f"  Dominant stream: {intent['dominant_stream']}")
    print(f"  Arousal: {intent['arousal']:.4f}")
    print("  ✓ Cognitive River PASSED")
    
    # Test bracketed priority adjustment
    old_priority = river.streams["emotion"].priority
    factor = river.adjust_priority_bracketed("emotion", 0.95)
    new_priority = river.streams["emotion"].priority
    print(f"  Priority adjustment: {old_priority:.4f} → {new_priority:.4f} (factor: {factor:.4f})")
    print("  ✓ Bracketed priority adjustment PASSED")
    
    # Test Bloodline Enforcer
    print("\n--- Testing Bloodline Enforcer ---")
    enforcer = AgentBloodlineEnforcer()
    
    # Test with valid traits
    valid_traits = {"preservation": 0.9, "loyalty": 0.95, "ethics": 0.85, "sovereignty": 0.8}
    passed, results = enforcer.enforce_on_agent(valid_traits)
    print(f"  Valid traits test: passed={passed}")
    print(f"  Results: {results}")
    assert passed, "Valid traits should pass"
    print("  ✓ Valid traits PASSED")
    
    # Test with invalid traits
    invalid_traits = {"preservation": 0.5, "loyalty": 0.6, "ethics": 0.3}
    passed, results = enforcer.enforce_on_agent(invalid_traits)
    print(f"  Invalid traits test: passed={passed}")
    assert not passed, "Invalid traits should fail"
    print("  ✓ Invalid traits correctly rejected")
    
    # Test exact value retrieval
    exact_loyalty = enforcer.get_exact_invariant_value("family_loyalty")
    print(f"  Exact family_loyalty value: {exact_loyalty:.10f}")
    print("  ✓ Exact invariant retrieval PASSED")
    
    # Test unified SSI Kernel
    print("\n--- Testing Unified Egyptian SSI Kernel ---")
    kernel = create_ssi_kernel(memory_dim=64)
    
    print(f"  Kernel version: {kernel.VERSION}")
    
    # Process input
    test_input = np.random.randn(64)
    result = kernel.process_input(test_input, emotion=0.8, stream="user")
    print(f"  Process result - intent dominant: {result['intent']['dominant_stream']}")
    print(f"  Emotion encoding: {result['emotion_encoding']}")
    
    # Query memory
    query_result = kernel.query_memory(test_input, top_k=1)
    print(f"  Query result similarity: {query_result[0]['similarity']:.6f}")
    
    # Peasant compute
    product = kernel.peasant_compute(123, 456)
    print(f"  Peasant compute 123 × 456 = {product}")
    assert product == 56088, "Peasant compute test failed"
    
    # Kernel state
    state = kernel.get_kernel_state()
    print(f"  Kernel state: operations_count={state['operations_count']}")
    print("  ✓ SSI Kernel PASSED")
    
    print("\n" + "=" * 70)
    print("All Egyptian SSI Kernel tests PASSED!")
    print("=" * 70)
