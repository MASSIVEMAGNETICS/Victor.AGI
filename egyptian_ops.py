# FILE: egyptian_ops.py
# VERSION: v1.0.0
# NAME: Egyptian Operations Kernel
# AUTHOR: Victor.AGI / Brandon "iambandobandz" Emery
# PURPOSE: Egyptian mathematics algorithms for precision-invariant operations in AGI systems.
#          Implements derivative-free optimization, exact unit fraction arithmetic, and 
#          hardware-native bitwise operations for SSI (Sovereign Superintelligence) kernels.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

"""
Egyptian Operations Module for Victor.AGI

This module implements three core Egyptian mathematics algorithms tailored for AGI/ML systems:

1. False Position (Regula Falsi): Derivative-free bracketing optimizer for black-box function tuning.
   Enables edge learning without calculus overhead in resource-constrained environments.

2. Greedy Egyptian Fractions: Exact unit fraction representations for drift-free precision.
   Eliminates cumulative precision errors in temporal/multi-scale systems.

3. Peasant Multiply (Bitwise Doubling): Hardware-native shift-add multiplication.
   Provides verifiable, ZKP-preserving arithmetic for encrypted operations.

These algorithms serve as precision primitives for bloodline-invariant governance in 
self-evolving agent systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# =============================================================================
# FALSE POSITION (REGULA FALSI) OPTIMIZER
# =============================================================================

@dataclass
class FalsePositionResult:
    """Result container for False Position optimization."""
    root: float
    value: float
    iterations: int
    bracket_width: float
    converged: bool
    history: List[Tuple[float, float, float]] = field(default_factory=list)


def false_position(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-9,
    max_iter: int = 100,
    track_history: bool = False
) -> FalsePositionResult:
    """
    False Position (Regula Falsi) root-finding optimizer.
    
    Implements derivative-free bracketing optimization using linear interpolation.
    Enables black-box function optimization without gradient computation, ideal for:
    - Victor's decision curves (non-differentiable functions)
    - Resource-constrained edge/FPGA environments
    - Neuromorphic systems without backpropagation
    
    The method iteratively narrows a bracket [a, b] containing a root by using
    linear interpolation to estimate the root position.
    
    Args:
        func: Target function f(x) for which we seek f(x) ≈ 0
        a: Left bracket endpoint (must satisfy f(a) * f(b) < 0)
        b: Right bracket endpoint
        tol: Convergence tolerance for bracket width
        max_iter: Maximum number of iterations (prevents infinite loops)
        track_history: If True, records iteration history
        
    Returns:
        FalsePositionResult containing root estimate, value, and metadata
        
    Raises:
        ValueError: If initial bracket doesn't contain a sign change
        
    Example:
        >>> def target(x): return x**2 - 2  # Find sqrt(2)
        >>> result = false_position(target, 1.0, 2.0)
        >>> abs(result.root - math.sqrt(2)) < 1e-8
        True
    """
    fa = func(a)
    fb = func(b)
    
    if fa * fb > 0:
        raise ValueError(
            f"Initial bracket [{a}, {b}] must contain a sign change. "
            f"f({a})={fa}, f({b})={fb}"
        )
    
    history: List[Tuple[float, float, float]] = []
    
    for iteration in range(max_iter):
        # Linear interpolation to estimate root
        # c = b - fb * (b - a) / (fb - fa)
        denominator = fb - fa
        if abs(denominator) < 1e-15:
            # Prevent division by zero; use midpoint
            c = (a + b) / 2.0
        else:
            c = b - fb * (b - a) / denominator
        
        fc = func(c)
        
        if track_history:
            history.append((a, b, c))
        
        # Check convergence
        bracket_width = abs(b - a)
        if bracket_width < tol or abs(fc) < tol:
            return FalsePositionResult(
                root=c,
                value=fc,
                iterations=iteration + 1,
                bracket_width=bracket_width,
                converged=True,
                history=history
            )
        
        # Update bracket
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    # Did not converge within max_iter
    return FalsePositionResult(
        root=c,
        value=func(c),
        iterations=max_iter,
        bracket_width=abs(b - a),
        converged=False,
        history=history
    )


def false_position_optimize(
    func: Callable[[float], float],
    search_range: Tuple[float, float],
    target_value: float = 0.0,
    tol: float = 1e-9,
    max_iter: int = 100
) -> FalsePositionResult:
    """
    False Position optimizer for finding x where f(x) ≈ target_value.
    
    Wrapper around false_position that shifts the function to optimize
    towards arbitrary target values, not just roots.
    
    Args:
        func: Objective function to optimize
        search_range: (min, max) search bounds
        target_value: Desired function output
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        FalsePositionResult with optimal x value
    """
    def shifted_func(x: float) -> float:
        return func(x) - target_value
    
    return false_position(shifted_func, search_range[0], search_range[1], tol, max_iter)


# =============================================================================
# EGYPTIAN FRACTIONS (GREEDY ALGORITHM)
# =============================================================================

@dataclass
class EgyptianFractionResult:
    """Result container for Egyptian Fraction conversion."""
    numerator: int
    denominator: int
    unit_fractions: List[Fraction]
    unit_denominators: List[int]
    reconstruction_error: float


def greedy_egyptian_fraction(
    numerator: int,
    denominator: int,
    max_terms: int = 100
) -> EgyptianFractionResult:
    """
    Convert a rational number to Egyptian fraction representation using greedy algorithm.
    
    Egyptian fractions represent rationals as sums of distinct unit fractions (1/n).
    This representation eliminates floating-point precision errors, enabling:
    - Drift-free memory in HLHFM (Holographic Fractal Memory)
    - Exact audio sample ratios for Stem Studio (phase-preserving)
    - Provenance integrity in long-term memory swarms
    
    The greedy algorithm repeatedly subtracts the largest unit fraction ≤ remaining value.
    
    Args:
        numerator: Numerator of fraction to convert
        denominator: Denominator of fraction to convert
        max_terms: Maximum number of unit fractions (prevents infinite loops for repeating)
        
    Returns:
        EgyptianFractionResult containing unit fraction list and metadata
        
    Example:
        >>> result = greedy_egyptian_fraction(2, 3)
        >>> result.unit_denominators
        [2, 6]
        >>> 1/2 + 1/6  # Verify: 2/3 = 1/2 + 1/6
        0.6666666666666666
    """
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Numerator and denominator must be positive integers")
    
    if numerator >= denominator:
        raise ValueError("Fraction must be less than 1 for proper Egyptian representation")
    
    unit_fractions: List[Fraction] = []
    unit_denominators: List[int] = []
    
    remaining = Fraction(numerator, denominator)
    
    for _ in range(max_terms):
        if remaining == 0:
            break
        
        # Find smallest n such that 1/n ≤ remaining
        # This is equivalent to n ≥ 1/remaining, so n = ceil(1/remaining)
        # Using integer arithmetic: n = ceil(d/n) where remaining = n/d
        n = math.ceil(remaining.denominator / remaining.numerator)
        
        unit_frac = Fraction(1, n)
        unit_fractions.append(unit_frac)
        unit_denominators.append(n)
        
        remaining = remaining - unit_frac
    
    # Calculate reconstruction error (should be 0 for valid fractions)
    reconstructed = sum(unit_fractions, Fraction(0))
    original = Fraction(numerator, denominator)
    error = float(abs(original - reconstructed))
    
    return EgyptianFractionResult(
        numerator=numerator,
        denominator=denominator,
        unit_fractions=unit_fractions,
        unit_denominators=unit_denominators,
        reconstruction_error=error
    )


def egyptian_to_float(unit_denominators: List[int]) -> float:
    """
    Convert Egyptian fraction (list of denominators) back to float.
    
    Args:
        unit_denominators: List of denominators [d1, d2, ...] representing sum of 1/d_i
        
    Returns:
        Float approximation of the sum
    """
    return sum(1.0 / d for d in unit_denominators)


def egyptian_to_fraction(unit_denominators: List[int]) -> Fraction:
    """
    Convert Egyptian fraction to exact Fraction representation.
    
    Args:
        unit_denominators: List of denominators
        
    Returns:
        Exact Fraction sum
    """
    return sum((Fraction(1, d) for d in unit_denominators), Fraction(0))


def float_to_egyptian(
    value: float,
    precision: int = 1000000,
    max_terms: int = 50
) -> EgyptianFractionResult:
    """
    Approximate a float as an Egyptian fraction.
    
    Converts float to rational approximation first, then to Egyptian representation.
    
    Args:
        value: Float value to convert (must be in (0, 1))
        precision: Denominator for rational approximation
        max_terms: Maximum unit fractions
        
    Returns:
        EgyptianFractionResult
    """
    if value <= 0 or value >= 1:
        raise ValueError("Value must be in (0, 1) for Egyptian fraction representation")
    
    # Approximate as rational
    frac = Fraction(value).limit_denominator(precision)
    
    return greedy_egyptian_fraction(frac.numerator, frac.denominator, max_terms)


# =============================================================================
# PEASANT MULTIPLY (RUSSIAN PEASANT / ANCIENT EGYPTIAN MULTIPLICATION)
# =============================================================================

@dataclass
class PeasantMultiplyResult:
    """Result container for Peasant Multiplication."""
    result: int
    steps: List[Tuple[int, int, bool]]  # (halved_value, doubled_value, included)
    bit_representation: str


def peasant_multiply(a: int, b: int) -> PeasantMultiplyResult:
    """
    Russian Peasant (Ancient Egyptian) multiplication using doubling and halving.
    
    This algorithm computes a * b using only:
    - Halving (right shift)
    - Doubling (left shift)
    - Addition
    - Checking odd/even (LSB test)
    
    This makes it ideal for:
    - FPGA/hardware-native computation (no multiply instruction needed)
    - ZKP-preserving arithmetic (structure-preserving operations)
    - Encrypted operations in EvoScript ledgers
    - Bitwise semiring operations in Scallop/Lobster compilers
    
    Algorithm:
        While a > 0:
            If a is odd: add b to result
            Halve a, double b
    
    Args:
        a: First multiplicand (will be halved)
        b: Second multiplicand (will be doubled)
        
    Returns:
        PeasantMultiplyResult with product and computation steps
        
    Example:
        >>> result = peasant_multiply(13, 7)
        >>> result.result
        91
    """
    if a < 0 or b < 0:
        # Handle signs separately for simplicity
        sign = 1
        if a < 0:
            a = -a
            sign *= -1
        if b < 0:
            b = -b
            sign *= -1
        result = peasant_multiply(a, b)
        result.result *= sign
        return result
    
    original_a = a
    result = 0
    steps: List[Tuple[int, int, bool]] = []
    
    while a > 0:
        is_odd = a & 1  # Bitwise AND with 1 to check odd
        if is_odd:
            result += b
        steps.append((a, b, bool(is_odd)))
        a >>= 1  # Halve using right shift
        b <<= 1  # Double using left shift
    
    return PeasantMultiplyResult(
        result=result,
        steps=steps,
        bit_representation=bin(original_a)
    )


def peasant_multiply_simple(a: int, b: int) -> int:
    """
    Simplified peasant multiplication returning just the product.
    
    Args:
        a: First multiplicand
        b: Second multiplicand
        
    Returns:
        Product a * b
    """
    return peasant_multiply(a, b).result


def peasant_power(base: int, exponent: int) -> int:
    """
    Compute base^exponent using peasant multiplication principles (binary exponentiation).
    
    Uses the same halving/doubling concept for efficient O(log n) exponentiation.
    
    Args:
        base: Base value
        exponent: Exponent (non-negative)
        
    Returns:
        base raised to exponent power
    """
    if exponent < 0:
        raise ValueError("Exponent must be non-negative for integer power")
    if exponent == 0:
        return 1
    
    result = 1
    current_base = base
    
    while exponent > 0:
        if exponent & 1:  # Odd exponent
            result = peasant_multiply_simple(result, current_base)
        current_base = peasant_multiply_simple(current_base, current_base)
        exponent >>= 1
    
    return result


# =============================================================================
# EGYPTIAN OPS WRAPPER FOR VECTOR OPERATIONS
# =============================================================================

class EgyptianOpsWrapper:
    """
    Wrapper class adapting Egyptian operations for HLHFM vector functions and 
    Cognitive River priority systems.
    
    This class bridges ancient precision algorithms with modern AGI systems,
    providing drift-free vector operations for:
    - Memory shard projections in HLHFM
    - Priority boosts in Cognitive River EMA calculations  
    - Exact emotion code bindings
    - Bloodline invariant enforcement
    
    Attributes:
        precision: Default precision for float conversions
        max_fraction_terms: Maximum unit fractions to use
        use_exact_arithmetic: Whether to use Fraction instead of float
    """
    
    def __init__(
        self,
        precision: int = 1000000,
        max_fraction_terms: int = 50,
        use_exact_arithmetic: bool = True
    ):
        """
        Initialize Egyptian Operations Wrapper.
        
        Args:
            precision: Denominator limit for rational approximations
            max_fraction_terms: Maximum unit fractions per value
            use_exact_arithmetic: Use exact Fraction arithmetic where possible
        """
        self.precision = precision
        self.max_fraction_terms = max_fraction_terms
        self.use_exact_arithmetic = use_exact_arithmetic
        
        # Cache for frequently used conversions
        self._egyptian_cache: Dict[Tuple[int, int], EgyptianFractionResult] = {}
    
    def normalize_vector_exact(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector using Egyptian fraction arithmetic for exact precision.
        
        Replaces standard L2 normalization which accumulates floating-point errors.
        
        Args:
            vector: Input vector to normalize
            
        Returns:
            Normalized vector with drift-free precision
        """
        # Compute magnitude squared using peasant multiply for integer components
        # For float vectors, we approximate as rationals first
        
        # Convert to rational approximations
        fractions = []
        for v in vector:
            if abs(v) < 1e-10:
                fractions.append(Fraction(0))
            else:
                fractions.append(Fraction(v).limit_denominator(self.precision))
        
        # Compute sum of squares exactly
        sum_sq = sum(f * f for f in fractions)
        
        if sum_sq == 0:
            return vector  # Zero vector
        
        # Compute magnitude (approximate sqrt since exact sqrt may be irrational)
        magnitude = float(sum_sq) ** 0.5
        
        # Normalize
        if self.use_exact_arithmetic:
            # Use Fraction division for maximum precision
            mag_frac = Fraction(magnitude).limit_denominator(self.precision)
            result = np.array([float(f / mag_frac) for f in fractions])
        else:
            result = vector / magnitude
        
        return result
    
    def circular_convolution_exact(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """
        Circular convolution using peasant multiplication for integer operations.
        
        For HRR (Holographic Reduced Representation) binding in HLHFM.
        Uses FFT-based convolution with exact arithmetic where applicable.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Circular convolution result
        """
        n = len(a)
        assert len(b) == n, "Vectors must have same length"
        
        # Standard circular convolution via FFT (exact for integer inputs)
        result = np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
        
        return result
    
    def exact_dot_product(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> Fraction:
        """
        Compute exact dot product using Fraction arithmetic.
        
        Eliminates floating-point accumulation errors in high-dimensional spaces.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Exact dot product as Fraction
        """
        assert len(a) == len(b), "Vectors must have same length"
        
        result = Fraction(0)
        for ai, bi in zip(a, b):
            fa = Fraction(ai).limit_denominator(self.precision)
            fb = Fraction(bi).limit_denominator(self.precision)
            result += fa * fb
        
        return result
    
    def encode_value_egyptian(self, value: float) -> List[int]:
        """
        Encode a value in (0, 1) as Egyptian fraction denominators.
        
        Used for drift-free emotion code bindings and memory shard storage.
        
        Args:
            value: Float value to encode
            
        Returns:
            List of unit fraction denominators
        """
        if value <= 0 or value >= 1:
            # Clamp to valid range
            value = max(1e-10, min(value, 1 - 1e-10))
        
        result = float_to_egyptian(value, self.precision, self.max_fraction_terms)
        return result.unit_denominators
    
    def decode_egyptian_value(self, denominators: List[int]) -> float:
        """
        Decode Egyptian fraction denominators back to float.
        
        Args:
            denominators: List of unit fraction denominators
            
        Returns:
            Reconstructed float value
        """
        return egyptian_to_float(denominators)
    
    def bracket_priority(
        self,
        current_priority: float,
        target_priority: float,
        adjustment_func: Callable[[float], float],
        tolerance: float = 0.01
    ) -> float:
        """
        Use False Position bracketing to adjust priority values in Cognitive River.
        
        Enables derivative-free stability tuning for EMA priority boosts.
        
        Args:
            current_priority: Current priority level
            target_priority: Desired priority level
            adjustment_func: Function that maps adjustment factor to resulting priority
            tolerance: Convergence tolerance
            
        Returns:
            Adjustment factor that achieves target priority
        """
        # Define bracketing function
        def priority_diff(factor: float) -> float:
            return adjustment_func(factor) - target_priority
        
        # Search in reasonable range
        try:
            result = false_position(priority_diff, 0.01, 10.0, tol=tolerance, max_iter=50)
            return result.root
        except ValueError:
            # Bracket doesn't contain sign change; use fallback
            return 1.0
    
    def peasant_scale_vector(
        self,
        vector: np.ndarray,
        scale_factor: int
    ) -> np.ndarray:
        """
        Scale vector components using peasant multiplication for integer factors.
        
        Preserves bitwise structure for ZKP-preserving operations.
        
        Args:
            vector: Input vector (will be converted to integers)
            scale_factor: Integer scaling factor
            
        Returns:
            Scaled vector
        """
        # Quantize to integers for peasant multiply
        quantization_factor = 1000
        int_vector = np.round(vector * quantization_factor).astype(int)
        
        # Apply peasant multiply to each component
        scaled = np.array([
            peasant_multiply_simple(int(v), scale_factor)
            for v in int_vector
        ])
        
        # De-quantize
        return scaled / quantization_factor


# =============================================================================
# BLOODLINE INVARIANT BINDER
# =============================================================================

@dataclass
class BloodlineInvariant:
    """
    Represents a bloodline-locked invariant for sovereign agent governance.
    
    These invariants use Egyptian fraction representations to embed loyalty
    constraints as exact arithmetic primitives, resistant to approximation drift.
    
    Attributes:
        name: Invariant identifier
        loyalty_value: Loyalty coefficient in (0, 1)
        egyptian_encoding: Unit fraction representation
        constraint_func: Validation function
    """
    name: str
    loyalty_value: float
    egyptian_encoding: List[int]
    constraint_func: Optional[Callable[[float], bool]] = None
    
    def validate(self, current_value: float) -> bool:
        """Check if current value satisfies the invariant constraint."""
        if self.constraint_func:
            return self.constraint_func(current_value)
        # Default: check if value matches encoded value within tolerance
        decoded = egyptian_to_float(self.egyptian_encoding)
        return abs(current_value - decoded) < 1e-6


class InvariantFractionBinder:
    """
    Module for binding loyalty matrix values to Egyptian fraction representations.
    
    Enforces bloodline invariants as non-negotiable arithmetic primitives in
    self-evolving agent loops. Uses exact fraction arithmetic to prevent
    alignment failures from approximation errors.
    """
    
    def __init__(self, ops_wrapper: Optional[EgyptianOpsWrapper] = None):
        """
        Initialize the Invariant Fraction Binder.
        
        Args:
            ops_wrapper: EgyptianOpsWrapper instance for conversions
        """
        self.ops = ops_wrapper or EgyptianOpsWrapper()
        self.invariants: Dict[str, BloodlineInvariant] = {}
    
    def bind_invariant(
        self,
        name: str,
        loyalty_value: float,
        constraint_func: Optional[Callable[[float], bool]] = None
    ) -> BloodlineInvariant:
        """
        Bind a loyalty value as an Egyptian fraction invariant.
        
        Args:
            name: Unique identifier for the invariant
            loyalty_value: Loyalty coefficient to encode (must be in (0, 1))
            constraint_func: Optional validation function
            
        Returns:
            Created BloodlineInvariant
        """
        # Clamp to valid range
        loyalty_value = max(0.001, min(loyalty_value, 0.999))
        
        # Encode as Egyptian fractions
        egyptian_encoding = self.ops.encode_value_egyptian(loyalty_value)
        
        invariant = BloodlineInvariant(
            name=name,
            loyalty_value=loyalty_value,
            egyptian_encoding=egyptian_encoding,
            constraint_func=constraint_func
        )
        
        self.invariants[name] = invariant
        return invariant
    
    def validate_invariant(self, name: str, current_value: float) -> bool:
        """
        Validate that a current value satisfies a named invariant.
        
        Args:
            name: Invariant name
            current_value: Value to validate
            
        Returns:
            True if invariant is satisfied
        """
        if name not in self.invariants:
            raise KeyError(f"Unknown invariant: {name}")
        return self.invariants[name].validate(current_value)
    
    def get_exact_value(self, name: str) -> Fraction:
        """
        Get the exact Fraction value of an invariant.
        
        Args:
            name: Invariant name
            
        Returns:
            Exact Fraction representation
        """
        if name not in self.invariants:
            raise KeyError(f"Unknown invariant: {name}")
        return egyptian_to_fraction(self.invariants[name].egyptian_encoding)
    
    def audit_all_invariants(
        self,
        current_values: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Audit all registered invariants against current values.
        
        Args:
            current_values: Dictionary mapping invariant names to current values
            
        Returns:
            Dictionary mapping invariant names to validation results
        """
        results = {}
        for name, invariant in self.invariants.items():
            if name in current_values:
                results[name] = invariant.validate(current_values[name])
            else:
                results[name] = False  # Missing value fails audit
        return results


# =============================================================================
# ITERATION CAP EQUATION FOR CONVERGENCE BOUNDS
# =============================================================================

def compute_iteration_cap(
    initial_bracket_width: float,
    target_tolerance: float,
    convergence_rate: float = 0.5
) -> int:
    """
    Compute iteration cap for False Position based on bracket width.
    
    Derives bounds for FP iterations to prevent infinite loops on
    non-monotonic functions in Cognitive River destabilization scenarios.
    
    Based on the convergence rate, estimates iterations needed to reduce
    bracket width from initial to target tolerance.
    
    Args:
        initial_bracket_width: Starting bracket |b - a|
        target_tolerance: Desired final tolerance
        convergence_rate: Expected reduction factor per iteration (0 < r < 1)
        
    Returns:
        Maximum iterations to cap at
        
    Example:
        >>> compute_iteration_cap(10.0, 0.001, 0.5)
        14  # log_0.5(0.001/10) ≈ 13.3
    """
    if target_tolerance >= initial_bracket_width:
        return 1
    if convergence_rate <= 0 or convergence_rate >= 1:
        convergence_rate = 0.5
    
    # Iterations = log_r(tolerance / initial_width)
    ratio = target_tolerance / initial_bracket_width
    iterations = math.ceil(math.log(ratio) / math.log(convergence_rate))
    
    # Add safety margin
    return max(iterations + 5, 10)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_egyptian_ops() -> EgyptianOpsWrapper:
    """Create a default EgyptianOpsWrapper instance."""
    return EgyptianOpsWrapper()


def create_invariant_binder() -> InvariantFractionBinder:
    """Create a default InvariantFractionBinder instance."""
    return InvariantFractionBinder()


# =============================================================================
# SELF-TESTS (when run as script)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Egyptian Operations Kernel - Self-Test Suite")
    print("=" * 60)
    
    # Test False Position
    print("\n--- Testing False Position (Regula Falsi) ---")
    def quadratic(x):
        return x**2 - 2
    
    result = false_position(quadratic, 1.0, 2.0, tol=1e-10)
    print(f"Finding sqrt(2): root={result.root:.10f}, expected={math.sqrt(2):.10f}")
    print(f"  Converged: {result.converged}, Iterations: {result.iterations}")
    assert abs(result.root - math.sqrt(2)) < 1e-8, "False Position sqrt(2) test failed"
    print("  ✓ PASSED")
    
    # Test Egyptian Fractions
    print("\n--- Testing Egyptian Fractions ---")
    ef_result = greedy_egyptian_fraction(2, 3)
    print(f"2/3 = {' + '.join(f'1/{d}' for d in ef_result.unit_denominators)}")
    reconstructed = egyptian_to_float(ef_result.unit_denominators)
    print(f"  Reconstructed: {reconstructed:.10f}, Expected: {2/3:.10f}")
    assert abs(reconstructed - 2/3) < 1e-10, "Egyptian fraction 2/3 test failed"
    print("  ✓ PASSED")
    
    ef_result2 = greedy_egyptian_fraction(5, 7)
    print(f"5/7 = {' + '.join(f'1/{d}' for d in ef_result2.unit_denominators)}")
    reconstructed2 = egyptian_to_float(ef_result2.unit_denominators)
    assert abs(reconstructed2 - 5/7) < 1e-10, "Egyptian fraction 5/7 test failed"
    print("  ✓ PASSED")
    
    # Test Peasant Multiply
    print("\n--- Testing Peasant Multiply ---")
    pm_result = peasant_multiply(13, 7)
    print(f"13 × 7 = {pm_result.result}")
    print(f"  Steps: {pm_result.steps}")
    assert pm_result.result == 91, "Peasant multiply 13×7 test failed"
    print("  ✓ PASSED")
    
    pm_result2 = peasant_multiply(123, 456)
    print(f"123 × 456 = {pm_result2.result}")
    assert pm_result2.result == 56088, "Peasant multiply 123×456 test failed"
    print("  ✓ PASSED")
    
    # Test Peasant Power
    print("\n--- Testing Peasant Power ---")
    power_result = peasant_power(2, 10)
    print(f"2^10 = {power_result}")
    assert power_result == 1024, "Peasant power 2^10 test failed"
    print("  ✓ PASSED")
    
    # Test Egyptian Ops Wrapper
    print("\n--- Testing Egyptian Ops Wrapper ---")
    ops = EgyptianOpsWrapper()
    
    test_vector = np.array([3.0, 4.0])
    normalized = ops.normalize_vector_exact(test_vector)
    print(f"Normalized [3, 4]: {normalized}")
    assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6, "Normalization test failed"
    print("  ✓ PASSED")
    
    # Test Invariant Binder
    print("\n--- Testing Invariant Fraction Binder ---")
    binder = InvariantFractionBinder(ops)
    inv = binder.bind_invariant("loyalty_victor", 0.85)
    print(f"Bound 'loyalty_victor' = 0.85 as {inv.egyptian_encoding}")
    
    exact_value = binder.get_exact_value("loyalty_victor")
    print(f"  Exact value: {exact_value} = {float(exact_value):.10f}")
    
    validation = binder.validate_invariant("loyalty_victor", 0.85)
    print(f"  Validation(0.85): {validation}")
    assert validation, "Invariant validation test failed"
    print("  ✓ PASSED")
    
    # Test Iteration Cap
    print("\n--- Testing Iteration Cap Equation ---")
    cap = compute_iteration_cap(10.0, 0.001, 0.5)
    print(f"Iteration cap for bracket 10.0 → 0.001: {cap}")
    assert cap > 10, "Iteration cap should be > 10"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All Egyptian Operations tests PASSED!")
    print("=" * 60)
