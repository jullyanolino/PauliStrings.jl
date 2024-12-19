# QuantumEvolution.jl

A Julia package for simulating quantum time evolution using various numerical methods. 
This package provides efficient implementations of both real-time and imaginary-time evolution algorithms for quantum systems.

## Features

- Multiple time evolution methods:
  - Euler method (1st order)
  - Runge-Kutta 4 (4th order)
  - Crank-Nicolson (2nd order, implicit)
  - Trotter-Suzuki decomposition
  - Krylov subspace method

- Supports both:
  - Real-time evolution (Schrödinger equation)
  - Imaginary-time evolution (Ground state calculation)

- Comprehensive benchmarking tools
- Visualization utilities for performance analysis
- High-precision error tracking

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/QuantumEvolution.jl")
```

## Quick Start

```julia
using QuantumEvolution
using LinearAlgebra

# Define a simple system (e.g., single qubit)
H = Complex{Float64}.([0 1; 1 0])  # Pauli-X matrix
ψ₀ = Complex{Float64}.([1.0, 0.0]) # Initial state |0⟩

# Evolution parameters
dt = 0.1
steps = 10

# Real-time evolution using different methods
ψ = copy(ψ₀)
euler_step!(ψ, H, dt)          # Euler method
rk4_step!(ψ, H, dt)           # RK4 method
krylov_step!(ψ, H, dt)        # Krylov method
```

## Detailed Usage

### Real-Time Evolution

```julia
# Complete time evolution
function evolve_state(method, ψ₀, H, dt, steps)
    ψ = copy(ψ₀)
    for _ in 1:steps
        method(ψ, H, dt)
    end
    return ψ
end

# Compare different methods
ψ_euler = evolve_state(euler_step!, ψ₀, H, dt, steps)
ψ_rk4 = evolve_state(rk4_step!, ψ₀, H, dt, steps)
ψ_krylov = evolve_state(krylov_step!, ψ₀, H, dt, steps)
```

### Imaginary-Time Evolution

```julia
# Ground state search
function find_ground_state(method, ψ₀, H, dτ, steps)
    ψ = normalize(copy(ψ₀))
    for _ in 1:steps
        method(ψ, H, dτ)
        normalize!(ψ)
    end
    return ψ
end

# Use different methods
gs_euler = find_ground_state(euler_step_imag!, ψ₀, H, dt, steps)
gs_rk4 = find_ground_state(rk4_step_imag!, ψ₀, H, dt, steps)
```

## Benchmarking

The package includes comprehensive benchmarking tools:

```julia
include("benchmark_quantum_evolution.jl")

# Run complete benchmark suite
results = run_detailed_benchmark(H, ψ₀, dt, total_time)

# Visualize results
using Plots
plot_benchmark_results(results)
```

## Performance Comparison

Recent benchmarks show:

- Krylov method: Best accuracy (≈10⁻¹⁶ error), moderate speed
- RK4: Excellent accuracy (≈10⁻⁷ error), good speed
- Euler: Moderate accuracy (≈10⁻² error), fastest execution
- Trotter: Best for specific Hamiltonian structures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### TODO
- [ ] Add support for time-dependent Hamiltonians
- [ ] Implement adaptive time-stepping
- [ ] Add more visualization options
- [ ] Extend to larger quantum systems
- [ ] Add support for density matrix evolution

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{quantumevolution_jl,
  author = {Your Name},
  title = {QuantumEvolution.jl: A Julia Package for Quantum Time Evolution},
  year = {2024},
  url = {https://github.com/yourusername/QuantumEvolution.jl}
}
```

## Acknowledgments

- Built with Julia's quantum computing ecosystem
- Inspired by various quantum simulation techniques
- Performance optimizations based on modern numerical methods
