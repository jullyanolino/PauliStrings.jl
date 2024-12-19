using LinearAlgebra
using BenchmarkTools
#using Plots

include("QuantumEvolution.jl")
using .QuantumEvolution

# Define the Hamiltonian and initial state
H = Complex{Float64}.([0 1; 1 0])  # Example Hamiltonian
ψ0 = Complex{Float64}.([1.0, 0.0])  # Initial state

# Define the time step and total time
Δt = 0.1
total_time = 1.0
num_steps = Int(total_time / Δt)

# Exact solution using matrix exponentiation
exact_solution(t) = exp(-1im * H * t) * ψ0
exact_solution_imag(τ) = exp(-H * τ) * ψ0

# Function to perform time evolution using a given method
function evolve(method::Function, ψ::Vector{ComplexF64}, H::Matrix{ComplexF64}, Δt::Float64, num_steps::Int)
    ψ_evolved = copy(ψ)
    for _ in 1:num_steps
        method(ψ_evolved, H, Δt)
    end
    return ψ_evolved
end

# Benchmark the methods
methods = [QuantumEvolution.euler_step!, QuantumEvolution.rk4_step!,
          (ψ, H, Δt) -> QuantumEvolution.trotter_step!(ψ, H, H, Δt, 10), QuantumEvolution.krylov_step!]
method_names = ["Euler", "RK4", "Trotter-Suzuki", "Krylov"]

imag_methods = [QuantumEvolution.euler_step_imag!, QuantumEvolution.rk4_step_imag!,
               (ψ, H, Δτ) -> QuantumEvolution.trotter_step_imag!(ψ, H, H, Δτ, 10), QuantumEvolution.krylov_step_imag!]
imag_method_names = ["Euler Imag", "RK4 Imag", "Trotter-Suzuki Imag", "Krylov Imag"]

println("Benchmarking Quantum Evolution Methods:")
results = []
imag_results = []

for (method, name) in zip(methods, method_names)
    println("Benchmarking $name method...")
    ψ_final = evolve(method, ψ0, H, Δt, num_steps)
    exact_ψ_final = exact_solution(total_time)
    error = norm(ψ_final - exact_ψ_final)
    time = @elapsed evolve(method, ψ0, H, Δt, num_steps)
    push!(results, (name, error, time))
    println("Error: $error, Time: $time seconds")
end

for (method, name) in zip(imag_methods, imag_method_names)
    println("Benchmarking $name method...")
    ψ_final_imag = evolve(method, ψ0, H, Δt, num_steps)
    exact_ψ_final_imag = exact_solution_imag(total_time)
    error_imag = norm(ψ_final_imag - exact_ψ_final_imag)
    time_imag = @elapsed evolve(method, ψ0, H, Δt, num_steps)
    push!(imag_results, (name, error_imag, time_imag))
    println("Error: $error_imag, Time: $time_imag seconds")
end


# Visualization
real_errors = [result[2] for result in results]
real_times = [result[3] for result in results]
imag_errors = [result[2] for result in imag_results]
imag_times = [result[3] for result in imag_results]


"""
#This code snippet needs some visual improvements
plot(real_times, real_errors, xlabel="Time (s)", ylabel="Error", label="Real-Time Evolution", title="Real-Time Evolution Benchmark")
plot!(imag_times, imag_errors, xlabel="Time (s)", ylabel="Error", label="Imaginary-Time Evolution", title="Imaginary-Time Evolution Benchmark")
"""
