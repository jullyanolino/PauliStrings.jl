using LinearAlgebra
using BenchmarkTools
#using QuantumEvolution
using Plots
using StatsPlots  # For additional plotting features
using Printf

include("QuantumEvolution.jl")
using .QuantumEvolution

"""
Structure to hold benchmark results
"""
struct BenchmarkResult
    method_name::String
    error::Float64
    execution_time::Float64
    energy_evolution::Vector{Float64}
    state_fidelity::Vector{Float64}
end

"""
Run benchmarks and collect detailed results
"""
function run_detailed_benchmark(H, ψ0, Δt, total_time)
    num_steps = Int(total_time / Δt)
    times = 0:Δt:total_time
    
    # Methods configuration
    methods = [
        (QuantumEvolution.euler_step!, "Euler"),
        (QuantumEvolution.rk4_step!, "RK4"),
        ((ψ, H, Δt) -> QuantumEvolution.trotter_step!(ψ, H, H, Δt, 10), "Trotter"),
        (QuantumEvolution.krylov_step!, "Krylov")
    ]
    
    imag_methods = [
        (QuantumEvolution.euler_step_imag!, "Euler Imag"),
        (QuantumEvolution.rk4_step_imag!, "RK4 Imag"),
        ((ψ, H, Δt) -> QuantumEvolution.trotter_step_imag!(ψ, H, H, Δt, 10), "Trotter Imag"),
        (QuantumEvolution.krylov_step_imag!, "Krylov Imag")
    ]
    
    real_results = Vector{BenchmarkResult}()
    imag_results = Vector{BenchmarkResult}()
    
    # Benchmark real-time methods
    for (method, name) in methods
        ψ = copy(ψ0)
        energy_evolution = Float64[]
        fidelity_evolution = Float64[]
        
        execution_time = @elapsed begin
            for t in times
                exact_ψ = exact_solution(t)
                push!(energy_evolution, real(dot(ψ, H * ψ)))
                push!(fidelity_evolution, abs2(dot(exact_ψ, ψ)))
                method(ψ, H, Δt)
            end
        end
        
        final_error = norm(ψ - exact_solution(total_time))
        push!(real_results, BenchmarkResult(name, final_error, execution_time,
                                          energy_evolution, fidelity_evolution))
    end
    
    # Benchmark imaginary-time methods
    for (method, name) in imag_methods
        ψ = copy(ψ0)
        energy_evolution = Float64[]
        fidelity_evolution = Float64[]
        
        execution_time = @elapsed begin
            for t in times
                exact_ψ = exact_solution_imag(t)
                push!(energy_evolution, real(dot(ψ, H * ψ)))
                push!(fidelity_evolution, abs2(dot(exact_ψ, ψ)))
                method(ψ, H, Δt)
                normalize!(ψ)
            end
        end
        
        final_error = norm(ψ - exact_solution_imag(total_time))
        push!(imag_results, BenchmarkResult(name, final_error, execution_time,
                                          energy_evolution, fidelity_evolution))
    end
    
    return real_results, imag_results, times
end

"""
Create comprehensive visualization of benchmark results
"""
function visualize_benchmark_results(real_results, imag_results, times)
    # 1. Performance Overview
    p1 = scatter(
        [r.execution_time for r in real_results],
        [r.error for r in real_results],
        xlabel="Execution Time (s)",
        ylabel="Final Error",
        label="Real-Time",
        title="Performance Overview",
        marker=:circle
    )
    scatter!(
        [r.execution_time for r in imag_results],
        [r.error for r in imag_results],
        label="Imaginary-Time",
        marker=:square
    )
    
    # 2. Energy Evolution
    p2 = plot(
        title="Energy Evolution",
        xlabel="Time",
        ylabel="Energy"
    )
    for result in real_results
        plot!(times, result.energy_evolution, label=result.method_name)
    end
    
    # 3. State Fidelity
    p3 = plot(
        title="State Fidelity",
        xlabel="Time",
        ylabel="Fidelity"
    )
    for result in real_results
        plot!(times, result.state_fidelity, label=result.method_name)
    end
    
    # 4. Error Bar Plot
    p4 = bar(
        getfield.(real_results, :method_name),
        [getfield.(real_results, :error) getfield.(imag_results, :error)],
        title="Final Error Comparison",
        xlabel="Method",
        ylabel="Error",
        label=["Real-Time" "Imaginary-Time"],
        bar_position=:dodge,
        rotation=45,
        legend=:topleft
    )

    # 5. Time Bar Plot
    p5 = bar(
        getfield.(real_results, :method_name),
        [getfield.(real_results, :execution_time) getfield.(imag_results, :execution_time)],
        title="Execution Time Comparison",
        xlabel="Method",
        ylabel="Time (s)",
        label=["Real-Time" "Imaginary-Time"],
        bar_position=:dodge,
        rotation=45,
        legend=:topleft
    )

    # Combine all plots
    final_plot = plot(p1, p2, p3, p4, p5,
                     layout=(2,3),
                     size=(1200,800),
                     plot_title="Quantum Evolution Methods Benchmark")
    
    return final_plot
end

# Run benchmark
H = Complex{Float64}.([0 1; 1 0])
ψ0 = normalize(Complex{Float64}.([1.0, 0.0]))
Δt = 0.1
total_time = 1.0

real_results, imag_results, times = run_detailed_benchmark(H, ψ0, Δt, total_time)
benchmark_plot = visualize_benchmark_results(real_results, imag_results, times)
savefig(benchmark_plot, "quantum_evolution_benchmark.png")

# Print numerical results
println("\nNumerical Results Summary:")
println("\nReal-Time Methods:")
for result in real_results
    @printf("%s: Error = %.2e, Time = %.3f s\n",
            result.method_name, result.error, result.execution_time)
end

println("\nImaginary-Time Methods:")
for result in imag_results
    @printf("%s: Error = %.2e, Time = %.3f s\n",
            result.method_name, result.error, result.execution_time)
end
