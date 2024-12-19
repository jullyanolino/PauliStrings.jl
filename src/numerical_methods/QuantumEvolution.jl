module QuantumEvolution

export euler_step!, rk4_step!, crank_nicolson_step!, trotter_step!, krylov_step!,
       euler_step_imag!, rk4_step_imag!, crank_nicolson_step_imag!, trotter_step_imag!, krylov_step_imag!

using LinearAlgebra

"""
    euler_step!(ψ, H, Δt)

Perform a single Euler step for real-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H::Matrix{ComplexF64}`: Hamiltonian matrix
- `Δt::Float64`: Time step
"""
function euler_step!(ψ::Vector{ComplexF64}, H::Matrix{ComplexF64}, Δt::Float64)
    ψ .= ψ + Δt * (-1im * H * ψ)
end

"""
    rk4_step!(ψ, H, Δt)

Perform a single Runge-Kutta 4th order step for real-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H::Matrix{ComplexF64}`: Hamiltonian matrix
- `Δt::Float64`: Time step
"""
function rk4_step!(ψ::Vector{ComplexF64}, H::Matrix{ComplexF64}, Δt::Float64)
    k1 = -1im * Δt * H * ψ
    k2 = -1im * Δt * H * (ψ + 0.5 * k1)
    k3 = -1im * Δt * H * (ψ + 0.5 * k2)
    k4 = -1im * Δt * H * (ψ + k3)
    ψ .= ψ + (k1 + 2k2 + 2k3 + k4) / 6
end

"""
    crank_nicolson_step!(ψ, H, Δt)

Perform a single Crank-Nicolson step for real-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H::Matrix{ComplexF64}`: Hamiltonian matrix
- `Δt::Float64`: Time step
"""
function crank_nicolson_step!(ψ::Vector{ComplexF64}, H::Matrix{ComplexF64}, Δt::Float64)
    I = Matrix{ComplexF64}(I, length(ψ), length(ψ))
    ψ .= (I + 0.5im * Δt * H) \ (I - 0.5im * Δt * H) * ψ
end

"""
    trotter_step!(ψ, H1, H2, Δt, n)

Perform a single Trotter-Suzuki step for real-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H1::Matrix{ComplexF64}`: First part of the Hamiltonian
- `H2::Matrix{ComplexF64}`: Second part of the Hamiltonian
- `Δt::Float64`: Time step
- `n::Int`: Number of Trotter steps
"""
function trotter_step!(ψ::Vector{ComplexF64}, H1::Matrix{ComplexF64}, H2::Matrix{ComplexF64}, Δt::Float64, n::Int)
    for _ in 1:n
        ψ .= exp(-1im * H1 * Δt / n) * exp(-1im * H2 * Δt / n) * ψ
    end
end

"""
    krylov_step!(ψ, H, Δt)

Perform a single Krylov subspace step for real-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H::Matrix{ComplexF64}`: Hamiltonian matrix
- `Δt::Float64`: Time step
"""
function krylov_step!(ψ::Vector{ComplexF64}, H::Matrix{ComplexF64}, Δt::Float64)
    ψ .= exp(-1im * Δt * H) * ψ
end

"""
    euler_step_imag!(ψ, H, Δτ)

Perform a single Euler step for imaginary-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H::Matrix{ComplexF64}`: Hamiltonian matrix
- `Δτ::Float64`: Imaginary time step
"""
function euler_step_imag!(ψ::Vector{ComplexF64}, H::Matrix{ComplexF64}, Δτ::Float64)
    ψ .= ψ + Δτ * (-H * ψ)
end

"""
    rk4_step_imag!(ψ, H, Δτ)

Perform a single Runge-Kutta 4th order step for imaginary-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H::Matrix{ComplexF64}`: Hamiltonian matrix
- `Δτ::Float64`: Imaginary time step
"""
function rk4_step_imag!(ψ::Vector{ComplexF64}, H::Matrix{ComplexF64}, Δτ::Float64)
    k1 = -Δτ * H * ψ
    k2 = -Δτ * H * (ψ + 0.5 * k1)
    k3 = -Δτ * H * (ψ + 0.5 * k2)
    k4 = -Δτ * H * (ψ + k3)
    ψ .= ψ + (k1 + 2k2 + 2k3 + k4) / 6
end

"""
    crank_nicolson_step_imag!(ψ, H, Δτ)

Perform a single Crank-Nicolson step for imaginary-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H::Matrix{ComplexF64}`: Hamiltonian matrix
- `Δτ::Float64`: Imaginary time step
"""
function crank_nicolson_step_imag!(ψ::Vector{ComplexF64}, H::Matrix{ComplexF64}, Δτ::Float64)
    I = Matrix{ComplexF64}(I, length(ψ), length(ψ))
    ψ .= (I + 0.5 * Δτ * H) \ (I - 0.5 * Δτ * H) * ψ
end

"""
    trotter_step_imag!(ψ, H1, H2, Δτ, n)

Perform a single Trotter-Suzuki step for imaginary-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H1::Matrix{ComplexF64}`: First part of the Hamiltonian
- `H2::Matrix{ComplexF64}`: Second part of the Hamiltonian
- `Δτ::Float64`: Imaginary time step
- `n::Int`: Number of Trotter steps
"""
function trotter_step_imag!(ψ::Vector{ComplexF64}, H1::Matrix{ComplexF64}, H2::Matrix{ComplexF64}, Δτ::Float64, n::Int)
    for _ in 1:n
        ψ .= exp(-H1 * Δτ / n) * exp(-H2 * Δτ / n) * ψ
    end
end

"""
    krylov_step_imag!(ψ, H, Δτ)

Perform a single Krylov subspace step for imaginary-time evolution.

# Arguments
- `ψ::Vector{ComplexF64}`: Current state vector
- `H::Matrix{ComplexF64}`: Hamiltonian matrix
- `Δτ::Float64`: Imaginary time step
"""
function krylov_step_imag!(ψ::Vector{ComplexF64}, H::Matrix{ComplexF64}, Δτ::Float64)
    ψ .= exp(-Δτ * H) * ψ
end

end
