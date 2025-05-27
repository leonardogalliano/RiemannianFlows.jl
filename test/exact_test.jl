using RiemannianFlows
using Random
using Test
using Manifolds
using OrdinaryDiffEq
using Zygote
using Statistics

function velocity_field(x, p, t)
    wx = p[1] * atan(x[2] / x[1])
    v = similar(x)
    v[1] = -wx * x[2]
    v[2] = wx * x[1]
    return v
end

RiemannianFlows.divergence_velocity_field(x, p, t, flow::Flow, ::RiemannianFlows.AnalyticalDivergence) = p[1]

struct UniformCircle end

function RiemannianFlows.log_density(::UniformCircle, x::Vector{T}) where {T}
    return [-T(log(2π))]
end

seed = 42
rng = Xoshiro(seed)
manifold = Sphere(1)
ν₀ = UniformCircle()
flow = Flow(manifold, velocity_field, ν₀; args=(Tsit5(),), tspan=(0.0, 1.0))
p = [1.0]
x₀ = rand(rng, manifold)

fwdiff_divergence = RiemannianFlows.divergence_velocity_field(x₀, p, flow.tspan[1], flow, flow.divergence_method)
true_divergence = RiemannianFlows.divergence_velocity_field(x₀, p, flow.tspan[1], flow, RiemannianFlows.AnalyticalDivergence())

@test fwdiff_divergence == true_divergence

x_t, logρ_t = flow_map_with_log_density(x₀, p, flow)

logρ_t_true = -log(2π) - p[1] * flow.tspan[2]
@test isapprox(logρ_t_true, logρ_t[1])
