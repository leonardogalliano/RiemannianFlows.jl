using RiemannianFlows
using Random
using Test
using Manifolds
using BenchmarkTools
using OrdinaryDiffEq
using Zygote

struct UniformSphere{M<:AbstractManifold}
    manifold::M
end

function RiemannianFlows.log_density(ν::UniformSphere, x::AbstractArray{T,n}) where {T,n}
    return [-T(log(4π)) for _ in 1:RiemannianFlows.get_batch_size(x, ν.manifold)]
end

seed = 42
rng = Xoshiro(seed)
manifold = Sphere(2)
d = prod(representation_size(manifold))
ν₀ = UniformSphere(manifold)
velocity_field(x, p, t) = p * x
flow = Flow(manifold, velocity_field, ν₀; args=(Tsit5(),))
p = rand(Float32, d, d)
x₀ = Float32.(rand(rng, manifold))

@assert flow_map(x₀, p, flow) == flow(x₀, p)
@assert isapprox(inverse_flow_map(flow_map(x₀, p, flow), p, flow), x₀, atol=1e-5)
sol = integrate_flow(x₀, p, flow; saveat=0.1)
@assert all(map(u -> is_point(manifold, u), sol.u))

# Differentiation test

loss(p) = sum(log_density(flow, p, x₀))
@show Zygote.gradient(loss, p)

# Benchmark
@btime integrate_flow(x₀, p, flow; save_everystep=true)
@btime flow_map(x₀, p, flow)
@btime inverse_flow_map(x₀, p, flow)
@btime integrate_augmented_flow(x₀, p, flow; save_everystep=true)
@btime flow_map_with_log_density(x₀, p, flow)
@btime log_density(flow, p, x₀)

# Batched case
x₀ = Float32.(cat(rand(manifold, 4)..., dims=length(representation_size(manifold)) + 1))
@assert flow_map(x₀, p, flow) == flow(x₀, p)
@assert isapprox(inverse_flow_map(flow_map(x₀, p, flow), p, flow), x₀, atol=1e-5)

# Batched Differentiation test

loss(p) = sum(log_density(flow, p, x₀))
@show Zygote.gradient(loss, p)

# Batched Benchmark
@btime integrate_flow(x₀, p, flow; save_everystep=true)
@btime flow_map(x₀, p, flow)
@btime inverse_flow_map(x₀, p, flow)
@btime integrate_augmented_flow(x₀, p, flow; save_everystep=true)
@btime flow_map_with_log_density(x₀, p, flow)
@btime log_density(flow, p, x₀)