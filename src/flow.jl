struct Flow{M<:AbstractManifold,V,B,T<:AbstractFloat,DM,A<:Tuple,K<:NamedTuple}
    manifold::M
    velocity_field::V
    ν₀::B
    tspan::Tuple{T,T}
    divergence_method::DM
    args::A
    kwargs::K

    function Flow(
        manifold::M,
        velocity_field::V,
        ν₀::B;
        tspan::Tuple{T,T}=(0.0f0, 1.0f0), # Default to Float32
        divergence_method::DM=ForwardDiffDivergence(),
        args::A=(),
        kwargs::K=NamedTuple(),
    ) where {M<:AbstractManifold,V,B,T<:AbstractFloat,DM,A<:Tuple,K<:NamedTuple}
        return new{M,V,B,T,DM,A,K}(manifold, velocity_field, ν₀, tspan, divergence_method, args, kwargs)
    end

end

# This function is needed if we want to work with batches
function get_batch_size(x::AbstractArray{T,n}, manifold::AbstractManifold) where {T,n}
    manifold_rank = length(representation_size(manifold))
    @assert 0 ≤ n - manifold_rank ≤ 1
    batch_size = (n == manifold_rank) ? 1 : size(x, n)
    return batch_size
end

# This is a trick to make the velocity field more efficient (in-place update)
# We also ensure that it stays in the tangent space (do we need this?)
function velocity_field!(dx, x, p, t, flow::Flow)
    batch_size = get_batch_size(x, flow.manifold)
    project!(flow.manifold^batch_size, dx, x, flow.velocity_field(x, p, t))
    return nothing
end

# Project x back into the manifold at each integration step
function projection_callback(manifold::AbstractManifold, batch_size)
    condition = (u, t, integrator) -> true
    affect! = integrator -> project!(manifold^batch_size, integrator.u, integrator.u)
    DiscreteCallback(condition, affect!, save_positions=(false, false))
end

# Apply the flow to a point `x₀` accounting for the geometry. Return the solution of the ODE
function integrate_flow(
    x₀::AbstractArray{T,n},
    p,
    flow::Flow{M,V,B,T,DM,A,K};
    tspan::Tuple{T,T}=flow.tspan,
    save_everystep=false,
    kwargs...
) where {M,V,B,T,DM,A,K,n}
    batch_size = get_batch_size(x₀, flow.manifold)
    is_point(flow.manifold^batch_size, x₀; error=:warn)
    prob = ODEProblem{true}((dx, x, p, t) -> velocity_field!(dx, x, p, t, flow), x₀, tspan, p)
    sol = solve(prob, flow.args...; callback=projection_callback(flow.manifold, batch_size), save_everystep = save_everystep, flow.kwargs..., kwargs...)
    return sol
end

# Apply the flow to a point `x₀`
function flow_map(x₀, p, flow)
    return last(integrate_flow(x₀, p, flow; save_everystep=false, save_start=false).u)
end

# Apply the inverse flow to a point `x`
function inverse_flow_map(x, p, flow)
    return last(integrate_flow(x, p, flow; tspan=reverse(flow.tspan), save_everystep=false, save_start=false).u)
end

# Utility
function (flow::Flow)(x₀, p)
    return flow_map(x₀, p, flow)
end

# This is the in-place version of the augmented vector field f(t, u) = (v, -∇⋅v)
function augmented_dynamics!(du, u, p, t, shape, flow::Flow)
    D = prod(shape)
    x = reshape(selectdim(u, 1, 1:D), shape)
    dx = reshape(selectdim(du, 1, 1:D), shape)
    dlogρ = @view du[D+1:end]
    velocity_field!(dx, x, p, t, flow)
    div_v = divergence_velocity_field_batched(x, p, t, flow, flow.divergence_method)
    dlogρ .= -div_v
    return nothing
end

# Project x back into the manifold at each augmented integration step
function augmented_projection_callback(manifold::AbstractManifold, batch_size, shape)
    condition = (u, t, integrator) -> true
    affect! = integrator -> begin
        x = reshape(selectdim(integrator.u, 1, 1:prod(shape)), shape)
        project!(manifold^batch_size, x, x)
    end
    DiscreteCallback(condition, affect!, save_positions=(false, false))
end

function log_density(ν, x) end

# Apply the flow to a point `x₀` and track the evolution of logρ
function integrate_augmented_flow(
    x₀::AbstractArray{T,n},
    p,
    flow::Flow{M,V,B,T,A,K};
    logρ₀=log_density(flow.ν₀, x₀),
    tspan::Tuple{T,T}=flow.tspan,
    save_everystep=false,
    kwargs...
) where {M,V,B,T,A,K,n}
    u₀ = vcat(reshape(x₀, :), logρ₀)
    shape = size(x₀)
    batch_size = get_batch_size(x₀, flow.manifold)
    is_point(flow.manifold^batch_size, x₀; error=:warn)
    prob = ODEProblem{true}((du, u, p, t) -> augmented_dynamics!(du, u, p, t, shape, flow), u₀, tspan, p)
    sol = solve(
        prob,
        flow.args...;
        callback=augmented_projection_callback(flow.manifold, batch_size, shape),
        save_everystep=save_everystep,
        flow.kwargs...,
        kwargs...
    )
    D = prod(shape)
    x_t = map(u -> reshape(selectdim(u, 1, 1:D), shape), sol.u)
    logρ_t = map(u -> selectdim(u, 1, D+1:length(u)), sol.u)
    return sol.t, x_t, logρ_t
end

# Apply the flow to a point `x₀` and return the log density
function flow_map_with_log_density(x₀, p, flow)
    return last.(integrate_augmented_flow(x₀, p, flow; save_everystep=false, save_start=false)[2:3])
end

# Get the density of the pushforward distribution of the flow at point `x`
function log_density(flow::Flow, p, x)
    batch_size = get_batch_size(x, flow.manifold)
    f = zeros(eltype(x), batch_size)
    x₀, f₀ = last.(integrate_augmented_flow(x, p, flow; logρ₀=f, tspan=reverse(flow.tspan), save_everystep=false, save_start=false)[2:3])
    logρ₀ = log_density(flow.ν₀, x₀)
    logρ = logρ₀ .- f₀
    return logρ
end

nothing