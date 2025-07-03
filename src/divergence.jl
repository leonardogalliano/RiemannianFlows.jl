abstract type DivergenceMethod end

struct AnalyticalDivergence <: DivergenceMethod end

struct ForwardDiffDivergence <: DivergenceMethod end

struct EnzymeReverseDivergence <: DivergenceMethod end

struct EnzymeForwardDivergence <: DivergenceMethod end

# WARNING: We still have to implement the curvature term if expressed in intrinsic coordinates
function divergence_velocity_field(x, p, t, flow::Flow, ::ForwardDiffDivergence)
    πx = project(flow.manifold, x)
    return tr(ForwardDiff.jacobian(x -> project(flow.manifold, x, flow.velocity_field(x, p, t)), πx))
end

# WARNING: We still have to implement the curvature term if expressed in intrinsic coordinates
function divergence_velocity_field(x, p, t, flow::Flow, ::EnzymeReverseDivergence)
    πx = project(flow.manifold, x)
    return tr(Enzyme.jacobian(Enzyme.Reverse, x -> project(flow.manifold, x, flow.velocity_field(x, p, t)), πx)[1])
end

function divergence_velocity_field_batched(x, p, t, flow::Flow, method::DivergenceMethod)
    xb = RiemannianFlows.get_batch_size(x, flow.manifold) == 1 ? reshape(x, size(x)..., 1) : x
    return map(eachslice(xb; dims=ndims(xb))) do x
        divergence_velocity_field(x, p, t, flow::Flow, method)
    end
end

nothing