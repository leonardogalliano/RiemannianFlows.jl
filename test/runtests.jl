using SafeTestsets

@safetestset "Exact Test" begin
    include("exact_test.jl")
end

@safetestset "Flow Test" begin
    include("flow_test.jl")
end
