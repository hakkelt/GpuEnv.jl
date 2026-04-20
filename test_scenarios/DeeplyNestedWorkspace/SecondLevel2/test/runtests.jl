using GPUEnv
using Test

GPUEnv.activate()

backends = GPUEnv.gpu_backends()
@test !isempty(backends)
for backend in backends
    @testset "$(backend.name)" begin
        array_type = gpu_wrapper(backend, Float32)
        gpu_type = backend.array_type
        @test array_type === gpu_type

        x_cpu = Float32.(1:64)
        x_gpu = to_gpu(backend, x_cpu)
        @test isapprox(x_cpu .* x_cpu, Array(x_gpu .* x_gpu); atol = 1.0e-6)

        x = gpu_allocate(backend, Float32, 1000)
        @test size(x) == (1000,)
        @test eltype(x) == Float32
        @test isa(x, gpu_type)

        x_zeros = gpu_zeros(backend, Float32, 1000)
        @test all(x_zeros .== 0)
        @test isa(x_zeros, gpu_type)

        x_ones = gpu_ones(backend, Float32, 1000)
        @test all(x_ones .== 1)
        @test isa(x_ones, gpu_type)

        x_randn = gpu_randn(backend, Float32, 1000)
        @test isa(x_randn, gpu_type)
    end
end

# JSON is a dependency of the workspace root, and should NOT be available:
@test Base.find_package("JSON") === nothing
# YAML is a dependency of the SecondLevel project, and should NOT be available:
@test Base.find_package("YAML") === nothing
# IniFile is a dependency of the ThirdLevel project, and should NOT be available:
@test Base.find_package("IniFile") === nothing
# CSV is a dependency of the workspace root test project, and should NOT be available in this package:
@test Base.find_package("CSV") === nothing
# JSON2 is a dependency of the SecondLevel2 project, and should be available in this package:
@test Base.find_package("JSON2") !== nothing
