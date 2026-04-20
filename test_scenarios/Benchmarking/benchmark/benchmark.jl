using GPUEnv

GPUEnv.activate()

backends = GPUEnv.gpu_backends()
if isempty(backends)
    error("No functional backend found for benchmark scenario.")
else
    for backend in backends
        @show backend
        array_type = gpu_wrapper(backend, Float32)
        gpu_type = backend.array_type
        array_type === gpu_type || error("gpu_wrapper did not return the same type as backend.array_type for Float32")

        x_cpu = Float32.(1:64)
        x_gpu = to_gpu(backend, x_cpu)
        isapprox(x_cpu .* x_cpu, Array(x_gpu .* x_gpu); atol = 1.0e-6) || error("GPU smoke test failed")

        x = gpu_allocate(backend, Float32, 1000)
        size(x) == (1000,) || error("gpu_allocate did not return an array of the correct size")
        eltype(x) == Float32 || error("gpu_allocate did not return an array of the correct element type")
        isa(x, gpu_type) || error("gpu_allocate did not return a GPU array")

        x_zeros = gpu_zeros(backend, Float32, 1000)
        all(x_zeros .== 0) || error("gpu_zeros did not return an array of all zeros")
        isa(x_zeros, gpu_type) || error("gpu_zeros did not return a GPU array")

        x_ones = gpu_ones(backend, Float32, 1000)
        all(x_ones .== 1) || error("gpu_ones did not return an array of all ones")
        isa(x_ones, gpu_type) || error("gpu_ones did not return a GPU array")

        x_randn = gpu_randn(backend, Float32, 1000)
        isa(x_randn, gpu_type) || error("gpu_randn did not return a GPU array")
    end
end

# Test if a dependency from parent project is available (JSON is a dependency of the root project, but not of this package)
try
    using JSON
catch e
    error("JSON dependency was not available through the parent project")
end
