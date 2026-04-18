using TestItems

@testmodule ScenarioHelpers begin
    using Pkg
    export activate_scenario, GPUEnv_root, scenarios_root

    const GPUEnv_root = normpath(joinpath(@__DIR__, ".."))
    const scenarios_root = joinpath(GPUEnv_root, "test_scenarios")

    function activate_scenario(name::AbstractString)
        root = joinpath(scenarios_root, name)
        Pkg.activate(root)
        Pkg.develop(path = GPUEnv_root)
    end
end

@testitem "TestTarget scenario" setup = [ScenarioHelpers] begin
    using Test, Pkg, Logging

    try
        activate_scenario("TestTarget")
        Pkg.test()
        @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
    catch e
        @error "TestTarget scenario failed with error: $e"
        @test false
    finally
        # Ensure we return to the root environment after the test
        Pkg.activate(GPUEnv_root)
    end
end

@testitem "TestProject scenario" setup = [ScenarioHelpers] begin
    using Test, Pkg, Logging

    try
        activate_scenario("TestProject")
        Pkg.test()
        @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
    catch e
        @error "TestProject scenario failed with error: $e"
        @test false
    finally
        # Ensure we return to the root environment after the test
        Pkg.activate(GPUEnv_root)
    end
end

@testitem "TestWorkspace scenario" setup = [ScenarioHelpers] begin
    using Test, Pkg, Logging

    if VERSION >= v"1.12"
        try
            activate_scenario("TestWorkspace")
            Pkg.test()
            @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
        catch e
            @error "TestWorkspace scenario failed with error: $e"
            @test false
        finally
            # Ensure we return to the root environment after the test
            Pkg.activate(GPUEnv_root)
        end
    end
end

@testitem "Benchmarking scenario" setup = [ScenarioHelpers] begin
    using Test, Pkg, Logging

    if VERSION >= v"1.12"
        try
            activate_scenario("Benchmarking/benchmark")
            include(joinpath(scenarios_root, "Benchmarking", "benchmark", "benchmark.jl"))
            @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
        catch e
            @error "Benchmarking scenario failed with error: $e"
            @test false
        finally
            # Ensure we return to the root environment after the test
            Pkg.activate(GPUEnv_root)
        end
    end
end

@testitem "DeeplyNestedWorkspace scenario" setup = [ScenarioHelpers] begin
    using Test, Pkg, Logging

    if VERSION >= v"1.12"
        @testset "DeeplyNestedWorkspace" begin
            try
                activate_scenario("DeeplyNestedWorkspace")
                Pkg.test()
                @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
            catch e
                @error "DeeplyNestedWorkspace scenario failed with error: $e"
                @test false
            finally
                # Ensure we return to the root environment after the test
                Pkg.activate(GPUEnv_root)
            end
        end
        @testset "DeeplyNestedWorkspace/SecondLevel" begin
            try
                activate_scenario("DeeplyNestedWorkspace/SecondLevel")
                Pkg.test()
                @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
            catch e
                @error "DeeplyNestedWorkspace/SecondLevel scenario failed with error: $e"
                @test false
            finally
                # Ensure we return to the root environment after the test
                Pkg.activate(GPUEnv_root)
            end
        end
        @testset "DeeplyNestedWorkspace/SecondLevel/ThirdLevel" begin
            try
                activate_scenario("DeeplyNestedWorkspace/SecondLevel/ThirdLevel")
                Pkg.test()
                @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
            catch e
                @error "DeeplyNestedWorkspace/SecondLevel/ThirdLevel scenario failed with error: $e"
                @test false
            finally
                # Ensure we return to the root environment after the test
                Pkg.activate(GPUEnv_root)
            end
        end
        @testset "DeeplyNestedWorkspace/SecondLevel2" begin
            try
                activate_scenario("DeeplyNestedWorkspace/SecondLevel2")
                Pkg.test()
                @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
            catch e
                @error "DeeplyNestedWorkspace/SecondLevel2 scenario failed with error: $e"
                @test false
            finally
                # Ensure we return to the root environment after the test
                Pkg.activate(GPUEnv_root)
            end
        end
    end
end
