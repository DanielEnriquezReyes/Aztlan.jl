module Optimization
    export adam

    function adam(Objective::Function, gradient::Function, bounds::Matrix;
        n_iter::Int = 100, α::Float64 = 0.02, β₁::Float64 = 0.8, β₂::Float64 = 0.999, ε::Float64 = 1e-8)

        x = bounds[:, 1] + rand(Float64, size(bounds)[1]) .* (bounds[:, 2] - bounds[:, 1])

        m = [0.0 for _ ∈ 1:size(bounds)[1]]
        v = [0.0 for _ ∈ 1:size(bounds)[1]]

        for t ∈ 1:n_iter
            grad = gradient(x...)
            
            m = β₁ .* m + (1.0 - β₁) .* grad
            v = β₂ .* v + (1.0 - β₂) .* grad.^2

            m̂ = m ./ (1.0 - β₁^(t))
            v̂ = v ./ (1.0 - β₂^(t))

            x = x .- α .* m̂ ./ (sqrt.(v̂) .+ ε)
        end

        return x
    end    
end




