module Graphon

struct SBM
	θ::Matrix{Float64}
	size::Vector{Float64}
end

function _rand(s::SBM, i, j)
	return Int(rand()<s.θ[findfirst(x -> i<=x, s.size),findfirst(x -> j<=x, s.size)])
end

function simulate(s::SBM, n)
	A = Matrix{Int}(undef, n, n)
	for i in 1:n
		A[i,i] = 0
		for j in i+1:n
			A[i,j] = _rand(s, i/n, j/n)
			A[j,i] = A[i,j]
		end
	end
	return A
end

export SBM, simulate

end
