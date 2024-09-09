export Convolution

"""
	Convolution(G)

Represents a convolution operator with function `G`. When applied to a function `f` it computes `∫_D G(x-y) u(y) dy for x ∈ D`. For now `D` is a `PeriodigSegment(a,b)` with `a<b∈R` and `space(G)` is either `Laurent(D)` or `Fourier(D)`.
"""

abstract type Convolution{S,T} <: Operator{T} end

struct ConcreteConvolution{S<:Space,T} <: Convolution{S,T}
    G::Fun{S,T}
end

function Convolution(G::Fun{S,T}) where {S,T}
    @assert isfinite(arclength(domain(G)))
    ConcreteConvolution(G)
end
ApproxFunBase.domain(C::ConcreteConvolution) = ApproxFunBase.domain(C.G)
ApproxFunBase.domainspace(C::ConcreteConvolution) = ApproxFunBase.space(C.G)
ApproxFunBase.rangespace(C::ConcreteConvolution) = ApproxFunBase.space(C.G)
ApproxFunBase.bandwidths(C::ConcreteConvolution) = error("Please implement convolution bandwidths on "*string(space(C.G)))
ApproxFunBase.defaultgetindex(C::ConcreteConvolution,k::Integer,j::Integer) = error("Please implement convolution getindex on "*string(space(C.G)))

ApproxFunBase.bandwidths(C::ConcreteConvolution{Laurent{PeriodicSegment{R},T}}) where {R<:Real,T} = 0,0
ApproxFunBase.bandwidths(C::ConcreteConvolution{Fourier{PeriodicSegment{R},T}}) where {R<:Real,T} = 1,1

function ApproxFunBase.defaultgetindex(C::ConcreteConvolution{Laurent{PeriodicSegment{R},T1},T2},k::Integer,j::Integer) where {R<:Real,T1,T2}
    fourier_index::Integer = if isodd(k) div(k-1,2) else -div(k,2) end
    if k == j && k ≤ ncoefficients(C.G)
        return (exp(-2pi*1im/arclength(domain(C.G))*fourier_index*first(domain(C.G)))*arclength(domain(C.G))*C.G.coefficients[k])::T2
    else
        return zero(T2)
    end
end

function ApproxFunBase.defaultgetindex(C::ConcreteConvolution{Fourier{PeriodicSegment{R},T1},T2},k::Integer,j::Integer) where {R<:Real,T1,T2}
    fourier_index::Integer = if isodd(k) div(k-1,2) else div(k,2) end
    if k<1 || j<1 || ncoefficients(C.G)==0
        return zero(T2)
    elseif k == 1
        if j==k
            return (arclength(domain(C.G))*C.G.coefficients[1])::T2
        else
            return zero(T2)
        end
    elseif 2*fourier_index ≤ ncoefficients(C.G)
        Gs = if 2*fourier_index ≤ ncoefficients(C.G) C.G.coefficients[2*fourier_index] else zero(T2) end # sine coefficient
        Gc = if 2*fourier_index+1 ≤ ncoefficients(C.G) C.G.coefficients[2*fourier_index+1] else zero(T2) end # cosine coefficient
        phase = 2pi/arclength(domain(C.G))*fourier_index*first(domain(C.G))
        if iseven(k) && j==k
            return (arclength(domain(C.G))*(Gc*cos(phase)-Gs*sin(phase))/2)::T2
        elseif iseven(k) && j==k+1
            return (arclength(domain(C.G))*(Gc*sin(phase)+Gs*cos(phase))/2)::T2
        elseif isodd(k) && j==k
            return (arclength(domain(C.G))*(Gc*cos(phase)-Gs*sin(phase))/2)::T2
        elseif isodd(k) && j==k-1
            return (arclength(domain(C.G))*(-Gc*sin(phase)-Gs*cos(phase))/2)::T2
        else
            return zero(T2)
        end
    else
        return zero(T2)
    end
end

