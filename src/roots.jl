function complexroots(cfs::Vector{T}) where T<:Union{BigFloat,Complex{BigFloat}}
    a = Fun(Taylor(Circle(BigFloat)),cfs)
    ap = a'
    rts = Array{Complex{BigFloat}}(complexroots(Vector{ComplexF64}(cfs)))
    # Do 3 Newton steps
    for _ = 1:3
        rts .-= a.(rts)./ap.(rts)
    end
    rts
end

complexroots(f::Fun{Laurent{DD,RR}}) where {DD,RR} =
    mappoint.(Ref(Circle()), Ref(domain(f)),
        complexroots(f.coefficients[2:2:end],f.coefficients[1:2:end]))
complexroots(f::Fun{Taylor{DD,RR}}) where {DD,RR} =
    mappoint.(Ref(Circle()), Ref(domain(f)), complexroots(f.coefficients))



function roots(f::Fun{Laurent{DD,RR}}) where {DD,RR}
    irts=filter!(z->in(z,Circle()),complexroots(Fun(Laurent(Circle()),f.coefficients)))
    if length(irts)==0
        Complex{Float64}[]
    else
        rts=fromcanonical.(f, tocanonical.(Ref(Circle()), irts))
        if isa(domain(f),PeriodicSegment)
            sort!(real(rts))  # Make type safe?
        else
            rts
        end
    end
end


roots(f::Fun{Fourier{D,R}}) where {D,R} = roots(Fun(f,Laurent))
