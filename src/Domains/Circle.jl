

export Circle


##  Circle


# T Must be in an Algebra
"""
    Circle(c,r,o)

represents the circle centred at `c` with radius `r` which is positively (`o=true`)
or negatively (`o=false`) oriented.
"""
struct Circle{T,V<:Real,TT} <: PeriodicDomain{TT}
	center::T
	radius::V
	orientation::Bool

end

Circle(c::Number,r::Real,o::Bool) = Circle{typeof(c),typeof(r),Complex{typeof(r)}}(c,r,o)
Circle(c::SVector,r::Real,o::Bool) = Circle{typeof(c),typeof(r),typeof(c)}(c,r,o)

Circle(::Type{T1},c::T2,r::V,o::Bool) where {T1,T2,V<:Real} = Circle(convert(promote_type(T1,T2,V),c),
															  convert(promote_type(real(T1),real(T2),V),r),o)
Circle(::Type{T1},c,r::Bool) where {T1<:Number} = Circle(T1,c,r)
Circle(::Type{T1},c,r::Real) where {T1<:Number} = Circle(T1,c,r)
Circle(c,r::Real) = Circle(c,r,true)
Circle(r::Real) = Circle(zero(r),r)
Circle(r::Int) = Circle(Float64,0.,r)
Circle(a::Tuple,r::Real) = Circle(SVector(a...),r)

Circle(::Type{V}) where {V<:Real} = Circle(one(V))
Circle() = Circle(1.0)



isambiguous(d::Circle{T}) where {T<:Number} = isnan(d.center) && isnan(d.radius)
isambiguous(d::Circle{T}) where {T<:SVector} = all(isnan,d.center) && isnan(d.radius)
convert(::Type{Circle{T,V}},::AnyDomain) where {T<:Number,V<:Real} = Circle{T,V}(NaN,NaN)
convert(::Type{IT},::AnyDomain) where {IT<:Circle} = Circle(NaN,NaN)


function _tocanonical(v)
	θ = atan(v[2]-0.0, v[1])  # -0.0 to get branch cut right
    mod2pi(θ)
end
function tocanonical(d::Circle{T},ζ) where T<:Number
    v=mappoint(d,Circle(),ζ)
	_tocanonical(reim(v))
end

function tocanonical(d::Circle{T},ζ) where T<:SVector
    v=mappoint(d,Circle((0.0,0.0),1.0),ζ)
	_tocanonical(v)
end

orientationsign(d::Circle) = d.orientation ? 1 : -1

fromcanonical(d::Circle{T,V,Complex{V}},θ) where {T<:Number,V<:Real} =
	d.radius * exp(orientationsign(d) * 1.0im * θ) + d.center
fromcanonicalD(d::Circle{T},θ) where {T<:Number} =
	orientationsign(d) * d.radius * 1.0im * exp(orientationsign(d) * 1.0im * θ)


fromcanonical(d::Circle{T},θ::Number) where {T<:SVector} =
	d.radius*SVector(cos(orientationsign(d)*θ),sin(orientationsign(d)*θ)) + d.center
fromcanonicalD(d::Circle{T},θ::Number) where {T<:SVector} =
	d.radius*orientationsign(d)*SVector(-sin(orientationsign(d)*θ),cos(orientationsign(d)*θ))


indomain(z,d::Circle) = norm(z-d.center) ≈ d.radius

arclength(d::Circle) = 2π*d.radius
complexlength(d::Circle) = orientationsign(d)*im*arclength(d)  #TODO: why?


==(d::Circle,m::Circle) = d.center == m.center && d.radius == m.radius && d.orientation == m.orientation



mappoint(d1::Circle{T},d2::Circle{V},z) where {T<:SVector,V<:Number} =
	mappoint(Circle(complex(d1.center...),d1.radius),d2,z[1]+im*z[2])

mappoint(d1::Circle{T},d2::Circle{V},z) where {T<:Number,V<:SVector} =
	mappoint(Circle(SVector(d1.center...),d1.radius),d2,SVector(real(z),imag(z)))

function mappoint(d1::Circle,d2::Circle,z)
   v=(z-d1.center)/d1.radius
   d1.orientation != d2.orientation && (v=1/v)
   v*d2.radius+d2.center
end



reverseorientation(d::Circle) = Circle(d.center,d.radius,!d.orientation)
conj(d::Circle) = Circle(conj(d.center),d.radius,!d.orientation)


for op in (:+,:-)
    @eval begin
        $op(c::Number,d::Circle) = Circle($op(c,d.center),d.radius,d.orientation)
        $op(d::Circle,c::Number) = Circle($op(d.center,c),d.radius,d.orientation)
    end
end


*(c::Real,d::Circle) = Circle(*(c,d.center),*(abs(c),d.radius),sign(c)<0 ? !d.orientation : d.orientation)
*(d::Circle,c::Real) = Circle(*(c,d.center),*(abs(c),d.radius),sign(c)<0 ? !d.orientation : d.orientation)



/(c::Number,d::Circle) =
	c==1 ? (d.center==0 ? Circle(d.center,1/d.radius,!d.orientation) :
				Circle(1/d.center,abs(1/(d.center+d.radius)-1/(d.center)),!d.orientation)) :
				c*(1/d)
