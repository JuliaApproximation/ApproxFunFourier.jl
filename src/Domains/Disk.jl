export Disk,𝔻

DomainSets.Disk(::AnyDomain) = Disk(NaN,(NaN,NaN))

const 𝔻 = Disk()

isambiguous(d::Disk) = isnan(radius(d)) && all(isnan, center(d))

polar(r,θ) = SVector(r*cos(θ),r*sin(θ))
ipolar(x,y) = SVector(sqrt(abs2(x)+abs2(y)), atan(y,x))
polar(rθ::SVector) = polar(rθ...)
ipolar(xy::SVector) = ipolar(xy...)

fromcanonical(D::Disk, xy) = radius(D).*xy .+ center(D)
tocanonical(D::Disk, xy) = (xy .- center(D))./radius(D)

checkpoints(d::Disk) = SVector{2}(fromcanonical(d,SVector(.1,.2243)), fromcanonical(d,SVector(-.212423,-.3)))

# function points(d::Disk,n,m,k)
#     ptsx=0.5*(1-gaussjacobi(n,1.,0.)[1])
#     ptst=points(PeriodicSegment(),m)
#
#     Float64[fromcanonical(d,x,t)[k] for x in ptsx, t in ptst]
# end
