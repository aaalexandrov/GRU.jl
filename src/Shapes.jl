module Shapes

import Base: similar, max, min, union!, empty!, isempty, convert

export Shape, Empty, Space, Line, Plane, Sphere, AABB, Convex
export isvalid, transform, getnormal, getpoint, volume, union!, setplane, getintersection, intersect, outside, similar, min, max, inside


iszero{T}(x::T) = abs(x) < eps(T)

len2(x, y, z) = x*x + y*y + z*z
len2(x) = sumabs2(x)
len(x, y, z) = sqrt(len2(x, y, z))
len(x) = sqrt(len2(x))

lerp(x, y, t) = x + (y-x)*t

function scale3{T}(m::AbstractArray{T, 2})
	l2 = max(len2(m[1,1], m[1,2], m[1,3]), len2(m[2,1], m[2,2], m[2,3]), len2(m[3,1], m[3,2], m[3,3]))
	sqrt(l2)
end

function transform_p3{T}(result::AbstractArray{T, 1}, m::AbstractArray{T, 2}, p::AbstractArray{T, 1})
	@simd for i = 1:3
		@inbounds result[i] = m[i,1]*p[1] + m[i,2]*p[2] + m[i,3]*p[3] + m[i,4]
	end
	nothing
end

function transform_p3{T}(result::AbstractArray{T, 2}, m::AbstractArray{T, 2}, p::AbstractArray{T, 2})
	for c = 1:size(p, 2)
		@simd for r = 1:3
			@inbounds result[r, c] = m[r,1]*p[1,c] + m[r,2]*p[2,c] + m[r,3]*p[3,c] + m[r,4]
		end
	end
	nothing
end

function transform_p4{T}(result::AbstractArray{T, 1}, m::AbstractArray{T, 2}, p::AbstractArray{T, 1})
	@simd for i = 1:3
		@inbounds result[i] = m[i,1]*p[1] + m[i,2]*p[2] + m[i,3]*p[3] + m[i,4]*p[4]
	end
	nothing
end

function transform_p4{T}(result::AbstractArray{T, 2}, m::AbstractArray{T, 2}, p::AbstractArray{T, 2})
	for c = 1:size(p, 2)
		@simd for r = 1:4
			@inbounds result[r, c] = m[r,1]*p[1,c] + m[r,2]*p[2,c] + m[r,3]*p[3,c] + m[r,4]*p[4,c]
		end
	end
	nothing
end

make_interval(a, b) = a < b ? (a, b) : (b, a)
empty_interval(i) = !(i[1] <= i[2]) # Will return true in case one of the values is NaN
infinite_interval(i) = isinf(i[1]) || isinf(i[2])
intersect_interval(min1, max1, min2, max2) = (max(min1, min2), min(max1, max2))
intersect_interval(i1, i2) = intersect_interval(i1[1], i1[2], i2[1], i2[2])
is_intersecting_interval(i1, i2) = i1[2] >= i2[1] && i1[1] <= i2[2] # will return false if either interval contains a NaN


# roots of ax^2+bx+c=0
function quadroots(a, b, c)
	d = b*b-4a*c
	if d < zero(d)
		return oftype(a, NaN), oftype(a, NaN)
	end
	sd = sqrt(d)
	x1 = (-b-sd)/2a
	x2 = (-b+sd)/2a
	return x1, x2
end

planevalue{T}(p::Vector{T}, point::Vector{T}) = dot(p[1:3], point) + p[4]
planenormal{T}(p::Vector{T}) = p[1:3]
function planepoint{T}(p::Vector{T})
	i = indmax(abs(p[i]) for i=1:3)
	return T[x==i ? -p[4]/p[i] : 0 for x=1:3]
end

function normalized_plane(dst, col, n, p)
	invLen = 1 / norm(n)
	dst[1, col] = n[1] * invLen
	dst[2, col] = n[2] * invLen
	dst[3, col] = n[3] * invLen
	dst[4, col] = -dot(p, n) * invLen
	nothing
end

function set_aabb_planes{T}(dst::Matrix{T}, ptMin::Vector{T}, ptMax::Vector{T})
	normalized_plane(dst, 1, T[ 1,  0,  0], ptMin)
	normalized_plane(dst, 2, T[ 0,  1,  0], ptMin)
	normalized_plane(dst, 3, T[ 0,  0,  1], ptMin)
	normalized_plane(dst, 4, T[-1,  0,  0], ptMax)
	normalized_plane(dst, 5, T[ 0, -1,  0], ptMax)
	normalized_plane(dst, 6, T[ 0,  0, -1], ptMax)
end

function normalize_plane(dst, col)
	invLen = 1 / len(dst[1, col], dst[2, col], dst[3, col])
	dst[1, col] *= invLen
	dst[2, col] *= invLen
	dst[3, col] *= invLen
	dst[4, col] *= invLen
	nothing
end

function normalize_planes{T}(planes::Matrix{T})
	for i = 1:size(planes, 2)
		normalize_plane(planes, i)
	end
end


abstract Shape{T <: Real}

similar{S <: Shape}(s::S) = S()
function transform{S <: Shape}(m::Matrix, s::S)
	dst = similar(s)
	transform(dst, m, s)
	dst
end

# empty shape
type Empty{T} <: Shape{T}
end

isvalid(e::Empty) = true
volume{T}(e::Empty{T}) = zero(T)
transform(eDest::Empty, m::Matrix, e::Empty) = nothing

# all space
type Space{T} <: Shape{T}
end

isvalid(s::Space) = true
volume{T}(s::Space{T}) = T(Inf)
transform(sDest::Space, m::Matrix, s::Space) = nothing


type Line{T} <: Shape{T}
	p::Array{T, 2}

	Line(p::Array{T, 2}) = new(p)
	Line() = new(Array(T, 3, 2))
end

function Line{T}(x1::T, y1, z1, x2, y2, z2)
	p = Array{T}(3, 2)
	p[1,1] = x1
	p[2,1] = y1
	p[3,1] = z1
	p[1,2] = x2
	p[2,2] = y2
	p[3,2] = z2
	Line{T}(p)
end

Line(p0, p1) = Line(p0..., p1...)
Line{T}(v1::Vector{T}, v2::Vector{T}) = Line(v1[1], v1[2], v1[3], v2[1], v2[2], v2[3])

isvalid{T}(l::Line{T}) = size(l.p)==(3, 2) && len2(l.p[1, 2] - l.p[1, 1], l.p[2, 2] - l.p[2, 1], l.p[3, 2] - l.p[3, 1]) >= eps(T)

function getvector{T}(l::Line{T})
	v = Vector{T}(3)
	v[1] = l.p[1, 2] - l.p[1, 1]
	v[2] = l.p[2, 2] - l.p[2, 1]
	v[3] = l.p[3, 2] - l.p[3, 1]
	v
end

getpoint{T}(l::Line{T}, t::T) = lerp(l.p[:, 1], l.p[:, 2], t)

function transform{T}(lDest::Line{T}, m::Matrix{T}, l::Line{T})
	for c = 1:2
		@simd for r = 1:3
			@inbounds lDest.p[r, c] = m[r,1]*l.p[1,c] + m[r,2]*l.p[2,c] + m[r,3]*l.p[3,c] + m[r,4]
		end
	end
	nothing
end

function plane2interval{T}(p::Vector{T}, l::Line{T}, interval::Tuple{T, T} = (T(-Inf), T(Inf)))
	@assert isvalid(l)
	@assert size(p) == (4,)

	int = line2plane(l.p, p)
	if !isinf(int)
		# line doesn't lie on the plane
		if isnan(int)
			# line is parallel to the plane
			if planevalue(p, l.p[:, 1]) < zero(T)
				# and on the negative side
				interval = (T(Inf), T(-Inf))
			end
		else
			rayInterval = dot(planenormal(p), getvector(l)) < 0? (T(-Inf), int) : (int, T(Inf))
			interval = intersect_interval(interval, rayInterval)
		end
	end
	interval
end


type Plane{T} <: Shape{T}
	p::Array{T, 1}

	Plane(p::Array{T, 1}) = new(p)
	Plane() = new(Array(T, 4))
end

Plane{T}(a::T, b, c, d) = Plane{T}(T[a, b, c, d])
Plane(n, p) = Plane(n..., -dot(p, n))

isvalid{T}(p::Plane{T}) = size(p.p) == (4,) && len2(p.p[1:3]) >= eps(T)

getnormal{T}(p::Plane{T}) = planenormal(p.p)
getvalue{T}(p::Plane{T}, point::Vector{T}) = planevalue(p.p, point)
getpoint{T}(p::Plane{T}) = planepoint(p.p)

# multiply plane by inverse transpose of matrix
function transform{T}(pDest::Plane{T}, m::Matrix{T}, p::Plane{T})
	m_it = transpose(inv(m))
	transform_p4(pDest.p, m_it, p.p)
	nothing
end

# transform with inverse transpose pre-computed
transform_it{T}(pDest::Plane{T}, m_it::Matrix{T}, p::Plane{T}) = transform_p4(pDest.p, m_it, p.p)


type Sphere{T} <: Shape{T}
	c::Array{T, 1}
	r::T

	Sphere(c::Array{T, 1}, r) = new(c, r)
	Sphere() = new(Array(T, 3), 0)
end

Sphere{T}(cx::T, cy, cz, r) = Sphere{T}(T[cx, cy, cz], convert(T, r))
Sphere(c, z) = Sphere(c..., z)

isvalid{T}(s::Sphere{T}) = size(s.c) == (3,) && !isempty(s)
volume(s::Sphere) = s.r^3*4pi/3
isempty{T}(s::Sphere{T}) = s.r < zero(T)

function empty!{T}(s::Sphere{T})
	s.r = -Inf
	s
end

function union!{T}(s::Sphere{T}, p::Vector{T})
	@assert isvalid(s)
	@assert length(p) == 3

	d = len(p-s.c)
	if d > s.r
		s.r = (d+s.r)/2
		s.c = lerp(p, s.c, s.r/d)
	end
	s
end

function union!{T}(s1::Sphere{T}, s2::Sphere{T})
	c12 = s2.c - s1.c
	d = norm(c12)
	if s1.r >= d + s2.r
		# s1 is the union, nothing needs changing
	elseif s2.r >= d + s1.r
		# s2 is the union
		s1.c[:] = s2.c
		s1.r = s2.r
	else
		d1 = (d + s2.r - s1.r) / 2
		s1.c += c12 * (d1 / d)
		s1.r += d1
	end
	s1
end

function transform{T}(sDest::Sphere{T}, m::Matrix{T}, s::Sphere{T})
	transform_p3(sDest.c, m, s.c)
	sDest.r = s.r * scale3(m)
	nothing
end


type AABB{T} <: Shape{T}
	p::Array{T, 2}

	AABB(p::Array{T, 2}) = new(p)
	AABB() = new(Array(T, 3, 2))
end

AABB{T}(xmin::T, ymin, zmin, xmax, ymax, zmax) = AABB{T}(T[xmin xmax; ymin ymax; zmin zmax])
AABB(pmin, pmax) = AABB(pmin..., pmax...)

isvalid{T}(aabb::AABB{T}) = size(aabb.p) == (3, 2) && !isempty(aabb)
volume(ab::AABB) = prod(ab.p[i, 2] - ab.p[i, 1] for i=1:3)
isempty{T}(ab::AABB{T}) = ab.p[1, 1] > ab.p[1, 2] || ab.p[2, 1] > ab.p[2, 2] || ab.p[3, 1] > ab.p[3, 2]

function empty!{T}(ab::AABB{T})
	ab.p[:, 1] = Inf
	ab.p[:, 2] = -Inf
	ab
end

min{T}(ab::AABB{T}) = ab.p[:, 1]
max{T}(ab::AABB{T}) = ab.p[:, 2]

function addpoint!{T}(minMax::Matrix{T}, p::Vector{T})
	@assert size(minMax) == (3, 2)
	@assert length(p) == 3

	for i = 1:3
		if p[i] < minMax[i, 1]
			minMax[i, 1] = p[i]
		end
		if minMax[i, 2] < p[i]
			minMax[i, 2] = p[i]
		end
	end
	nothing
end

union!{T}(ab::AABB{T}, p::Vector{T}) = addpoint!(ab.p, p)
function union!{T}(ab1::AABB{T}, ab2::AABB{T})
	for i = 1:3
		ab1.p[i, 1] = min(ab1.p[i, 1], ab2.p[i, 1])
		ab1.p[i, 2] = max(ab1.p[i, 2], ab2.p[i, 2])
	end
	ab1
end

function transform{T}(abDest::AABB{T}, m::Matrix{T}, ab::AABB{T})
	t = Array(T, 3)
	p = Array(T, 3)
	p[1] = ab.p[1, 1]
	p[2] = ab.p[2, 1]
	p[3] = ab.p[3, 1]
	transform_p3(t, m, p)
	abDest.p[1, 1] = t[1]
	abDest.p[2, 1] = t[2]
	abDest.p[3, 1] = t[3]
	abDest.p[1, 2] = t[1]
	abDest.p[2, 2] = t[2]
	abDest.p[3, 2] = t[3]
	for i = 1:7
		x = i & 1
		y = (i & 2)>>1
		z = (i & 4)>>2
		p[1] = ab.p[1, x+1]
		p[2] = ab.p[2, y+1]
		p[3] = ab.p[3, z+1]
		transform_p3(t, m, p)
		addpoint!(abDest.p, t)
	end
	nothing
end


# todo: add Capsule and OBB


type EdgeInfo{T <: Real}
	line::Line{T}
	interval::Tuple{T, T}
end

type Adjacency{T <: Real}
	faces::Vector{Vector{Int}}
	edges::Dict{Tuple{Int, Int}, EdgeInfo{T}}
	Adjacency() = new()
end

isvalid(a::Adjacency) = isdefined(a, :faces) && isdefined(a, :edges)

function init{T}(adj::Adjacency{T}, planes::Matrix{T})
	adj.faces = [Int[] for i=1:size(planes, 2)]
	adj.edges = Dict{Tuple{Int, Int}, EdgeInfo{T}}()
	calcedges(adj, planes)
end

function init{T}(adj::Adjacency{T}, ab::AABB{T})
	adj.faces = [filter(j->i != j != (i+3-1)%6+1, 1:6) for i=1:6]
	adj.edges = Dict{Tuple{Int, Int}, EdgeInfo{T}}()

	sz = ab.p[:, 2] - ab.p[:, 1]
	for i = 1:6, j in adj.faces[i]
		iDir = (i-1)%3+1
		if i < j
			jDir = (j-1)%3+1
			pts = Array{T}(3, 2)
			for k = 1:3
				m = 1
				d = 0
				if k == iDir
					m = (i > 3) + 1
				elseif k == jDir
					m = (j > 3) + 1
				else
					d = sz[k]
				end
				pts[k, 1] = ab.p[k, m]
				pts[k, 2] = pts[k, 1] + d
			end
			l = Line{T}(pts)
			int = (T(0), T(1))
			adj.edges[(i, j)] = EdgeInfo(l, int)
		end
	end
end

function addplane{T}(adj::Adjacency{T}, planes::Matrix{T}, index::Int, planeRange::Range{Int} = 1:index-1)
	@assert isvalid(adj)
	@assert size(planes, 1) == 4

	p = Plane{T}(planes[:, index])
	update_edges(adj, p, planeRange)
	for i = planeRange
		pi = Plane{T}(planes[:, i])
		intersection = getintersection(p, pi)
		if isa(intersection, Line)
			interval = (T(-Inf), T(Inf))
			for k = planeRange
				if k != i
					interval = plane2interval(planes[:, k], intersection, interval)
					empty_interval(interval) && break
				end
			end
			if !empty_interval(interval)
				f1 = min(i, index)
				f2 = max(i, index)
				adj.edges[(f1, f2)] = EdgeInfo(intersection, interval)
				push!(adj.faces[f1], f2)
				push!(adj.faces[f2], f1)
			end
		end
	end
end

function update_edges{T}(adj::Adjacency{T}, p::Plane{T}, planeRange::Range{Int})
	for i in planeRange
		k = 1
		while k < length(adj.faces[i])
			j = adj.faces[i][k]
			if j > i && j in planeRange
				e = adj.edges[(i, j)]
				interval = plane2interval(p.p, e.line, e.interval)
				if empty_interval(interval)
					deleteat!(adj.faces[i], k)
					deleteat!(adj.faces[j], findfirst(adj.faces[j], i))
					delete!(adj.edges, (i, j))
					continue
				else
					e.interval = interval
				end
			end
			k += 1
		end
	end
end

function combine{T}(dst::Adjacency{T}, src1::Adjacency{T}, src2::Adjacency{T}, planes::Matrix{T})
	@assert isvalid(src1)
	@assert isvalid(src2)
	@assert size(planes, 2) == length(src1.faces) + length(src2.faces)

	dst.faces = deepcopy(src1.faces)
	dst.edges = deepcopy(src2.edges)

	src2Base = length(src1.faces)
	for f in src2.faces
		push!(dst.faces, map(i->i+src2Base, f))
	end
	for (f, e) in src2.edges
		dst.edges[(f[1] + src2Base, f[2] + src2Base)] = e
	end

	calcedges(dst, planes, src2Base)
end

function calcedges{T}(adj::Adjacency{T}, planes::Matrix{T}, splitLength::Int = 0)
	@assert size(planes, 1) == 4

	cols = size(planes, 2)
	range1 = 1:splitLength
	range2 = splitLength+1:cols
	for i in range1
		update_edges(adj, Plane{T}(Planes[:, i]), range2)
	end
	for i in range2
		if splitLength == 0
			range1 = 1:i-1
		end
		addplane(adj, planes, i, range1)
	end


	#=
	iRange = 1:(splitLength == 0? cols : splitLength)
	for i = iRange
		pi = Plane{T}(planes[:, i])
		jRange = (splitLength == 0? i+1 : splitLength+1):cols
		for j = i+1:cols
			pj = Plane{T}(planes[:, j])
			intersection = getintersection(pi, pj)
			if isa(intersection, Line)
				interval = (T(-Inf), T(Inf))
				for k = 1:cols
					if k != i && k != j
						interval = plane2interval(planes[:, k], intersection, interval)
						empty_interval(interval) && break
					end
				end
				if !empty_interval(interval)
					adj.edges[(i, j)] = EdgeInfo(intersection, interval)
					push!(adj.faces[i], j)
					push!(adj.faces[j], i)
				end
			end
		end
	end
	=#
	adj
end

function select_planes{T}(adj::Adjacency{T}, keep::Vector{Bool})
	@assert length(adj.faces) == length(keep)
	ind = Vector{T}(length(keep))
	nextInd = 1
	for i = 1:length(ind)
		if keep[i]
			ind[i] = nextInd
			nextInd += 1
		else
			ind[i] = 0
		end
	end
	adj.faces = adj.faces[keep]
	for i = 1:length(adj.faces)
		filter!(f->keep[f], adj.faces[i])
		map!(f->ind[f], adj.faces[i])
	end
	edges = adj.edges
	adj.edges = Dict{Tuple{Int, Int}, EdgeInfo{T}}()
	for (f, e) in edges
		if keep[f[1]] && keep[f[2]]
			adj.edges[(ind[f[1]], ind[f[2]])] = e
		end
	end
end

function transform{T}(dst::Adjacency{T}, m::Matrix{T}, src::Adjacency{T})
	dst.faces = deepcopy(src.faces)
	if !isdefined(dst.edges)
		dst.edges = Dict{Tuple{Int, Int}, EdgeInfo{T}}()
	else
		empty!(dst.edges)
	end
	for (f, e) in src.edges
		lDst = Line{T}()
		transform(lDst, m, e.line)
		dst.edges[f] = EdgeInfo(lDst, e.interval)
	end
	nothing
end

type Convex{T} <: Shape{T}
	planes::Array{T, 2}
	adj::Adjacency{T}

	Convex(planes::Matrix{T}) = new(planes, Adjacency{T}())
end

Convex{T}(::Type{T}, planeCount::Int) = Convex{T}(zeros(T, 4, planeCount))
Convex{T}(c1::Convex{T}, c2::Convex{T}) = combine(Convex(T, size(c1.planes, 2) + size(c2.planes, 2)), c1, c2)
Convex{T}(ab::AABB{T}) = set(Convex{T}(Array{T}(4, 6)), ab)

function Convex{T}(planes::Plane{T}...)
	c = Convex(T, length(planes))
	for i in 1:length(planes)
		setplane(c, i, planes[i])
	end
	c
end

isvalid{T}(c::Convex{T}) = size(c.planes, 1) == 4 && size(c.planes, 2) > 0
setplane{T}(c::Convex{T}, planeIndex::Int, p::Vector{T}) = c.planes[:, planeIndex] = p / len(p[1:3])
isclosed{T}(c::Convex{T}) = !(isempty(c.edges) || any(e->infinite_interval(e.interval), values(c.edges)))

function isempty{T}(c::Convex{T})
	@assert(isvalid(c))
	@assert(isvalid(c.adj))
	!isempty(c.planes) && isempty(c.adj.edges)
end

function set{T}(c::Convex{T}, ab::AABB{T})
	@assert size(c.planes) == (4, 6)
	set_aabb_planes(c.planes, ab.p[:, 1], ab.p[:, 2])
	init(c.adj, ab)
	c
end

function combine{T}(dst::Convex{T}, src1::Convex{T}, src2::Convex{T})
	src1Size = size(src1.planes, 2)
	src2Size = size(src2.planes, 2)
	if size(dst.planes, 2) != src1Size + src2Size
		dsp.planes = Array{T}(4, src1Size + src2Size)
	end
	dst.planes[:, 1:src1Size] = src1.planes
	dst.planes[:, src1Size+1:src1Size+src2Size] = src2.planes
	combine(dst.adj, src1.adj, src2.adj, dst.planes)
	dst
end

function calcedges{T}(c::Convex{T})
	@assert isvalid(c)
	normalize_planes(c.planes)
	init(c.adj, c.planes)
end

function select_planes{T}(c::Convex{T}, keep::Vector{Bool})
	c.planes = c.planes[:, keep]
	select_planes(c.adj, keep)
end

function remove_redundant{T}(c::Convex{T})
	@assert isvalid(c)
	@assert isvalid(c.adj)

	cols = size(c.planes, 2)
	keep = [true for i=1:cols]
	if isempty(c.adj.edges)
		for i = 1:cols, j = i+1:cols
			if keep[i] && keep[j] && iszero(dot(planenormal(c.planes[:, i]), planenormal(c.planes[:, j])) - 1)
				# planes are parallel, one of them is redundant
				jInFront = planevalue(c.planes[:, j], planepoint(c.planes[:, i])) < 0
				keep[i] = !jInFront
				keep[j] = jInFront
			end
		end
	else
		for i = 1:cols
			# plane does not form a common edge with any plane if it's redundant
			keep[i] = !isempty(c.adj.faces[i])
		end
	end
	if !all(keep)
		# we're actually removing planes
		select_planes(adj, keep)
	end
end

function transform{T}(cDest::Convex{T}, m::Matrix{T}, c::Convex{T})
	transform_it(cDest, transpose(inv(m)), c)
	if isvalid(c.adj)
		transform(cDest.adj, m, c.adj)
	end
	nothing
end

function transform_it{T}(cDest::Convex{T}, m_it::Matrix{T}, c::Convex{T})
	cols = size(c.planes, 2)
	if size(cDest.planes, 2) != cols
		cDest.planes = Array(T, 3, cols)
	end
	A_mul_B!(cDest.planes, m_it, c.planes)
	for i = 1:cols
		invL = 1/len(cDest.planes[1, i], cDest.planes[2, i], cDest.planes[3, i])
		cDest.planes[1, i] *= invL
		cDest.planes[2, i] *= invL
		cDest.planes[3, i] *= invL
		cDest.planes[4, i] *= invL
	end
	nothing
end

# Conversion functions

convert{T}(::Type{AABB{T}}, s::Sphere{T}) = AABB(s.c-s.r, s.c+s.r)
convert{T}(::Type{AABB{T}}, e::Empty{T}) = empty!(AABB{T}())
convert{T}(::Type{Sphere{T}}, ab::AABB{T}) = Sphere{T}(0.5(min(ab) + max(ab)), 0.5len(max(ab) - min(ab)))
convert{T}(::Type{Sphere{T}}, e::Empty{T}) = empty!(Sphere{T}())


# Intersection functions

getintersection{T}(s1::Shape{T}, s2::Shape{T}) = getintersection(s2, s1)
intersect{T}(s1::Shape{T}, s2::Shape{T}) = intersect(s2, s1)

function line2plane{T}(l::Matrix{T}, plane::Vector{T})
	lv = Vector{T}(3)
	lv[1] = l[1, 2] - l[1, 1]
	lv[2] = l[2, 2] - l[2, 1]
	lv[3] = l[3, 2] - l[3, 1]
	pn = planenormal(plane)
	po = planepoint(plane)
	pl = Vector{T}(3)
	pl[1] = po[1] - l[1, 1]
	pl[2] = po[2] - l[2, 1]
	pl[3] = po[3] - l[3, 1]

	vn = dot(lv, pn)
	d = dot(pl, pn)
	if iszero(vn)
		# line and plane are parallel
		return iszero(d) ? T(Inf) : T(NaN)
	end

	return d / vn
end

function getintersection{T}(l::Line{T}, p::Plane{T})
	@assert isvalid(l)
	@assert isvalid(p)
	line2plane(l.p, p.p)
end

intersect{T}(l::Line{T}, p::Plane{T}) = !isnan(getintersection(l, p))


function getintersection{T}(l::Line{T}, s::Sphere{T})
	@assert isvalid(l)
	@assert isvalid(s)

	po = l.p[:, 1]
	vo = l.p[:, 2] - po
	co = po - s.c
	return quadroots(dot(vo, vo), 2dot(vo, co), dot(co, co)-s.r*s.r)
end

intersect{T}(l::Line{T}, s::Sphere{T}) = !empty_interval(getintersection(l, s))


function getintersection{T}(l::Line{T}, ab::AABB{T})
	@assert isvalid(l)
	@assert isvalid(ab)

	tInt = (T(-Inf), T(Inf))
	for i = 1:3
		lo = l.p[i, 1]
		d = l.p[i, 2] - lo
		if iszero(d)
			# line is parallel to this box's dimension
			if ab.p[i, 1] <= lo <= ab.p[i, 2]
				continue
			else
				tInt = (T(Inf), T(-Inf))
				break
			end
		end
		dimInt = make_interval((ab.p[i, 1] - lo)/d, (ab.p[i, 2] - lo)/d)
		tInt = intersect_interval(tInt, dimInt)
		empty_interval(tInt) && break
	end
	tInt
end

intersect{T}(l::Line{T}, ab::AABB{T}) = !empty_interval(getintersection(l, ab))

function getintersection{T}(ab1::AABB{T}, ab2::AABB{T})
	result = AABB{T}()
	for i = 1:3
		result.p[i, 1] = max(ab1.p[i, 1], ab2.p[i, 1])
		result.p[i, 2] = min(ab1.p[i, 2], ab2.p[i, 2])
	end
	return result
end

intersect{T}(ab1::AABB{T}, ab2::AABB{T}) = isvalid(getintersection(ab1, ab2))


# returns a line, a plane or nothing depending on the relative position of the planes
function getintersection{T}(p1::Plane{T}, p2::Plane{T})
	@assert isvalid(p1)
	@assert isvalid(p2)

	n1 = getnormal(p1)
	n2 = getnormal(p2)
	v = cross(n1, n2)
	anypt2 = getpoint(p2)
	if len2(v) < eps(T)
		#planes are parallel
		if iszero(getvalue(p1, anypt2))
			# a point on one plane lies on the other, so they are coincident
			return p1
		else
			# no intersection
			return nothing
		end
	end
	axis2 = cross(v, n2)
	lineOnP2 = Line(anypt2, anypt2+axis2)
	t = getintersection(lineOnP2, p1)
	@assert !isnan(t) && !isinf(t)
	ptIntersection = getpoint(lineOnP2, t)
	return Line(ptIntersection, ptIntersection + v)
end

intersect(p1::Plane, p2::Plane) = getintersection(p1, p2) != nothing


intersect(c::Convex, e::Empty) = false
intersect(c::Convex, s::Space) = isvalid(c)

function getintersection{T}(c::Convex{T}, l::Line{T}, interval::Tuple{T, T} = (T(-Inf), T(Inf)))
	@assert isvalid(l)
	@assert isvalid(c)

	for i = 1:size(c.planes, 2)
		interval = plane2interval(c.planes[:, i], l, interval)
		empty_interval(interval) && break
	end
	interval
end

intersect(c::Convex, l::Line) = !empty_interval(getintersection(c, l))

function getintersection{T}(c::Convex{T}, p::Plane{T})
	c = Convex{T}(hcat(c.planes, p.p))
	calcedges(c)
	remove_redundant(c)
	c
end

intersect(c::Convex, p::Plane) = !isempty(getintersection(c, p))


function intersect{T}(c::Convex{T}, s::Sphere{T})
	@assert isvalid(c)
	@assert isvalid(c.adj)
	@assert isvalid(s)

	cols = size(c.planes, 2)
	incidentPlanes = Int[]
	for i = 1:cols
		dist = planevalue(c.planes[:, 4], s.c)
		dist < -s.r && return false
		if dist <= s.r
			push!(incidentPlanes, i)
		end
	end
	# if the sphere isn't intersected by any of the planes, and is not wholly on the negative side of any of them, it's inside
	isempty(incidentPlanes) && return true
	for i in incidentPlanes
		plane = c.planes[:, i]
		pn = planenormal(plane)
		po = planepoint(plane)
		projCenter = s.c - pn * dot(s.c - po, pn) # sphere center projected on the plane
		@assert iszero(planevalue(plane, projCenter))
		centerIsInside = true
		for j in c.adj.faces[i]
			# planes with indices i and j have a common edge so they are incident, check if the projected center is on the positive side of j
			if planevalue(c.planes[j], projCenter) < 0
				centerIsInside = false
				break
			end
		end
		# the projection of the center on the plane is inside the area defined by the incident planes of the convex
		centerIsInside && return true
	end
	# the sphere is incident with some planes, but its center projected on any of them doesn't lie inside a convex face
	# in this case the sphere intersects the convex only if one of the convex edges intersects the sphere
	# and the only edges the sphere can intersect are those formed between pairs of incident planes
	for i = 1:length(incidentPlanes), j = i+1:length(incidentPlanes)
		planePair = (incidentPlanes[i], incidentPlanes[j])
		if haskey(c.adj.edges, planePair)
			edge = c.adj.edges[planePair]
			interval = getintersection(edge.line, s)
			# do the line's edge interval and sphere interval intersect?
			is_intersecting_interval(interval, edge.interval) && return true
		end
	end
	return false
end

getintersection{T}(c::Convex{T}, ab::AABB{T}) = getintersection(c, Convex(ab))
intersect{T}(c::Convex{T}, ab::AABB{T}) = !isempty(getintersection(c, ab))


function getintersection{T}(c1::Convex{T}, c2::Convex{T})
	c = Convex(c1, c2)
	remove_redundant(c)
	c
end

intersect(c1::Convex, c2::Convex) = !isempty(getintersection(c1, c2))


outside(c::Convex, e::Empty) = true
outside(c::Convex, s::Space) = false

function outside{T}(c::Convex{T}, s::Sphere{T})
	@assert isvalid(c)
	@assert isvalid(s)
	for i = 1:size(c.planes, 2)
		if dot(c.planes[1:3, i], s.c) + c.planes[4, i] < -s.r
			return true
		end
	end
	return false
end

function outside{T}(c::Convex{T}, ab::AABB{T})
	@assert isvalid(c)
	@assert isvalid(ab)

	for i = 1:size(c.planes, 2)
		n = c.planes[1:3, i]
		# select the point on the box with the maximum value when substituted in the plane equation
		minPt = [n[j]<0 ? ab.p[j, 1] : ab.p[j, 2] for j=1:3]
		if dot(n, minPt) + c.planes[4, i] < 0
			return true
		end
	end
	return false
end

function inside{T}(enclosing::AABB{T}, ab::AABB{T})
	@assert size(enclosing.p) == (3,2)
	@assert size(ab.p) == (3,2)

	@simd for i = 1:3
		@inbounds if enclosing.p[i,1] > ab.p[i,1] || enclosing.p[i,2] < ab.p[i,2]
			return false
		end
	end
	return true
end

function inside{T}(enclosing::AABB{T}, p::Vector{T})
	@assert size(enclosing.p) == (3,2)
	@assert size(p) == (3,)

	@simd for i = 1:3
		@inbounds if !(enclosing.p[i,1] <= p[i] <= enclosing.p[i,2])
			return false
		end
	end
	return true
end

function inside{T}(enclosing::Sphere{T}, p::Vector{T})
	@assert size(enclosing.c) == (3,)
	@assert size(p) == (3,)

	len(p - enclosing.c) <= enclosing.r
end

function inside{T}(enclosing::Convex{T}, p::Vector{T})
	@assert isvalid(enclosing)
	@assert size(p) == (3,)

	for i = 1:size(enclosing.planes, 2)
		planevalue(enclosing.planes[:, i], p) < 0 && return false
	end
	true
end

end
