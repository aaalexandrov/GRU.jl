abstract type AbstractImmutableVector end

struct Vec2 <: AbstractImmutableVector
	x::Float32
	y::Float32

	Vec2() = new()
	Vec2(x, y) = new(x, y)
end

struct Vec3 <: AbstractImmutableVector
	x::Float32
	y::Float32
	z::Float32

	Vec3() = new()
	Vec3(x, y, z) = new(x, y, z)
end

struct Vec4 <: AbstractImmutableVector
	x::Float32
	y::Float32
	z::Float32
	w::Float32

	Vec4() = new()
	Vec4(x, y, z, w) = new(x, y, z, w)
end

vec2array(v::T) where T = [v.x, v.y, v.z]
position_func(positionField::Symbol) = v->vec2array(getfield(v, positionField))

struct MatrixColumn3
	e1::Float32
	e2::Float32
	e3::Float32
end

struct MatrixColumn4
	e1::Float32
	e2::Float32
	e3::Float32
	e4::Float32
end

abstract type AbstractImmutableMatrix end

struct Matrix3 <: AbstractImmutableMatrix
	c1::MatrixColumn3
	c2::MatrixColumn3
	c3::MatrixColumn3

	Matrix4() = new()
end

struct Matrix4 <: AbstractImmutableMatrix
	c1::MatrixColumn4
	c2::MatrixColumn4
	c3::MatrixColumn4
	c4::MatrixColumn4

	Matrix4() = new()
end

import Base: size, eltype

eltype(::Type{V}) where V<:AbstractImmutableVector = Float32
size(::Type{Vec2}) = (2,)
size(::Type{Vec3}) = (3,)
size(::Type{Vec4}) = (4,)

isvector(t::DataType) = t <: AbstractImmutableVector
size(::Type{T}, i) where T<:AbstractImmutableVector = size(T)[i]

eltype(::Type{Matrix3}) = Float32
size(::Type{Matrix3}) = (3, 3)

eltype(::Type{Matrix4}) = Float32
size(::Type{Matrix4}) = (4, 4)

size(::Type{T}, i) where T<:AbstractImmutableMatrix = size(T)[i]

ismatrix(t::DataType) = t <: AbstractImmutableMatrix

mutable struct SamplerType{GLTYPE}
end

get_sampler_type(::Type{SamplerType{GLTYPE}}) where GLTYPE = GLTYPE

const gl2jlTypes =
	Dict{UInt16, DataType}([
		(GL_FLOAT, Float32),
		(GL_FLOAT_VEC2, Vec2),
		(GL_FLOAT_VEC3, Vec3),
		(GL_FLOAT_VEC4, Vec4),
		(GL_FLOAT_MAT3, Matrix3),
		(GL_FLOAT_MAT4, Matrix4),
		(GL_SAMPLER_2D, SamplerType{Int64(GL_SAMPLER_2D)})
	])

gl2jltype(glType::Integer) = gl2jlTypes[glType]

const jl2glTypes =
	Dict{DataType, UInt16}([
		(Float32, GL_FLOAT),
		(Float64, GL_DOUBLE),
		(Float16, GL_HALF_FLOAT),
		(UInt16,  GL_UNSIGNED_SHORT),
		(Int16,   GL_SHORT),
		(UInt8,   GL_UNSIGNED_BYTE),
		(Int8,	GL_BYTE),
		(UInt32,  GL_UNSIGNED_INT),
		(Int32,   GL_INT)
	])

jl2gltype(jlType::DataType) = jl2glTypes[jlType]

function typeelements(dataType::DataType)
	fieldNames = fieldnames(dataType)
	if isempty(fieldNames)
		# simple type
		return (dataType, 1)
	else
		elType = Void
		for t in dataType.types
			if elType == Void
				elType = t
			elseif elType != t
				error("typeelements: A field with non-uniform types found")
			end
		end
		return (elType, length(fieldNames))
	end
end

function set_array_field(vert::Vector{T}, field::Symbol, values::Array) where T
	fieldInd = findfirst(fieldnames(T), field)
	elType, elCount = typeelements(fieldtype(T, fieldInd))
	offs = fieldoffset(T, fieldInd)
	dst = convert(Ptr{elType}, pointer(vert)) + offs
	valueStride = div(length(values), length(vert))
	srcBase = 0
	for vertInd = 0:length(vert)-1
		for elInd = 1:elCount
			unsafe_store!(dst, values[srcBase+elInd], elInd)
		end
		dst += sizeof(T)
		srcBase += valueStride
	end
end

function set_array_fields(vert::Vector{T}, fieldArrays::Dict{Symbol, Array}) where T
	for field in fieldnames(T)
		if haskey(fieldArrays, field)
			set_array_field(vert, field, fieldArrays[field])
		else
			info("GRU.set_array_fields(): Missing data for field $field")
		end
	end
end
