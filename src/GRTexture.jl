mutable struct Texture <: AbstractTexture
	texture::GLuint
	id::Symbol
	renderer::Renderer

	Texture() = new(0)
end

isvalid(tex::Texture) = tex.texture != 0

function init(tex::Texture, renderer::Renderer, data::Ptr{UInt8}, w::Integer, h::Integer, pixelFormat::GLenum; id::Symbol = :texture)
	@assert !isvalid(tex)

	init_resource(tex, renderer, id)
	texture = GLuint[0]
	glGenTextures(1, texture)
	tex.texture = texture[1]

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
	glBindTexture(GL_TEXTURE_2D, tex.texture)
	glTexImage2D(GL_TEXTURE_2D, 0, pixelFormat, w, h, 0, pixelFormat, GL_UNSIGNED_BYTE, data)
	glGenerateMipmap(GL_TEXTURE_2D)
	if glGetError() != GL_NO_ERROR
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
	end
	tex
end

const size2glFormat = [GL_RED, GL_RG, GL_RGB, GL_RGBA]

function init(tex::Texture, renderer::Renderer, data::Array; id::Symbol = :texture)
	w, h = size(data)
	pixelSize = sizeof(eltype(data))
	pixelFormat = size2glFormat[pixelSize]
	init(tex, renderer, pointer(data), w, h, pixelFormat; id = id)
end

# todo: add support for DXT textures

import FileIO, ImageIO, ColorTypes, FixedPointNumbers

function init(tex::Texture, renderer::Renderer, texPath::AbstractString; id::Symbol = Symbol(texPath))
	img = FileIO.load(texPath)
	w, h = size(img)
	elem = eltype(img)
	local fmt, u8elem
	if elem <: ColorTypes.AbstractRGBA
		fmt = GL_RGBA
		u8elem = ColorTypes.RGBA{FixedPointNumbers.N0f8}
	elseif elem <: ColorTypes.AbstractRGB
		fmt = GL_RGB
		u8elem = ColorTypes.RGB{FixedPointNumbers.N0f8}
	elseif elem <: ColorTypes.AbstractGrayA
		fmt = GL_RG
		u8elem = ColorTypes.GrayA{FixedPointNumbers.N0f8}
	elseif elem <: ColorTypes.AbstractGray
		fmt = GL_RED
		u8elem = ColorTypes.Gray{FixedPointNumbers.N0f8}
	else
		error("Unsupported image element type $elem for image file $texPath")
	end

	if elem != u8elem
		img = map(u8elem, img)
	end

	init(tex, renderer, pointer(reinterpret(UInt8, img)), w, h, fmt; id = id)
	tex
end


function done(tex::Texture)
	if isvalid(tex)
		texture = GLuint[tex.texture]
		glDeleteTextures(1, texture)
		tex.texture = 0
		remove_renderer_resource(tex)
	end
end

function apply(tex::Texture, index::Int)
	@assert isvalid(tex)

	glActiveTexture(convert(GLenum, GL_TEXTURE0 + index))
	glBindTexture(GL_TEXTURE_2D, tex.texture)
end

# todo: add functions to set texture / sampler parameters
