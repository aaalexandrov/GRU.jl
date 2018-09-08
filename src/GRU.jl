__precompile__()

module GRU

include("GLHelper.jl")
include("Shapes.jl")
include("Math2D.jl")
include("Math3D.jl")
include("FTFont.jl")

using LinearAlgebra

using ModernGL
using .GLHelper

import DevIL
import .Shapes

abstract type AbstractRenderer end
abstract type Renderable end
abstract type Resource end

function init_resource(r::Resource, renderer::AbstractRenderer, id::Symbol)
	r.renderer = renderer
	r.id = id
	add_renderer_resource(r)
	finalizer(done, r)
end

getid(r::Resource) = r.id
getrenderer(r::Resource) = r.renderer

abstract type AbstractMesh <: Resource end
abstract type AbstractTexture <: Resource end

include("GRTypes.jl")

include("GRCamera.jl")
include("GRState.jl")
include("GRRenderer.jl")
include("GRUniforms.jl")
include("GRShader.jl")
include("GRMesh.jl")
include("GRTexture.jl")
include("GRMaterial.jl")
include("GRModel.jl")
include("GRFont.jl")


export Vec2, Vec3, Vec4, Matrix4
export VertexLayout, Mesh, Shader, Texture, Material, Model
export isvalid, init, done, apply, setuniform, getuniform, render, position_func

end
