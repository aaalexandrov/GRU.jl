abstract type RenderState end
abstract type AlphaBlendState <: RenderState end
abstract type StencilState <: RenderState end
abstract type DepthState <: RenderState end
abstract type CullState <: RenderState end


mutable struct AlphaBlendDisabled <: AlphaBlendState
end

function setstate(::AlphaBlendDisabled)
	glDisable(GL_BLEND)
end

mutable struct AlphaBlendSrcAlpha <: AlphaBlendState
end

function setstate(::AlphaBlendSrcAlpha)
	glEnable(GL_BLEND)
	glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD)
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ZERO)
end

mutable struct AlphaBlendAdditive <: AlphaBlendState
end

function setstate(::AlphaBlendAdditive)
	glEnable(GL_BLEND)
	glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD)
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ZERO)
end

mutable struct AlphaBlendConstant <: AlphaBlendState
	color::Tuple{Float32, Float32, Float32, Float32}
end

function setstate(state::AlphaBlendConstant)
	glEnable(GL_BLEND)
	glBlendColor(state.color...)
	glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD)
	@assert glGetError() == GL_NO_ERROR
	glBlendFuncSeparate(GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR, GL_CONSTANT_ALPHA, GL_ZERO)
	@assert glGetError() == GL_NO_ERROR
end


mutable struct StencilStateDisabled <: StencilState
end

function setstate(::StencilStateDisabled)
	glDisable(GL_STENCIL_TEST)
end


mutable struct DepthStateDisabled <: DepthState
end

function setstate(::DepthStateDisabled)
	glDisable(GL_DEPTH_TEST)
end

mutable struct DepthStateLess <: DepthState
end

function setstate(::DepthStateLess)
	glEnable(GL_DEPTH_TEST)
	glDepthFunc(GL_LESS)
end


mutable struct CullStateDisabled <: CullState
end

function setstate(::CullStateDisabled)
	glDisable(GL_CULL_FACE)
end

mutable struct CullStateCCW <: CullState
end

function setstate(::CullStateCCW)
	glEnable(GL_CULL_FACE)
	# Poly orientation is CCW and we cull the back faces
	glFrontFace(GL_CCW)
	glCullFace(GL_BACK)
end


mutable struct RenderStateHolder
	# we store the states with their type's supertype as key, i.e. all the AlphaBlendStates will go to the same position
	states::Dict{DataType, RenderState}

	RenderStateHolder() = new(Dict{DataType, RenderState}())
end

function resetstate(holder::RenderStateHolder, state::RenderState)
	@assert supertype(T) == RenderState
	delete!(holder.states, state)
end

function setstate(holder::RenderStateHolder, state::RenderState)
	holder.states[supertype(typeof(state))] = state
end

function getstate(holder::RenderStateHolder, ::Type{T}, default) where T <: RenderState
	@assert supertype(T) == RenderState
	get(holder.states, T, default)
end

function set_and_apply(holder::RenderStateHolder, state::RenderState)
	key = supertype(typeof(state))
	if !haskey(holder.states, key) || holder.states[key] != state
		holder.states[key] = state
		setstate(state)
	end
end
