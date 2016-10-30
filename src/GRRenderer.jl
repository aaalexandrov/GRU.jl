typealias Color Tuple{Float32, Float32, Float32, Float32}

type Renderer <: AbstractRenderer
	camera::Camera
	resources::Dict{Symbol, WeakRef}
	renderState::RenderStateHolder
	toRender::Vector{Renderable}
	sortFunc::Function
	clearColor::Nullable{Color}
	clearStencil::Nullable{Int}
	clearDepth::Nullable{Float64}

	Renderer() = new(Camera(), Dict{Symbol, WeakRef}(), RenderStateHolder(), Array(Renderable, 0), identity, (0f0, 0f0, 0f0, 1f0), 0, 1.0)
end

function init(renderer::Renderer)
	DevIL.ilInit()
end

function done(renderer::Renderer)
	DevIL.ilShutDown()
	while !isempty(renderer.resources)
		id, res = first(renderer.resources)
		if res.value != nothing
		  done(res.value)
		else
		  delete!(renderer.resources, id)
		end
	end
end

function add_renderer_resource(resource::Resource)
	renderer = resource.renderer
	if haskey(renderer.resources, resource.id)
		resource.id = Symbol("$(resource.id)_$(object_id(resource))")
		@assert !haskey(renderer.resources, resource.id)
	end
	renderer.resources[resource.id] = WeakRef(resource)
end

function remove_renderer_resource(resource::Resource)
	delete!(resource.renderer.resources, resource.id)
end

function has_resource(renderer::Renderer, id::Symbol)
	!haskey(renderer.resources, id) && return false
	renderer.resources[id].value != nothing && return true
	delete!(renderer.resources, id)
	return false
end

get_resource(renderer::Renderer, id::Symbol) = renderer.resources[id].value

function add(renderer::Renderer, renderable::Renderable)
	push!(renderer.toRender, renderable)
end

set_clear_color(renderer::Renderer, c) = renderer.clearColor = c
set_clear_stencil(renderer::Renderer, s) = renderer.clearStencil = s
set_clear_depth(renderer::Renderer, d) = renderer.clearDepth = d

function render_frame(renderer::Renderer)
	gl_clear_buffers(renderer.clearColor, renderer.clearDepth, renderer.clearStencil)
	# cull
	frustum = getfrustum(renderer.camera)
	filter!(r->!Shapes.outside(frustum, getbound(r)), renderer.toRender)
	# sort
	if renderer.sortFunc != identity
		sort!(renderer.toRender, lt = renderer.sortFunc)
	end
	# render
	for r in renderer.toRender
		render(r, renderer)
	end

	empty!(renderer.toRender)
end

function apply(holder::RenderStateHolder, renderer::Renderer)
	for state in values(holder.states)
		set_and_apply(renderer.renderState, state)
	end
end
