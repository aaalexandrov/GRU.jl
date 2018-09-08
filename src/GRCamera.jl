mutable struct Camera
	xform::Matrix{Float32}
	view::Matrix{Float32}
	proj::Matrix{Float32}
	frustum::Shapes.Convex{Float32}
	viewDirty::Bool
	frustumDirty::Bool

	Camera() = new(eye(Float32, 4), eye(Float32, 4), eye(Float32, 4), Shapes.Convex(Float32, 6), false, true)
end

function settransform(cam::Camera, xform::Matrix{Float32})
	cam.xform[:,:] = xform
	cam.viewDirty = true
	cam.frustumDirty = true
end

gettransform(cam::Camera) = cam.xform

function getview(cam::Camera)
	if cam.viewDirty
		cam.view[:,:] = inv(cam.xform)
		cam.viewDirty = false
	end
	cam.view
end

function setproj(cam::Camera, proj::Matrix{Float32})
	cam.proj[:,:] = proj
	cam.frustumDirty = true
end

getproj(cam::Camera) = cam.proj

function getfrustum(cam::Camera)
	if cam.frustumDirty
		calc_frustum(cam, cam.frustum.planes)
		#Shapes.calcedges(cam.frustum)
		cam.frustumDirty = false
	end
	cam.frustum
end

function calc_frustum(cam::Camera, dest::Matrix{Float32})
	ptMin = Float32[-1, -1, -1]
	ptMax = Float32[ 1,  1,  1]
	projSpaceFrust = Array{Float32}(4, 6)
	Shapes.set_aabb_planes(projSpaceFrust, ptMin, ptMax)
	dest[:,:] = At_mul_B(cam.proj * getview(cam), projSpaceFrust)
end
