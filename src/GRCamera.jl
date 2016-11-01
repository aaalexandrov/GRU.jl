type Camera
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
		cam.frustumDirty = false
	end
	cam.frustum
end

function normalized_plane(dst, col, n, p)
	invLen = 1 / norm(n)
	dst[1, col] = n[1] * invLen
	dst[2, col] = n[2] * invLen
	dst[3, col] = n[3] * invLen
	dst[4, col] = -dot(p, n) * invLen
end


function calc_frustum(cam::Camera, dest::Matrix{Float32})
	ptMin = Float32[-1, -1, -1]
	ptMax = Float32[ 1,  1,  1]
	projSpaceFrust = Array(Float32, 4, 6)
	normalized_plane(projSpaceFrust, 1, Float32[ 1,  0,  0], ptMin)
	normalized_plane(projSpaceFrust, 2, Float32[ 0,  1,  0], ptMin)
	normalized_plane(projSpaceFrust, 3, Float32[ 0,  0,  1], ptMin)
	normalized_plane(projSpaceFrust, 4, Float32[-1,  0,  0], ptMax)
	normalized_plane(projSpaceFrust, 5, Float32[ 0, -1,  0], ptMax)
	normalized_plane(projSpaceFrust, 6, Float32[ 0,  0, -1], ptMax)

	dest[:,:] = At_mul_B(cam.proj * getview(cam), projSpaceFrust)
end
