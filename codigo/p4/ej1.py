from meshpy.triange import build, MeshInfo

def solve_elliptic_2d(polygon, max_volumen, f, g, eq_solve):
	"""
	polygon: shape (n,2)
	"""
	# triangulación o mallado
	coords, elems, boundary = triangulate(polygon, max_volumen)
	
	def local_contrib(elem):
		# contribuciones locales
		return A_e, F_e
	
	# ensamblaje
	A, F = assemble(coords, elems, local_contrib)
	
	# condiciones de borde
	A, F = apply_boundary(A, F, coords, boundary, g)
	
	# solve
	u = eq_solve(A, F)
	
	return u, coords, elems


def triangulate(polygon, max_volumen):
	"""
	polygon: np.array con shape (n,2), en orden, anti horario
	"""
	n = polygon.shape[0]
	facets = [ [i, (i+1) % n] for i in range(n)  ]
	info = MeshInfo()
	info.set_points(polygon)
	info.set_facets(facets)
	mesh = build(info, max_volume = max_volume )
	
	coords = np.array(mesh.points)
	elems = np.array([e for e in mesh.elements])
	
	boundary = np.unique(np.array(mesh.facets).flatten())
	return coords, elems, boundary


def assemble(coords, elems, local_contrib):
	n = coords.shape[0]
	A = np.zeros((n,n))
	F = np.zeros(n)
	for K in elems:
		A_K, F_K = local_contrib( coords[K] )
		J,I = np.meshgrid(K,K)
		A[I,J] += A_K
		F[K] += F_K
	return A, F
