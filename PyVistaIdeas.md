Looks like I might be able to do:

def update():
    for dat in data:
        plotter.update_scalars(dat, mesh=mesh)
        time.sleep(1)
        plotter.update()
        
plotter.add_callback(update, interval=100)

to get periodic updates



## Animation Loop:
# Creating random data
N = 100
data = np.zeros((1, N, N))
data[:] = np.random.randint(0, 10000, data.shape)

# Creating a mesh from our data
g = pv.UniformGrid()
g.dimensions = np.array(data.shape) + 1
g.spacing = (10, 10, 10)
g.cell_data['data'] = data.flatten()
#Careful with threshold as it will turn your data into UnstructuredGrid
#g = g.threshold([0.0001, int(data.max())])

# Creating scene and loading the mesh
p = pv.Plotter()
p.add_mesh(g, opacity=0.5, name='data', cmap='gist_ncar')
p.show(interactive_update=True)

# Animation
for i in range(5, 1000):
    # Updating our data
    data[:] = np.random.randint(0, 10000, data.shape)
    # Updating scalars
    p.update_scalars(data.flatten())
    #p.mesh['data'] = data.flatten() # setting data to the specified mesh is also possible
    # Redrawing
    p.update()

Although, the shape (essentially the number of cells or points) of the data must stay the same. This means that if data array size changes or data filtered through threshold changes it's number of cells, the loaded mesh will reject it.

A workaround is basically to load a new mesh into the Plotter every time your data is updated.

Swap #Animation section with this snippet and the plane will grow some volume:

# Animation
for i in range(5, 1000):
    # Updating our data
    data = np.full((i, N, N),0)
    data[:] = np.random.randint(0,1000000, data.shape)
    
    # Recreating the mesh
    g = pv.UniformGrid()
    g.dimensions = np.array(data.shape) + 1
    g.spacing = (10, 10, 10)
    g.cell_data['data'] = data.flatten()

    # Reloading the mesh to the scene
    p.clear()
    p.add_mesh(g, opacity=0.5, name='data')

    # Redrawing
    p.update()

Note that scene is interactive only when its updating, so lowering update frequency will make scene pretty laggy