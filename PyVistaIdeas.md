Looks like I might be able to do:

def update():
    for dat in data:
        plotter.update_scalars(dat, mesh=mesh)
        time.sleep(1)
        plotter.update()
        
plotter.add_callback(update, interval=100)

to get periodic updates