import matplotlib.pyplot as plt
import mpldatacursor

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

def on_datacursor_move(sel):
    sel.annotation.draggable(True)
    
mpldatacursor.datacursor(formatter='{x:.2f}, {y:.2f}', 
                         bbox=dict(fc='white', ec='black', alpha=0.9),
                         draggable=True,
                         hover=True,
                         on_move_callback=on_datacursor_move)

def edit_properties(sel):
    title = input("Enter new title: ")
    xlabel = input("Enter new x label: ")
    ylabel = input("Enter new y label: ")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.draw()
    
fig.canvas.mpl_connect('button_press_event', edit_properties)

plt.show()
