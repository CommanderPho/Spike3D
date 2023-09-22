import time
import tkinter as tk
from tkinter import font

# clock_format = '%H:%M:%S' # 24-hour
clock_format = '%I:%M %p'
time_display_duration_ms = int(round(0.75 * 1000)) # 2 seconds
# intra_display_delay_seconds = 8 # 8 seconds # 60.0 * 0.25
intra_display_delay_seconds = int(round(60.0 * 5)) # 8 seconds # 60.0 * 0.25

def show_time():
	current_time = time.strftime(clock_format)
	root = tk.Tk()
	root.overrideredirect(True)
	root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))

	big_font = font.Font(family='Helvetica', size=200, weight='bold')
	label = tk.Label(root, text=current_time, font=big_font)
	label.pack(expand=True)

	root.after(time_display_duration_ms, root.destroy)  # Destroy window after 0.5 seconds
	print(f'showing time at {current_time}')
	root.mainloop()

while True:
	show_time()
	# time.sleep(600)  # Wait for 10 minutes
	time.sleep(intra_display_delay_seconds)  # Wait for 10 minutes
