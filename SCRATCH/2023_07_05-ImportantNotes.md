
# TOPIC: Non-blocking plotting in matplotlib
https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
https://discourse.matplotlib.org/t/plot-show-behavior-with-block-false/23145
# plt.ion()
# ## PLOT: https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
# plt.draw()
# plt.pause(0.001)



# ==================================================================================================================== #
# TODO 2023-07-05 17:59: - Pretty printing custom objects                                                                       #
# ==================================================================================================================== #
https://rethinkdb.com/blog/make-debugging-easier-with-custom-pretty-printers
https://realpython.com/python-pretty-print/
https://sourceware.org/gdb/onlinedocs/gdb/Writing-a-Pretty_002dPrinter.html
https://ipython.readthedocs.io/en/stable/api/generated/IPython.lib.pretty.html#module-IPython.lib.pretty
https://ipython.readthedocs.io/en/stable/config/integrating.html

# from pprint import pprint, pformat
# from pprint import PrettyPrinter
# custom_printer = PrettyPrinter(
#     indent=4,
#     width=100,
#     depth=2,
#     compact=True,
#     sort_dicts=False,
#     underscore_numbers=True
# )

# custom_printer.pprint(users[0])


# Changing General Jupyter/IPython rendering
https://stackoverflow.com/questions/19010036/how-to-make-ipython-output-a-list-without-line-breaks-after-elements


You can use %pprint command to turn on/off pprint feature:
If you want to turn off pprint permanently, make a profile, and add c.PlainTextFormatter.pprint = False to the profile file.


https://stackoverflow.com/questions/21971449/how-do-i-increase-the-cell-width-of-the-jupyter-ipython-notebook-in-my-browser


https://stackoverflow.com/questions/21971449/how-do-i-increase-the-cell-width-of-the-jupyter-ipython-notebook-in-my-browser # this one is about classical Jupyter (not JupyterLab)
https://stackoverflow.com/questions/25583428/ipython-how-to-set-terminal-width

max_width
from pprint import pprint

_ipython_display_


# TODO 2023-07-05 18:20: - [ ] CRITICAL: See IPython.lib.pretty.PrettyPrinter(output, max_width=79, newline='\n', max_seq_length=1000 for potential info about how to fix the shitty default Jupyter linewrappping for outputs
IPython.lib.pretty.PrettyPrinter(output, max_width=79, newline='\n', max_seq_length=1000
                                 


# TOPIC: Should be able to search for existing notes like: #workaround AND #jupyter


# TOPIC: https://stackoverflow.com/questions/395735/how-to-check-whether-a-variable-is-a-class-or-not
