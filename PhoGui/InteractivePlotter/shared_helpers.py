# InteractivePyvistaPlotterBuildIfNeededMixin

# from pyvistaqt import BackgroundPlotter

from pyvistaqt import BackgroundPlotter

class InteractivePyvistaPlotterBuildIfNeededMixin:
    @staticmethod
    def build_new_plotter_if_needed(pActiveTuningCurvesPlotter=None, shape=(1,1)):
        if (pActiveTuningCurvesPlotter is not None):
            if isinstance(pActiveTuningCurvesPlotter, BackgroundPlotter):
                if pActiveTuningCurvesPlotter.app_window.isHidden():
                    print('No open BackgroundPlotter')
                    pActiveTuningCurvesPlotter.close() # Close it to start over fresh
                    pActiveTuningCurvesPlotter = None
                    needs_create_new_backgroundPlotter = True
                else:
                    print('BackgroundPlotter already open, reusing it.. NOT Forcing creation of a new one!')
                    pActiveTuningCurvesPlotter.close() # Close it to start over fresh
                    pActiveTuningCurvesPlotter = None
                    needs_create_new_backgroundPlotter = True
            else:
                print('No open BackgroundPlotter, p is a Plotter object')
                pActiveTuningCurvesPlotter.close()
                pActiveTuningCurvesPlotter = None
                needs_create_new_backgroundPlotter = True
        else:
            print('No extant BackgroundPlotter')
            needs_create_new_backgroundPlotter = True

        if needs_create_new_backgroundPlotter:
            print('Creating a new BackgroundPlotter')
            pActiveTuningCurvesPlotter = BackgroundPlotter(window_size=(1920, 1080), shape=(1,1), off_screen=False) # Use just like you would a pv.Plotter() instance
            print('done.')
        return pActiveTuningCurvesPlotter


class InteractivePyvistaPlotter_ObjectManipulationMixin:
    """ Has a self.plots dict that uses string keys to access named plots
        This mixin adds functions that enables interactive manipulation of plotted objects post-hoc
    """
    ## Plot Manipulation Helpers:
    @property
    def get_plot_objects_list(self):
        """ a list of all valid plot objects """
        return list(self.plots.keys())

    @staticmethod
    def __toggle_visibility(mesh):
        new_vis = not bool(mesh.GetVisibility())
        mesh.SetVisibility(new_vis)
        # return new_vis

    def safe_get_plot(self, plot_id):
        a_plot = self.plots.get(plot_id, None)
        if a_plot is not None:
            return a_plot
        else:
            raise IndexError


    def set_plot_visibility(self, plot_id, is_visibie):
        self.safe_get_plot(plot_id).SetVisibility(is_visibie)

    def toggle_plot_visibility(self, plot_id):
        return InteractivePyvistaPlotter_ObjectManipulationMixin.__toggle_visibility(self.safe_get_plot(plot_id))


