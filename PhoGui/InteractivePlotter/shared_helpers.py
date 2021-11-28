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
