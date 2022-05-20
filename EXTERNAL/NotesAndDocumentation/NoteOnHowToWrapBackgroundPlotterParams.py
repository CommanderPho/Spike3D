    def __init__(self, interactor, string, on_editing_end=None,
                 highlight_color=None, highlight_opacity=None, **kwargs):

        super(Textbox, self).__init__(interactor, string, **kwargs)
        
        