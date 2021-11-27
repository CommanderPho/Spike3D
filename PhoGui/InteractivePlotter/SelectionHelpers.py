

class StartEndDatetimeMixin(DateTimeRenderMixin, object):

    def get_start_date(self):
        return datetime.fromtimestamp(float(self.start_date) / 1000.0)

    def get_end_date(self):
        if self.end_date is None:
            return None
        else:
            return datetime.fromtimestamp(float(self.end_date) / 1000.0)

    def StartEndDatetimeMixin_get_JSON(self):
        return {"start_date":self.get_full_long_date_time_string(self.get_start_date()), "end_date":self.get_full_long_date_time_string(self.get_end_date())}






pActiveTuningCurvesPlotter.picked_cells

class GlobalUserSelectionsManager:
    """
    Note: can remove UserSelectionManager using del user_selection_managers[self.interactor]

    Returns:
        [type]: [description]
    """
    user_selection_managers = {}
    def delete_manager(inter):
        if inter is None:
            return None
        manager = user_selection_managers.get(inter, None)
        if manager is not None and len(manager.widgets) == 0:
            manager.detach()
            del user_selection_managers[inter]

    def manager_exists(inter):
        if inter is None:
            return False
        return inter in user_selection_managers

    def get_manager(inter):
        if inter is None:
            return None
        if user_selection_managers.get(inter, None) is None:
            user_selection_managers[inter] = UserSelectionsManager(inter)
        return user_selection_managers[inter]


class UserSelectionsManager:
    """ Holds a set of user-selections to be saved or restored
        Usage:
    """
    def __init__(self, selection_group_name, selection_group_subname=None, selection_group_unique_key=None):
        self.selection_group_name = selection_group_name        
        if (selection_group_subname is None):
            selection_group_subname = '' # set to empty string
        self.selection_group_subname = selection_group_subname

        if (not (selection_group_unique_key is None)):
            self.selection_group_unique_key = selection_group_unique_key
        else:
            # build a selection_group_unique_key out of the name and subname if it's not set:
            if (not (selection_group_subname is None)):
                self.selection_group_unique_key = '{}_{}'.format(selection_group_name, selection_group_subname)
            else:
                self.selection_group_unique_key = selection_group_name
        # Initialize the list that will hold the selections
        self.selections_list = list()
  
    def __repr__(self) -> str:
        return f"<UserSaveSelections: selection_group_name: {self.selection_group_name}; selection_group_subname: {self.selection_group_subname}>: {self.selections_list}"

class UserAnnotationRecord:
    """
        Usage:
            
    """
    def __init__(self, group_key, item_key, item_data):
        self.group_key = group_key
        self.item_key = item_key
        self.item_data = item_data
        
    def __repr__(self) -> str:
        return f"<UserAnnotationRecord: group_key: {self.group_key}; item_key: {self.item_key}>: {self.item_data}"



