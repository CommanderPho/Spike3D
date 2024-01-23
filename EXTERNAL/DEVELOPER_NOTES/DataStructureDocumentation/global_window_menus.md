global_window_menus - printed by print_keys_if_possible on 2024-01-23
===================================================================================================


    global_window_menus: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
	│   ├── name: str (children omitted) - OMITTED TYPE WITH NO SHAPE
	│   ├── menuConnections: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
		│   ├── name: str (children omitted) - OMITTED TYPE WITH NO SHAPE
		│   ├── top_level_menu: PyQt5.QtWidgets.QMenu
		│   ├── actions_dict: dict
			│   ├── actionConnect_Child: PyQt5.QtWidgets.QAction
			│   ├── actionDisconnect_from_driver: PyQt5.QtWidgets.QAction
			│   ├── actionMenuConnections: PyQt5.QtWidgets.QAction
	│   ├── create_new_connected_widget: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
		│   ├── name: str (children omitted) - OMITTED TYPE WITH NO SHAPE
		│   ├── top_level_menu: PyQt5.QtWidgets.QMenu
		│   ├── actions_dict: dict
			│   ├── actionNewConnected2DRaster: PyQt5.QtWidgets.QAction
			│   ├── actionNewConnected3DRaster_PyQtGraph: PyQt5.QtWidgets.QAction
			│   ├── actionNewConnected3DRaster_Vedo: PyQt5.QtWidgets.QAction
			│   ├── actionNewConnectedDataExplorer_ipc: PyQt5.QtWidgets.QAction
			│   ├── actionNewConnectedDataExplorer_ipspikes: PyQt5.QtWidgets.QAction
			│   ├── actionAddMatplotlibPlot_DecodedPosition: PyQt5.QtWidgets.QAction
			│   ├── actionDecoded_Epoch_Slices_Laps: PyQt5.QtWidgets.QAction
			│   ├── actionDecoded_Epoch_Slices_PBEs: PyQt5.QtWidgets.QAction
			│   ├── actionDecoded_Epoch_Slices_Ripple: PyQt5.QtWidgets.QAction
			│   ├── actionDecoded_Epoch_Slices_Replay: PyQt5.QtWidgets.QAction
			│   ├── actionDecoded_Epoch_Slices_Custom: PyQt5.QtWidgets.QAction
			│   ├── actionMenuCreateNewConnectedWidget: PyQt5.QtWidgets.QAction
			│   ├── actionMenuCreateNewConnectedDecodedEpochSlices: PyQt5.QtWidgets.QAction
		│   ├── menuCreateNewConnectedDecodedEpochSlices: PyQt5.QtWidgets.QMenu
	│   ├── debug: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
		│   ├── name: str (children omitted) - OMITTED TYPE WITH NO SHAPE
		│   ├── top_level_menu: PyQt5.QtWidgets.QMenu
		│   ├── actions_dict: dict
			│   ├── actionMenuDebug: PyQt5.QtWidgets.QAction
			│   ├── actionMenuDebugMenuActiveDrivers: PyQt5.QtWidgets.QAction
			│   ├── actionMenuDebugMenuActiveDrivables: PyQt5.QtWidgets.QAction
			│   ├── actionMenuDebugMenuActiveConnections: PyQt5.QtWidgets.QAction
		│   ├── menu_provider_obj: pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.DebugMenuProviderMixin.DebugMenuProviderMixin
			│   ├── _render_widget: pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget.Spike3DRasterWindowWidget
				│   ├── ui: pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Uic_AUTOGEN_Spike3DRasterWindowBase.Ui_RootWidget
				│   ├── applicationName: str
				│   ├── enable_debug_print: bool
				│   ├── _connection_man: pyphocorehelpers.gui.Qt.GlobalConnectionManager.GlobalConnectionManager
				│   ├── params: pyphocorehelpers.DataStructure.general_parameter_containers.VisualizationParameters
				│   ├── playback_controller: pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin.TimeWindowPlaybackController
				│   ├── animationThread: pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin.UpdateRunner
				│   ├── _scheduledAnimationSteps: int
				│   ├── main_menu_window: pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.MainWindowWrapper.PhoBaseMainWindow
			│   ├── _root_window: pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.MainWindowWrapper.PhoBaseMainWindow
				│   ├── _app: PyQt5.QtWidgets.QApplication
				│   ├── ui: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
				│   ├── _connection_man: pyphocorehelpers.gui.Qt.GlobalConnectionManager.GlobalConnectionManager
		│   ├── active_drivers_menu: PyQt5.QtWidgets.QMenu
		│   ├── active_drivables_menu: PyQt5.QtWidgets.QMenu
		│   ├── active_connections_menu: PyQt5.QtWidgets.QMenu
	│   ├── docked_widgets: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
		│   ├── name: str (children omitted) - OMITTED TYPE WITH NO SHAPE
		│   ├── top_level_menu: PyQt5.QtWidgets.QMenu
		│   ├── actions_dict: dict
			│   ├── actionNewDockedMatplotlibView: PyQt5.QtWidgets.QAction
			│   ├── actionNewDockedContextNested: PyQt5.QtWidgets.QAction
			│   ├── actionLongShortDecodedEpochsDockedMatplotlibView: PyQt5.QtWidgets.QAction
			│   ├── actionDirectionalDecodedEpochsDockedMatplotlibView: PyQt5.QtWidgets.QAction
			│   ├── actionPseudo2DDecodedEpochsDockedMatplotlibView: PyQt5.QtWidgets.QAction
			│   ├── actionNewDockedCustom: PyQt5.QtWidgets.QAction
			│   ├── actionMenuDockedWidgets: PyQt5.QtWidgets.QAction
			│   ├── actionAddDockedWidget: PyQt5.QtWidgets.QAction
		│   ├── menu_provider_obj: pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.DockedWidgets_MenuProvider.DockedWidgets_MenuProvider
			│   ├── _render_widget: pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget.Spike3DRasterWindowWidget
				│   ├── ui: pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Uic_AUTOGEN_Spike3DRasterWindowBase.Ui_RootWidget
				│   ├── applicationName: str
				│   ├── enable_debug_print: bool
				│   ├── _connection_man: pyphocorehelpers.gui.Qt.GlobalConnectionManager.GlobalConnectionManager
				│   ├── params: pyphocorehelpers.DataStructure.general_parameter_containers.VisualizationParameters
				│   ├── playback_controller: pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin.TimeWindowPlaybackController
				│   ├── animationThread: pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin.UpdateRunner
				│   ├── _scheduledAnimationSteps: int
				│   ├── main_menu_window: pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.MainWindowWrapper.PhoBaseMainWindow
			│   ├── _root_window: pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.MainWindowWrapper.PhoBaseMainWindow
				│   ├── _app: PyQt5.QtWidgets.QApplication
				│   ├── ui: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
				│   ├── _connection_man: pyphocorehelpers.gui.Qt.GlobalConnectionManager.GlobalConnectionManager
		│   ├── add_docked_widget_menu: PyQt5.QtWidgets.QMenu
	│   ├── create_linked_widget: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
		│   ├── name: str (children omitted) - OMITTED TYPE WITH NO SHAPE
		│   ├── top_level_menu: PyQt5.QtWidgets.QMenu
		│   ├── actions_dict: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
			│   ├── name: str (children omitted) - OMITTED TYPE WITH NO SHAPE
			│   ├── actionCreate_paired_time_synchronized_widget: PyQt5.QtWidgets.QAction
			│   ├── actionTimeSynchronizedOccupancyPlotter: PyQt5.QtWidgets.QAction
			│   ├── actionTimeSynchronizedPlacefieldsPlotter: PyQt5.QtWidgets.QAction
			│   ├── actionCombineTimeSynchronizedPlotterWindow: PyQt5.QtWidgets.QAction
			│   ├── actionTimeSynchronizedDecoderPlotter: PyQt5.QtWidgets.QAction
			│   ├── actionMenuCreateLinkedWidget: PyQt5.QtWidgets.QAction
		│   ├── menu_provider_obj: pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.CreateLinkedWidget_MenuProvider.CreateLinkedWidget_MenuProvider
			│   ├── _render_widget: pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget.Spike3DRasterWindowWidget
				│   ├── ui: pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Uic_AUTOGEN_Spike3DRasterWindowBase.Ui_RootWidget
				│   ├── applicationName: str
				│   ├── enable_debug_print: bool
				│   ├── _connection_man: pyphocorehelpers.gui.Qt.GlobalConnectionManager.GlobalConnectionManager
				│   ├── params: pyphocorehelpers.DataStructure.general_parameter_containers.VisualizationParameters
				│   ├── playback_controller: pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin.TimeWindowPlaybackController
				│   ├── animationThread: pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin.UpdateRunner
				│   ├── _scheduledAnimationSteps: int
				│   ├── main_menu_window: pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.MainWindowWrapper.PhoBaseMainWindow
			│   ├── _root_window: pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.MainWindowWrapper.PhoBaseMainWindow
				│   ├── _app: PyQt5.QtWidgets.QApplication
				│   ├── ui: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer
				│   ├── _connection_man: pyphocorehelpers.gui.Qt.GlobalConnectionManager.GlobalConnectionManager
