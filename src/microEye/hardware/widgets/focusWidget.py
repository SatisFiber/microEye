import json
from enum import Enum, auto

import numpy as np
import pyqtgraph as pg
from pyqtgraph.widgets.PlotWidget import PlotItem, PlotWidget
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView

from microEye.hardware.stages.stabilizer import (
    Axis,
    FocusPlot,
    FocusStabilizer,
    FocusStabilizerParams,
    FocusStabilizerView,
    RejectionMethod,
    StabilizationMethods,
)
from microEye.qt import Qt, QtWidgets
from microEye.utils.gui_helper import GaussianOffSet


class focusWidget(QtWidgets.QDockWidget):
    __plot_refs = {
        FocusPlot.LOCALIZATIONS: None,
        FocusPlot.LINE_PROFILE: None,
        FocusPlot.LINE_PROFILE_FIT: None,
        FocusPlot.X_SHIFT: {
            'mean': None,
            'rois': [],
        },
        FocusPlot.Y_SHIFT: {
            'mean': None,
            'rois': [],
        },
        FocusPlot.Z_SHIFT: None,
        FocusPlot.XY_POINTS: None,
        FocusPlot.Z_HISTOGRAM: None,
    }

    def __init__(self):
        '''
        Initialize the focusWidget instance.

        Set up the GUI layout, including the ROI settings, buttons, and graph widgets.
        '''
        super().__init__('Focus Stabilization')

        # Remove close button from dock widgets
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        self.focusStabilizerView = FocusStabilizerView()

        self.init_layout()

        self.connectUpdateGui()

    def init_layout(self):
        # display tab
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Graphics Layout
        self.graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        graphicsLayout = self.graphicsLayoutWidget.ci
        pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
        pg.setConfigOption('imageAxisOrder', 'row-major')

        # IR Camera GraphView
        self.view_box = graphicsLayout.addViewBox(
            0, 0, rowspan=2, colspan=2, invertY=True
        )
        self.view_box.setAspectLocked()
        self._image_item = pg.ImageItem(axisOrder='row-major')
        self._image_item.setImage(np.random.normal(size=(512, 512)))

        # --- ROI Items ---
        # Line ROI for REFLECTION
        self.line_roi = pg.ROI(
            pg.Point(50.5, 25.5), pg.Point(100, 100), angle=0, pen='r'
        )
        self.line_roi.addTranslateHandle([0, 0], [0, 1])
        self.line_roi.addScaleRotateHandle([0, 1], [0, 0])

        # Rect ROI for BEADS/ASTIGMATIC/HYBRID
        self.rect_roi_z = pg.RectROI([50.5, 25.5], [100, 100], pen='g')
        self.rect_roi_z.setZValue(10)
        self.rect_roi_z.setVisible(False)
        self.rect_roi_xy = pg.RectROI([150.5, 150.5], [100, 100], pen='b')
        self.rect_roi_xy.setZValue(10)
        self.rect_roi_xy.setVisible(False)

        self.set_rois()

        # Line profile graph
        self._line_profile = graphicsLayout.addPlot(2, 0, rowspan=1, colspan=3)
        self._line_profile.setLabel('bottom', 'Pixel', **self.labelStyle)
        self._line_profile.setLabel('left', 'Intensity [ADU]', **self.labelStyle)
        self._line_profile.showGrid(x=True, y=True)

        # X Shift Graph
        self._x_shift = graphicsLayout.addPlot(3, 0, rowspan=1, colspan=1)
        self._x_shift.setLabel('bottom', 'Time [s]', **self.labelStyle)
        self._x_shift.setLabel('left', 'X', **self.labelStyle)
        self._x_shift.showGrid(x=True, y=True)

        # Y Shift Graph
        self._y_shift = graphicsLayout.addPlot(3, 1, rowspan=1, colspan=1)
        self._y_shift.setLabel('bottom', 'Time [s]', **self.labelStyle)
        self._y_shift.setLabel('left', 'Y', **self.labelStyle)
        self._y_shift.showGrid(x=True, y=True)

        # Z Shift Graph
        self._z_shift = graphicsLayout.addPlot(3, 2, rowspan=1, colspan=1)
        self._z_shift.setLabel('bottom', 'Time [s]', **self.labelStyle)
        self._z_shift.showGrid(x=True, y=True)
        self._z_shift.setLabel('left', 'Z', **self.labelStyle)

        # XY Scatter Graph
        self._xy_scatter = graphicsLayout.addPlot(0, 2, rowspan=1, colspan=1)
        self._xy_scatter.setLabel('bottom', 'X', **self.labelStyle)
        self._xy_scatter.setLabel('left', 'Y', **self.labelStyle)
        self._xy_scatter.showGrid(x=True, y=True)

        # Z Histogram Graph
        self._z_hist = graphicsLayout.addPlot(1, 2, rowspan=1, colspan=1)
        self._z_hist.setLabel('bottom', 'Z', **self.labelStyle)
        self._z_hist.setLabel('left', 'Counts', **self.labelStyle)

        self._init_plot_refs()

        def roiChanged(cls):
            x, y = self.getRoiCoords(cls)

            if cls in [self.rect_roi_z, self.line_roi]:
                roi_manager = FocusStabilizer.instance().roi_manager
                z_roi = roi_manager.get_roi('z')
                z_roi.x1, z_roi.x2 = x
                z_roi.y1, z_roi.y2 = y

            if cls is self.rect_roi_xy:
                roi_manager = FocusStabilizer.instance().roi_manager
                xy_roi = roi_manager.get_roi('xy')
                xy_roi.x1, xy_roi.x2 = x
                xy_roi.y1, xy_roi.y2 = y

            self.set_rois()

        self.line_roi.sigRegionChangeFinished.connect(roiChanged)
        self.rect_roi_z.sigRegionChangeFinished.connect(roiChanged)
        self.rect_roi_xy.sigRegionChangeFinished.connect(roiChanged)

        self.view_box.addItem(self._image_item)
        self.view_box.addItem(self.line_roi)
        self.view_box.addItem(self.rect_roi_z)
        self.view_box.addItem(self.rect_roi_xy)

        splitter.addWidget(self.graphicsLayoutWidget)
        splitter.addWidget(self.focusStabilizerView)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        self.setWidget(splitter)

    def _init_plot_refs(self):
        # Add ScatterPlotItem for localizations
        scatter_locs = pg.ScatterPlotItem()
        scatter_locs.setBrush(color='b')
        scatter_locs.setSymbol('x')
        scatter_locs.setZValue(999)  # Ensure points are on top of image
        self.view_box.addItem(scatter_locs)

        self.__plot_refs = {
            FocusPlot.LOCALIZATIONS: scatter_locs,
            FocusPlot.LINE_PROFILE: self._line_profile.plot(pen='r'),
            FocusPlot.LINE_PROFILE_FIT: self._line_profile.plot(pen='b'),
            FocusPlot.X_SHIFT: {
                'mean': self._x_shift.plot(pen='r'),
                'rois': [],
            },
            FocusPlot.Y_SHIFT: {
                'mean': self._y_shift.plot(pen='r'),
                'rois': [],
            },
            FocusPlot.Z_SHIFT: self._z_shift.plot(pen='r'),
            FocusPlot.XY_POINTS: self._xy_scatter.plot(
                [], pen=None, symbolBrush='r', symbolSize=4, symbolPen=None
            ),
            FocusPlot.Z_HISTOGRAM: pg.BarGraphItem(
                x=range(5), height=[1, 5, 2, 4, 3], width=0.5, brush='#00b9e7'
            ),
        }

        self._z_hist.addItem(self.__plot_refs[FocusPlot.Z_HISTOGRAM])

    def _update_plot_visibility(self, method: StabilizationMethods):
        self._line_profile.setVisible(method == StabilizationMethods.REFLECTION)
        # Update ROI display according to method
        self.line_roi.setVisible(method == StabilizationMethods.REFLECTION)
        self.rect_roi_z.setVisible(
            method
            in [
                StabilizationMethods.BEADS,
                StabilizationMethods.BEADS_ASTIGMATIC,
                StabilizationMethods.HYBRID,
            ]
        )
        self.rect_roi_xy.setVisible(method == StabilizationMethods.HYBRID)

    @property
    def buffer(self):
        return FocusStabilizer.instance().buffer

    def connectUpdateGui(self):
        FocusStabilizer.instance().updateViewBox.connect(self.updateViewBox)
        FocusStabilizer.instance().updatePlots.connect(self.updatePlots)
        self.focusStabilizerView.methodChanged.connect(self._update_plot_visibility)

    def updateViewBox(self, data: np.ndarray):
        self._image_item.setImage(data, _callSync='off')

    def updatePlots(self, kwargs: dict = None):
        '''Updates the graphs.'''

        time: np.ndarray = kwargs.get('time')
        time -= time[0]  # start from 0

        positions: np.ndarray = kwargs.get('positions')
        X, Y, Z = positions[:, 0], positions[:, 1], positions[:, 2]

        Localizations: dict = kwargs.get('localizations')
        line_profile: dict = kwargs.get('line_profile')

        # Line profile
        data = line_profile.get('y')
        xdata = np.arange(len(data))
        self.__plot_refs[FocusPlot.LINE_PROFILE].setData(xdata, data)

        if line_profile.get('fit_params') is not None:
            self.__plot_refs[FocusPlot.LINE_PROFILE_FIT].setData(
                xdata, GaussianOffSet(xdata, *line_profile.get('fit_params'))
            )

        # Shifts
        self.__plot_refs[FocusPlot.X_SHIFT]['mean'].setData(time, X)
        self.__plot_refs[FocusPlot.Y_SHIFT]['mean'].setData(time, Y)
        self.__plot_refs[FocusPlot.Z_SHIFT].setData(time, Z)

        # plot Z histogram
        hist, bin_edges = np.histogram(Z, bins=30, range=(np.nanmin(Z), np.nanmax(Z)))

        z_hist: pg.BarGraphItem = self.__plot_refs[FocusPlot.Z_HISTOGRAM]
        z_hist.setOpts(
            x=np.mean(
                (
                    bin_edges[:-1],
                    bin_edges[1:],
                ),
                axis=0,
            ),
            height=hist,
            width=bin_edges[1] - bin_edges[0],
        )

        # XY Scatter
        self.__plot_refs[FocusPlot.XY_POINTS].setData(X, Y)

        # Localizations
        self.__plot_refs[FocusPlot.LOCALIZATIONS].setData(
            Localizations.get('x', []), Localizations.get('y', [])
        )

    def getRoiCoords(self, roi: pg.ROI = None):
        if isinstance(roi, pg.RectROI):
            x1, y1 = roi.pos()
            x2, y2 = roi.pos() + roi.size()
            return [x1, x2], [y1, y2]
        else:
            x1, y1 = roi.pos()
            angle_rad = np.radians(-roi.angle())
            length = roi.size()[1]  # FIXED: use [0] for length
            dx = length * np.sin(angle_rad)
            dy = length * np.cos(angle_rad)
            x2, y2 = x1 + dx, y1 + dy
            return [x1, x2], [y1, y2]

    def set_rois(self):
        roi_manager = FocusStabilizer.instance().roi_manager

        z_roi = roi_manager.get_roi('z')

        dx = z_roi.x2 - z_roi.x1
        dy = z_roi.y2 - z_roi.y1
        length = np.hypot(dx, dy)
        angle = - np.degrees(np.arctan2(dx, dy))

        self.line_roi.setPos([z_roi.x1, z_roi.y1], finish=False)
        self.line_roi.setSize([1, length], finish=False)
        self.line_roi.setAngle(angle, finish=False)

        self.rect_roi_z.setPos([z_roi.x1, z_roi.y1], finish=False)
        self.rect_roi_z.setSize(
            [dx, dy], finish=False
        )

        xy_roi = roi_manager.get_roi('xy')

        self.rect_roi_xy.setPos([xy_roi.x1, xy_roi.y1], finish=False)
        self.rect_roi_xy.setSize(
            [xy_roi.x2 - xy_roi.x1, xy_roi.y2 - xy_roi.y1], finish=False
        )

    def get_config(self) -> dict:
        return self.focusStabilizerView.get_config()

    def load_config(self, config: dict) -> None:
        if not isinstance(config, dict):
            raise TypeError('Configuration must be a dictionary.')

        self.focusStabilizerView.load_config(config)
        self.set_rois()

    def __str__(self):
        return 'Focus Stabilization Widget'
