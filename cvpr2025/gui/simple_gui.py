import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from cc_hardware.tools.dashboard import Dashboard, DashboardConfig
from cc_hardware.utils import config_wrapper


@config_wrapper
class CVPR25DashboardConfig(DashboardConfig):
    x_range: tuple[float, float] = (-1.0, 1.0)
    y_range: tuple[float, float] = (-1.0, 1.0)
    point_size: float = 1.0


class CVPR25Dashboard(Dashboard[CVPR25DashboardConfig]):
    def setup(self):
        self.app = QtWidgets.QApplication([])
        self.win = QtWidgets.QMainWindow()
        self.plot = pg.PlotWidget()
        self.win.setCentralWidget(self.plot)

        self.pred_scatter = pg.ScatterPlotItem(
            size=self.config.point_size, brush=pg.mkBrush(255, 0, 0, 255)
        )
        self.plot.addItem(self.pred_scatter)

        self.gt_scatter = pg.ScatterPlotItem(
            size=self.config.point_size, brush=pg.mkBrush(0, 255, 0, 255)
        )
        self.plot.addItem(self.gt_scatter)

        self.label = pg.TextItem("", anchor=(0, 1))
        self.plot.addItem(self.label)
        self.label.setPos(self.config.x_range[0], self.config.y_range[1])

        self.plot.setXRange(*self.config.x_range)
        self.plot.setYRange(*self.config.y_range)
        self.win.show()

    def update(
        self,
        frame: int,
        positions: list[tuple[float, float]],
        gt_positions: list[tuple[float, float]],
        **kwargs,
    ):
        self.pred_scatter.setData(*zip(*positions))
        self.gt_scatter.setData(*zip(*gt_positions))
        self.label.setText(f"x: {positions[-1][0]:.2f}, y: {positions[-1][1]:.2f}")
        self.app.processEvents()

    def run(self):
        self.app.exec()

    @property
    def is_okay(self) -> bool:
        return not self.win.isHidden()

    def close(self):
        QtWidgets.QApplication.quit()
        if hasattr(self, "win") and self.win is not None:
            self.win.close()
            self.win = None
        if hasattr(self, "app") and self.app is not None:
            self.app.quit()
            self.app = None
