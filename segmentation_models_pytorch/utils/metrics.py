from . import base
from . import functional as F
from ..base.modules import Activation


class IoU(base.Metric):
    __name__ = "iou_score"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, single_class=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.single_class = single_class

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            single_class=self.single_class
        )


class Fscore(base.Metric):
    __name__ = "dice_score"

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, single_class=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.single_class = single_class

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            single_class=self.single_class
        )


class Accuracy(base.Metric):
    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, single_class=False, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.single_class = single_class

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr,
            y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            single_class=self.single_class
        )


class Recall(base.Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, single_class=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.single_class = single_class

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            single_class=self.single_class
        )


class Precision(base.Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, single_class=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.single_class = single_class

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            single_class=self.single_class
        )

class EuclideanDist(base.Metric):
    __name__ = "euclidean_dist"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, single_class=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.single_class = single_class

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.euclidean_distance(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            single_class=self.single_class
        )

class HammingDist(base.Metric):
    __name__ = "hamming_dist"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, single_class=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.single_class = single_class

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.hamming_distance(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            single_class=self.single_class
        )

class DirectedHausdorff(base.Metric):
    __name__ = "directed_hausdorff"

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, single_class=False, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.single_class = single_class

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.directed_hausdorff(
            y_pr,
            y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            single_class=self.single_class
        )
