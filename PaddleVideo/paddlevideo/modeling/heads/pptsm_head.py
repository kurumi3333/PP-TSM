import math
import paddle
import paddle.nn.functional as F

from .tsn_head import TSNHead
from ..registry import HEADS

from ..weight_init import weight_init_


@HEADS.register()
class ppTSMHead(TSNHead):
    """ ppTSM Head
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 drop_ratio=0.5,
                 std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         drop_ratio=drop_ratio,
                         std=std,
                         **kwargs)

        self.stdv = 1.0 / math.sqrt(self.in_channels * 1.0)

    def init_weights(self):
        """Initiate the FC layer parameters"""

        weight_init_(self.fc,
                     'Uniform',
                     'fc_0.w_0',
                     'fc_0.b_0',
                     low=-self.stdv,
                     high=self.stdv)
        self.fc.bias.learning_rate = 2.0
        self.fc.bias.regularizer = paddle.regularizer.L2Decay(0.)
