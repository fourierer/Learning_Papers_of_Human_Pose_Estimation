Learning_Papers_for_Human_Pose_Estimation

### 人体姿态估计论文学习

#### top-down

##### 一、论文1

论文：Simple Baselines for Human Pose Estimation and Tracking  

repo： https://github.com/microsoft/human-pose-estimation.pytorch 

数据集：MSCOCO2017

（简介：这篇文章的思路就是使用ResNet作为backbone，将最后的分类层更改为反卷积层来生成heatmap）



1.安装各种环境（原repo中1-6）

这里选择pytorch1.1.0以及匹配版本的torchvision，python3.6，没有disable cudnn for batch_norm，并且在requirements.txt中删去了torchvision(前面安装pytorch1.1.0的时候已经装了torchvision)。



2.下载模型以及构建目录（原repo中7-9）

下载imagenet上预训练的模型6个，3个pytorch版本的，3个caffe版本的；下载coco预训练模型6个；下载mpii预训练模型6个；googledrive上需要翻墙下载，imagenet上训好的pytorch模型下载方式可以在我之前写的repo https://github.com/fourierer/interview/blob/master/pytorch_tensorflow.md 中看到。

模型下载好之后，按照原repo上写的来构建目录。



3.数据集准备

按照原repo给的链接下载各种文件即可，这里可以在MSCOCO数据集和~/data/coco/images之间建个软链接来节省空间，就不需要额外占用空间。

**COCO_val2017_detection_AP_H_56_person.json文件中是已经使用faster-RCNN检测出来的人，可以直接用于后续的全卷积网络。**



4.训练和测试

torch1.1以上才可以使用tensorboard，在运行训练代码的时候报错：

```python
ImportError:TensorBoard logging requires TensorBoard with Python summary writer installed. This should be available in 1.14 or above.
```

提示TensorBoard版本需要是1.14或者以上，使用pip list查看安装列表，显示TensorBoard版本是2.1，这里需要再安装两个库即可解决，具体原因个人也不太清楚。

```python
pip install tb-nightly
pip install future
```

训练运行指令：

```shell
python pose_estimation/train.py --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml
```



网络代码解读pose_resnet.py：

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS, # 3
            extra.NUM_DECONV_FILTERS, # 256 256 256
            extra.NUM_DECONV_KERNELS, # 4 4 4
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1], # 256
            out_channels=cfg.MODEL.NUM_JOINTS, # 17
            kernel_size=extra.FINAL_CONV_KERNEL, # 1
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        # num_layers:最后反卷积层的层数，3
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS # 默认是50
    style = cfg.MODEL.STYLE

    block_class, layers = resnet_spec[num_layers]

    if style == 'caffe':
        block_class = Bottleneck_CAFFE

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
```

在图中标注了一些尺寸信息，整体来说就是将resnet50的全连接层变为3层反卷积层，最后再通过一个卷积层生成17个通道的heatmap。注意以下两点：

（1）nn.ConvTranspose2d函数

函数参数如下：

```python
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
```

- `in_channels(int)` – 输入信号的通道数
- `out_channels(int)` – 卷积产生的通道数
- `kerner_size(int or tuple)` - 卷积核的大小
- `stride(int or tuple,optional)` - 卷积步长
- `padding(int or tuple, optional)` - 输入的每一条边补充0的层数
- `output_padding(int or tuple, optional)` - 输出的每一条边补充0的层数
- `dilation(int or tuple, optional)`– 卷积核元素之间的间距
- `groups(int, optional)` – 从输入通道到输出通道的阻塞连接数
- `bias(bool, optional)` - 如果`bias=True`，添加偏置

反卷积的操作流程实际上也是卷积，只不过在卷积之前先进行插值（插0），再经过正常的卷积，具体如下：

假设原图尺寸为$H*W$，在原先高度以及宽度方向的每两个相邻中间插上$stride−1$列0（这里的步长是参数中的步长），则插值之后的尺寸为$H+(stride-1)*(H-1)$以及$W+(stride-1)*(W-1)$；在此基础上进行卷积操作，这时的步长直接固定为1，卷积核尺寸为参数中的kernel_size，填充值为$size−padding−1$，$padding$为参数中的填充（padding，不是output_padding），卷积尺寸变化公式为：
$$
Heightout=(Heightin+2∗padding−kernelsize)/strides+1
$$
将插值之后的尺寸代入$Heightin$，得到：
$$
[H+(stride−1)∗(H−1)+2∗(size−padding−1)−size]/1+1
$$
化简得到：
$$
(H−1)∗stride−2∗padding+size
$$
最后再加上output_padding，尺寸公式总结为：
$$
H_{out}=(H_{in}−1)stride−2padding+kernelsize+output\_padding
$$
例如：

**输入特征图A：**`3*3`
**输入卷积核K：**`kernel`为`3*3`， `stride`为2， `padding`为1

**新的特征图A’：**`3+(3−1)∗(2−1)=3+2=5`，注意加上`padding`之后才是7。
**新的卷积核设置K’:** `kernel`不变，`stride`为1，`padding`=3−1−1=1

**最终结果：**(5+2−3)/1+1=5



（2）修改train.py代码，直接查看模型结构以及输出

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from torchsummmaryX import summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    print(model)
    img = torch.zeros([1, 3, 256, 192])
    y = model(img)
    print(y.size())
    summary(model, torch.zeros(1, 3, 256, 192)) # 测试模型大小和计算量

    '''
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_perf = 0.0
    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, tb_log_dir,
                                  writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
    '''


if __name__ == '__main__':
    main()
```

输出信息如下：

```
PoseResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (layer2): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (layer3): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (4): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (5): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (layer4): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (deconv_layers): Sequential(
    (0): ConvTranspose2d(2048, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace)
  )
  (final_layer): Conv2d(256, 17, kernel_size=(1, 1), stride=(1, 1))
)
torch.Size([1, 17, 64, 48])
=====================================================================================================
                                             Kernel Shape       Output Shape  \
Layer
0_conv1                                     [3, 64, 7, 7]   [1, 64, 128, 96]
1_bn1                                                [64]   [1, 64, 128, 96]
2_relu                                                  -   [1, 64, 128, 96]
3_maxpool                                               -    [1, 64, 64, 48]
4_layer1.0.Conv2d_conv1                    [64, 64, 1, 1]    [1, 64, 64, 48]
5_layer1.0.BatchNorm2d_bn1                           [64]    [1, 64, 64, 48]
6_layer1.0.ReLU_relu                                    -    [1, 64, 64, 48]
7_layer1.0.Conv2d_conv2                    [64, 64, 3, 3]    [1, 64, 64, 48]
8_layer1.0.BatchNorm2d_bn2                           [64]    [1, 64, 64, 48]
9_layer1.0.ReLU_relu                                    -    [1, 64, 64, 48]
10_layer1.0.Conv2d_conv3                  [64, 256, 1, 1]   [1, 256, 64, 48]
11_layer1.0.BatchNorm2d_bn3                         [256]   [1, 256, 64, 48]
12_layer1.0.downsample.Conv2d_0           [64, 256, 1, 1]   [1, 256, 64, 48]
13_layer1.0.downsample.BatchNorm2d_1                [256]   [1, 256, 64, 48]
14_layer1.0.ReLU_relu                                   -   [1, 256, 64, 48]
15_layer1.1.Conv2d_conv1                  [256, 64, 1, 1]    [1, 64, 64, 48]
16_layer1.1.BatchNorm2d_bn1                          [64]    [1, 64, 64, 48]
17_layer1.1.ReLU_relu                                   -    [1, 64, 64, 48]
18_layer1.1.Conv2d_conv2                   [64, 64, 3, 3]    [1, 64, 64, 48]
19_layer1.1.BatchNorm2d_bn2                          [64]    [1, 64, 64, 48]
20_layer1.1.ReLU_relu                                   -    [1, 64, 64, 48]
21_layer1.1.Conv2d_conv3                  [64, 256, 1, 1]   [1, 256, 64, 48]
22_layer1.1.BatchNorm2d_bn3                         [256]   [1, 256, 64, 48]
23_layer1.1.ReLU_relu                                   -   [1, 256, 64, 48]
24_layer1.2.Conv2d_conv1                  [256, 64, 1, 1]    [1, 64, 64, 48]
25_layer1.2.BatchNorm2d_bn1                          [64]    [1, 64, 64, 48]
26_layer1.2.ReLU_relu                                   -    [1, 64, 64, 48]
27_layer1.2.Conv2d_conv2                   [64, 64, 3, 3]    [1, 64, 64, 48]
28_layer1.2.BatchNorm2d_bn2                          [64]    [1, 64, 64, 48]
29_layer1.2.ReLU_relu                                   -    [1, 64, 64, 48]
30_layer1.2.Conv2d_conv3                  [64, 256, 1, 1]   [1, 256, 64, 48]
31_layer1.2.BatchNorm2d_bn3                         [256]   [1, 256, 64, 48]
32_layer1.2.ReLU_relu                                   -   [1, 256, 64, 48]
33_layer2.0.Conv2d_conv1                 [256, 128, 1, 1]   [1, 128, 64, 48]
34_layer2.0.BatchNorm2d_bn1                         [128]   [1, 128, 64, 48]
35_layer2.0.ReLU_relu                                   -   [1, 128, 64, 48]
36_layer2.0.Conv2d_conv2                 [128, 128, 3, 3]   [1, 128, 32, 24]
37_layer2.0.BatchNorm2d_bn2                         [128]   [1, 128, 32, 24]
38_layer2.0.ReLU_relu                                   -   [1, 128, 32, 24]
39_layer2.0.Conv2d_conv3                 [128, 512, 1, 1]   [1, 512, 32, 24]
40_layer2.0.BatchNorm2d_bn3                         [512]   [1, 512, 32, 24]
41_layer2.0.downsample.Conv2d_0          [256, 512, 1, 1]   [1, 512, 32, 24]
42_layer2.0.downsample.BatchNorm2d_1                [512]   [1, 512, 32, 24]
43_layer2.0.ReLU_relu                                   -   [1, 512, 32, 24]
44_layer2.1.Conv2d_conv1                 [512, 128, 1, 1]   [1, 128, 32, 24]
45_layer2.1.BatchNorm2d_bn1                         [128]   [1, 128, 32, 24]
46_layer2.1.ReLU_relu                                   -   [1, 128, 32, 24]
47_layer2.1.Conv2d_conv2                 [128, 128, 3, 3]   [1, 128, 32, 24]
48_layer2.1.BatchNorm2d_bn2                         [128]   [1, 128, 32, 24]
49_layer2.1.ReLU_relu                                   -   [1, 128, 32, 24]
50_layer2.1.Conv2d_conv3                 [128, 512, 1, 1]   [1, 512, 32, 24]
51_layer2.1.BatchNorm2d_bn3                         [512]   [1, 512, 32, 24]
52_layer2.1.ReLU_relu                                   -   [1, 512, 32, 24]
53_layer2.2.Conv2d_conv1                 [512, 128, 1, 1]   [1, 128, 32, 24]
54_layer2.2.BatchNorm2d_bn1                         [128]   [1, 128, 32, 24]
55_layer2.2.ReLU_relu                                   -   [1, 128, 32, 24]
56_layer2.2.Conv2d_conv2                 [128, 128, 3, 3]   [1, 128, 32, 24]
57_layer2.2.BatchNorm2d_bn2                         [128]   [1, 128, 32, 24]
58_layer2.2.ReLU_relu                                   -   [1, 128, 32, 24]
59_layer2.2.Conv2d_conv3                 [128, 512, 1, 1]   [1, 512, 32, 24]
60_layer2.2.BatchNorm2d_bn3                         [512]   [1, 512, 32, 24]
61_layer2.2.ReLU_relu                                   -   [1, 512, 32, 24]
62_layer2.3.Conv2d_conv1                 [512, 128, 1, 1]   [1, 128, 32, 24]
63_layer2.3.BatchNorm2d_bn1                         [128]   [1, 128, 32, 24]
64_layer2.3.ReLU_relu                                   -   [1, 128, 32, 24]
65_layer2.3.Conv2d_conv2                 [128, 128, 3, 3]   [1, 128, 32, 24]
66_layer2.3.BatchNorm2d_bn2                         [128]   [1, 128, 32, 24]
67_layer2.3.ReLU_relu                                   -   [1, 128, 32, 24]
68_layer2.3.Conv2d_conv3                 [128, 512, 1, 1]   [1, 512, 32, 24]
69_layer2.3.BatchNorm2d_bn3                         [512]   [1, 512, 32, 24]
70_layer2.3.ReLU_relu                                   -   [1, 512, 32, 24]
71_layer3.0.Conv2d_conv1                 [512, 256, 1, 1]   [1, 256, 32, 24]
72_layer3.0.BatchNorm2d_bn1                         [256]   [1, 256, 32, 24]
73_layer3.0.ReLU_relu                                   -   [1, 256, 32, 24]
74_layer3.0.Conv2d_conv2                 [256, 256, 3, 3]   [1, 256, 16, 12]
75_layer3.0.BatchNorm2d_bn2                         [256]   [1, 256, 16, 12]
76_layer3.0.ReLU_relu                                   -   [1, 256, 16, 12]
77_layer3.0.Conv2d_conv3                [256, 1024, 1, 1]  [1, 1024, 16, 12]
78_layer3.0.BatchNorm2d_bn3                        [1024]  [1, 1024, 16, 12]
79_layer3.0.downsample.Conv2d_0         [512, 1024, 1, 1]  [1, 1024, 16, 12]
80_layer3.0.downsample.BatchNorm2d_1               [1024]  [1, 1024, 16, 12]
81_layer3.0.ReLU_relu                                   -  [1, 1024, 16, 12]
82_layer3.1.Conv2d_conv1                [1024, 256, 1, 1]   [1, 256, 16, 12]
83_layer3.1.BatchNorm2d_bn1                         [256]   [1, 256, 16, 12]
84_layer3.1.ReLU_relu                                   -   [1, 256, 16, 12]
85_layer3.1.Conv2d_conv2                 [256, 256, 3, 3]   [1, 256, 16, 12]
86_layer3.1.BatchNorm2d_bn2                         [256]   [1, 256, 16, 12]
87_layer3.1.ReLU_relu                                   -   [1, 256, 16, 12]
88_layer3.1.Conv2d_conv3                [256, 1024, 1, 1]  [1, 1024, 16, 12]
89_layer3.1.BatchNorm2d_bn3                        [1024]  [1, 1024, 16, 12]
90_layer3.1.ReLU_relu                                   -  [1, 1024, 16, 12]
91_layer3.2.Conv2d_conv1                [1024, 256, 1, 1]   [1, 256, 16, 12]
92_layer3.2.BatchNorm2d_bn1                         [256]   [1, 256, 16, 12]
93_layer3.2.ReLU_relu                                   -   [1, 256, 16, 12]
94_layer3.2.Conv2d_conv2                 [256, 256, 3, 3]   [1, 256, 16, 12]
95_layer3.2.BatchNorm2d_bn2                         [256]   [1, 256, 16, 12]
96_layer3.2.ReLU_relu                                   -   [1, 256, 16, 12]
97_layer3.2.Conv2d_conv3                [256, 1024, 1, 1]  [1, 1024, 16, 12]
98_layer3.2.BatchNorm2d_bn3                        [1024]  [1, 1024, 16, 12]
99_layer3.2.ReLU_relu                                   -  [1, 1024, 16, 12]
100_layer3.3.Conv2d_conv1               [1024, 256, 1, 1]   [1, 256, 16, 12]
101_layer3.3.BatchNorm2d_bn1                        [256]   [1, 256, 16, 12]
102_layer3.3.ReLU_relu                                  -   [1, 256, 16, 12]
103_layer3.3.Conv2d_conv2                [256, 256, 3, 3]   [1, 256, 16, 12]
104_layer3.3.BatchNorm2d_bn2                        [256]   [1, 256, 16, 12]
105_layer3.3.ReLU_relu                                  -   [1, 256, 16, 12]
106_layer3.3.Conv2d_conv3               [256, 1024, 1, 1]  [1, 1024, 16, 12]
107_layer3.3.BatchNorm2d_bn3                       [1024]  [1, 1024, 16, 12]
108_layer3.3.ReLU_relu                                  -  [1, 1024, 16, 12]
109_layer3.4.Conv2d_conv1               [1024, 256, 1, 1]   [1, 256, 16, 12]
110_layer3.4.BatchNorm2d_bn1                        [256]   [1, 256, 16, 12]
111_layer3.4.ReLU_relu                                  -   [1, 256, 16, 12]
112_layer3.4.Conv2d_conv2                [256, 256, 3, 3]   [1, 256, 16, 12]
113_layer3.4.BatchNorm2d_bn2                        [256]   [1, 256, 16, 12]
114_layer3.4.ReLU_relu                                  -   [1, 256, 16, 12]
115_layer3.4.Conv2d_conv3               [256, 1024, 1, 1]  [1, 1024, 16, 12]
116_layer3.4.BatchNorm2d_bn3                       [1024]  [1, 1024, 16, 12]
117_layer3.4.ReLU_relu                                  -  [1, 1024, 16, 12]
118_layer3.5.Conv2d_conv1               [1024, 256, 1, 1]   [1, 256, 16, 12]
119_layer3.5.BatchNorm2d_bn1                        [256]   [1, 256, 16, 12]
120_layer3.5.ReLU_relu                                  -   [1, 256, 16, 12]
121_layer3.5.Conv2d_conv2                [256, 256, 3, 3]   [1, 256, 16, 12]
122_layer3.5.BatchNorm2d_bn2                        [256]   [1, 256, 16, 12]
123_layer3.5.ReLU_relu                                  -   [1, 256, 16, 12]
124_layer3.5.Conv2d_conv3               [256, 1024, 1, 1]  [1, 1024, 16, 12]
125_layer3.5.BatchNorm2d_bn3                       [1024]  [1, 1024, 16, 12]
126_layer3.5.ReLU_relu                                  -  [1, 1024, 16, 12]
127_layer4.0.Conv2d_conv1               [1024, 512, 1, 1]   [1, 512, 16, 12]
128_layer4.0.BatchNorm2d_bn1                        [512]   [1, 512, 16, 12]
129_layer4.0.ReLU_relu                                  -   [1, 512, 16, 12]
130_layer4.0.Conv2d_conv2                [512, 512, 3, 3]     [1, 512, 8, 6]
131_layer4.0.BatchNorm2d_bn2                        [512]     [1, 512, 8, 6]
132_layer4.0.ReLU_relu                                  -     [1, 512, 8, 6]
133_layer4.0.Conv2d_conv3               [512, 2048, 1, 1]    [1, 2048, 8, 6]
134_layer4.0.BatchNorm2d_bn3                       [2048]    [1, 2048, 8, 6]
135_layer4.0.downsample.Conv2d_0       [1024, 2048, 1, 1]    [1, 2048, 8, 6]
136_layer4.0.downsample.BatchNorm2d_1              [2048]    [1, 2048, 8, 6]
137_layer4.0.ReLU_relu                                  -    [1, 2048, 8, 6]
138_layer4.1.Conv2d_conv1               [2048, 512, 1, 1]     [1, 512, 8, 6]
139_layer4.1.BatchNorm2d_bn1                        [512]     [1, 512, 8, 6]
140_layer4.1.ReLU_relu                                  -     [1, 512, 8, 6]
141_layer4.1.Conv2d_conv2                [512, 512, 3, 3]     [1, 512, 8, 6]
142_layer4.1.BatchNorm2d_bn2                        [512]     [1, 512, 8, 6]
143_layer4.1.ReLU_relu                                  -     [1, 512, 8, 6]
144_layer4.1.Conv2d_conv3               [512, 2048, 1, 1]    [1, 2048, 8, 6]
145_layer4.1.BatchNorm2d_bn3                       [2048]    [1, 2048, 8, 6]
146_layer4.1.ReLU_relu                                  -    [1, 2048, 8, 6]
147_layer4.2.Conv2d_conv1               [2048, 512, 1, 1]     [1, 512, 8, 6]
148_layer4.2.BatchNorm2d_bn1                        [512]     [1, 512, 8, 6]
149_layer4.2.ReLU_relu                                  -     [1, 512, 8, 6]
150_layer4.2.Conv2d_conv2                [512, 512, 3, 3]     [1, 512, 8, 6]
151_layer4.2.BatchNorm2d_bn2                        [512]     [1, 512, 8, 6]
152_layer4.2.ReLU_relu                                  -     [1, 512, 8, 6]
153_layer4.2.Conv2d_conv3               [512, 2048, 1, 1]    [1, 2048, 8, 6]
154_layer4.2.BatchNorm2d_bn3                       [2048]    [1, 2048, 8, 6]
155_layer4.2.ReLU_relu                                  -    [1, 2048, 8, 6]
156_deconv_layers.ConvTranspose2d_0     [256, 2048, 4, 4]   [1, 256, 16, 12]
157_deconv_layers.BatchNorm2d_1                     [256]   [1, 256, 16, 12]
158_deconv_layers.ReLU_2                                -   [1, 256, 16, 12]
159_deconv_layers.ConvTranspose2d_3      [256, 256, 4, 4]   [1, 256, 32, 24]
160_deconv_layers.BatchNorm2d_4                     [256]   [1, 256, 32, 24]
161_deconv_layers.ReLU_5                                -   [1, 256, 32, 24]
162_deconv_layers.ConvTranspose2d_6      [256, 256, 4, 4]   [1, 256, 64, 48]
163_deconv_layers.BatchNorm2d_7                     [256]   [1, 256, 64, 48]
164_deconv_layers.ReLU_8                                -   [1, 256, 64, 48]
165_final_layer                           [256, 17, 1, 1]    [1, 17, 64, 48]
...具体参数信息略去
```

输入是$batch\_size*256*192*3$，输出$64*48*17$的heatmap。

去除后三个反卷积层之后，整体结构和ResNet-50一样，如将ResNet-50的输入由[batch_size,3,224,224]变为[batch_size,3,256,198]，则二者中间的feature map的尺寸完全一致。



运行指令：

```shell
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar
```

或者使用自己训好的模型测试：

```shell
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file output/coco/pose_resnet_50/256x192_d256x3_adam_lr1e-3/final_state.pth.tar
```

测试结果如下（自己训好的模型）：

```python
DONE (t=0.06s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.724
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.915
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.804
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.699
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.769
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.756
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.927
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.824
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.724
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.805
=> coco eval results saved to output/coco/pose_resnet_50/256x192_d256x3_adam_lr1e-3/results/keypoints_val2017_results.pkl
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| 256x192_pose_resnet_50_d256d256d256 | 0.724 | 0.915 | 0.804 | 0.699 | 0.769 | 0.756 | 0.927 | 0.824 | 0.724 | 0.805 |
```

如果此时将参数.yaml文件中的TEST.USE_GT_BBOX改为false的话，那么测试过程就是没有使用数据集box的groudtruth，直接衡量两个阶段的整体效果，效果会比直接使用第一阶段的gt要差一点。运行结果如下：

```python
DONE (t=0.25s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.703
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.887
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.777
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.668
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.772
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.762
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.927
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.831
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.718
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.825
=> coco eval results saved to output/coco/pose_resnet_50/256x192_d256x3_adam_lr1e-3/results/keypoints_val2017_results.pkl
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| 256x192_pose_resnet_50_d256d256d256 | 0.703 | 0.887 | 0.777 | 0.668 | 0.772 | 0.762 | 0.927 | 0.831 | 0.718 | 0.825 |
```





##### 二、论文2

论文：Deep High-Resolution Representation Learning for Human Pose Estimation  

repo：https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

数据集：MSCOCO2017

训练代码和训练流程和Simple Baseline基本一致，测试过程可以选择是否使用测试集box的groundtruth。

训练指令：

```shell
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
```

测试指令1（不使用box的groundtruth，以及使用官方给出的模型）：

```shell
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    TEST.USE_GT_BBOX False
```

测试结果：

```python
DONE (t=0.25s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.744
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.905
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.819
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.708
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.810
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.798
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.942
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.865
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.757
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.858
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hrnet | 0.744 | 0.905 | 0.819 | 0.708 | 0.810 | 0.798 | 0.942 | 0.865 | 0.757 | 0.858 |
```



测试指令2（使用box的groundtruth，以及使用官方给出的模型）：

```shell
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth
```

测试结果：

```python
DONE (t=0.07s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.765
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.935
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.837
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.739
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.808
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.793
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.945
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.858
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.762
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.841
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hrnet | 0.765 | 0.935 | 0.837 | 0.739 | 0.808 | 0.793 | 0.945 | 0.858 | 0.762 | 0.841 |
```



测试指令3（不使用box的groundtruth，以及使用自己训好的模型）：

```shell
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE output/coco/pose_hrnet/w32_256x192_adam_lr1e-3/final_state.pth \
    TEST.USE_GT_BBOX False
```

测试结果：

```python
DONE (t=0.38s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.745
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.900
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.820
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.710
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.812
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.798
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.938
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.864
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.757
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.858
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hrnet | 0.745 | 0.900 | 0.820 | 0.710 | 0.812 | 0.798 | 0.938 | 0.864 | 0.757 | 0.858 |
```



测试指令4（使用box的groundtruth，以及使用自己训好的模型）：

```shell
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE output/coco/pose_hrnet/w32_256x192_adam_lr1e-3/final_state.pth
```

测试结果：

```python
DONE (t=0.08s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.767
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.936
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.838
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.739
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.811
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.795
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.943
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.860
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.764
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.842
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hrnet | 0.767 | 0.936 | 0.838 | 0.739 | 0.811 | 0.795 | 0.943 | 0.860 | 0.764 | 0.842 |
```



**注：默认的参数TEST.USE_GT_BBOX为true，即在运行训练脚本train.py的时候，train和val的过程都是默认使用的数据集box的gt（groundtruth）结果，个人理解是为了避免top-down方法中第一阶段box提取器好坏对第二阶段的关键点检测效果产生影响。在运行测试脚本test.py的时候，可以选择是否使用box的gt结果，从结果中可以看出，使用box的gt结果的效果要比不使用gt的效果好，因为第一阶段faster-rcnn检测人的box会产生一定的偏差导致整体检测结果效果不好。**



##### 三、论文3

论文：Simple and Lightweight Human Pose Estimation

repo：https://github.com/zhang943/lpn-pytorch

数据集：MSCOCO2017

（注：这篇文章实际上是在SimpleBaseline的基础上进行改进的，创新有三点：（1）将backbone换成轻量级网络；（2）使用迭代训练策略，避免了预训练；（3）改变了关键点坐标的获取方式，提出一种新的计算方法。）

SimpleBaseline中的heatmap是一个高斯卷积核，公式如下：
$$
H_k(x,y)=e^{-\frac{(x-x_k)^2+(y-y_k)^2}{2\sigma^2}}
$$
没有归一化（因为需要中心元素为1），通过获取高斯核最大值的位置作为关键点的位置。

有学者提出使用$Soft-Argmax$归一化方法，公式如下：
$$
S_k(x,y)=\frac{e^{H_k(x,y)}}{\sum_x\sum_ye^{H_k(x,y)}}
$$
由于$e^0=1$，这种归一化方法在heatmap上如果有很多0的情况下，会导致中心元素的值非常小，这篇文章提出了$\beta-Soft-Argmax$归一化方法，公式如下：
$$
S_k(x,y)=\frac{e^{\beta H_k(x,y)}}{\sum_x\sum_ye^{\beta H_k(x,y)}}
$$
再使用和$Soft-Argmax$一样的获取关键点坐标的方式：
$$
\hat{x}=\sum S_k\circ W_x\\
\hat{y}=\sum S_k\circ W_y
$$
其中$W_x，W_y$是常数矩阵，$\circ$是逐像素点乘。



该论文的代码是在HRNet开源代码的基础上开发的，并没有全部开源，只开源了测试代码。按照和HRNet的repo中的data复制一份到lpn下，同时创建一个models文件夹，放置训好的lpn模型，即可运行测试脚本。



消融实验结果（lpn50为例）：

1.仅使用GC-block，不使用迭代训练策略以及$\beta-soft-argmax$，仅训练stage0的150个epoch，lpn50,101,152的mAP分别提升为（64.4%->66.9%），（67.8%->68.9%），（69.0->69.4）；

2.使用GC-block，迭代训练策略，mAP变化为：

stage0：66.9%，

stage1：67.73%，

stage2：68.12%，

stage3：68.28%，

stage4：68.69%，

stage5：68.89%，

stage6：68.92%；

3.使用GC-block，迭代训练策略以及$\beta-soft-Argmax$，通过将$\beta$设置为160时得到最优结果，mAP为69.1%，使用一般的Argmax结果为68.9%。



网络代码解读：

```python
import torch
import torch.nn as nn

BN_MOMENTUM = 0.1


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        hidden_dim = in_channels // reduction if in_channels // reduction >= 16 else 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class GCBlock(nn.Module):

    def __init__(self, inplanes, planes, pool, fusions):
        super(GCBlock, self).__init__()
        assert pool in ['avg', 'att']
        #GC-block的最终融合方式有相加或者相乘
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            #如果参数是'att'，那就是采用获取attention的方式从C*H*W到C*1*1
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            #如果参数是'avg'，那就是和SE-block一样，C*H*W均值池化到C*1*1
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class LW_Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention='GC'):
        super(LW_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if attention == 'SE':
            self.att = SELayer(planes * self.expansion)
        elif attention == 'GC':
            out_planes = planes * self.expansion // 16 if planes * self.expansion // 16 >= 16 else 16
            self.att = GCBlock(planes * self.expansion, out_planes, 'att', ['channel_add'])
        else:
            self.att = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.att is not None:
            out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

```python
import os
import logging
import math
import torch
import torch.nn as nn
from .lightweight_modules import LW_Bottleneck

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class LPN(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        super(LPN, self).__init__()
        extra = cfg.MODEL.EXTRA

        self.inplanes = 64
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.attention = extra.get('ATTENTION')

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS, # 2
            extra.NUM_DECONV_FILTERS, # 256, 256
            extra.NUM_DECONV_KERNELS, # 4, 4
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1], # 256
            out_channels=cfg.MODEL.NUM_JOINTS, # 17
            kernel_size=extra.FINAL_CONV_KERNEL, # 1
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        #每个阶段的第一个bottleneck，shortcut使用downsample
        layers.append(block(self.inplanes, planes, stride, downsample, self.attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention=self.attention))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i) # kernel=4,padding=1,output_padding=0

            planes = num_filters[i] # 256
            # 组转置卷积，先在行列中间插值0，然后再进行阻卷积，同样可以减少很多参数
            # math.gcd(x,y)，返回x,y的最大公约数，即组数是输入通道数和输出通道数的最大公约数
            layers.extend([
                nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel,
                                   stride=2, padding=padding, output_padding=output_padding,
                                   groups=math.gcd(self.inplanes, planes), bias=self.deconv_with_bias),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ])
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = self.deconv_layers(x)
        x = self.final_layer(features)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    50: (LW_Bottleneck, [3, 4, 6, 3]),
    101: (LW_Bottleneck, [3, 4, 23, 3]),
    152: (LW_Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = LPN(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
```



测试指令：

```shell
python test.py \
    --cfg experiments/coco/lpn/lpn50_256x192_gd256x2_gc.yaml
```

测试结果：

```python
DONE (t=0.28s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.691
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.881
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.766
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.659
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.757
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.749
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.923
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.818
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.707
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.810
|    Arch    |    AP    |  Ap .5   |  AP .75  |  AP (M)  |  AP (L)  |    AR    |  AR .5   |  AR .75  |  AR (M)  |  AR (L)  |
|    lpn     |  0.6907  |  0.8813  |  0.7661  |  0.6591  |  0.7572  |  0.7488  |  0.9232  |  0.8175  |  0.7068  |  0.8095  |
```

使用测试集的gt结果进行测试结果：（和Simple Baseline以及HRNet一样，效果会好一点）

```python
DONE (t=0.07s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.712
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.916
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.784
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.687
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.752
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.744
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.924
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.810
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.713
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.791
|    Arch    |    AP    |  Ap .5   |  AP .75  |  AP (M)  |  AP (L)  |    AR    |  AR .5   |  AR .75  |  AR (M)  |  AR (L)  |
|    lpn     |  0.7115  |  0.9155  |  0.7844  |  0.6871  |  0.7524  |  0.7438  |  0.9243  |  0.8100  |  0.7129  |  0.7905  |
```



测试指令：

```shell
python test.py \
    --cfg experiments/coco/lpn/lpn101_256x192_gd256x2_gc.yaml
```

使用GT：

```python
DONE (t=0.07s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.727
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.916
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.805
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.701
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.771
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.758
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.929
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.827
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.727
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.805
|    Arch    |    AP    |  Ap .5   |  AP .75  |  AP (M)  |  AP (L)  |    AR    |  AR .5   |  AR .75  |  AR (M)  |  AR (L)  |
|    lpn     |  0.7267  |  0.9155  |  0.8055  |  0.7013  |  0.7706  |  0.7582  |  0.9293  |  0.8273  |  0.7272  |  0.8055  |
```

不使用GT：

```python
DONE (t=0.30s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.704
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.886
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.781
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.672
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.772
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.762
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.929
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.831
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.720
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.822
|    Arch    |    AP    |  Ap .5   |  AP .75  |  AP (M)  |  AP (L)  |    AR    |  AR .5   |  AR .75  |  AR (M)  |  AR (L)  |
|    lpn     |  0.7043  |  0.8864  |  0.7811  |  0.6717  |  0.7724  |  0.7622  |  0.9287  |  0.8306  |  0.7205  |  0.8223  |
```



测试指令：

```shell
python test.py \
    --cfg experiments/coco/lpn/lpn152_256x192_gd256x2_gc.yaml
```

使用GT：

```python

```

不使用GT：

```python
DONE (t=0.31s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.710
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.892
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.786
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.678
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.768
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.933
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.834
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.726
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.827
|    Arch    |    AP    |  Ap .5   |  AP .75  |  AP (M)  |  AP (L)  |    AR    |  AR .5   |  AR .75  |  AR (M)  |  AR (L)  |
|    lpn     |  0.7099  |  0.8915  |  0.7858  |  0.6783  |  0.7767  |  0.7677  |  0.9331  |  0.8338  |  0.7259  |  0.8274  |
```



##### 四、论文4

论文：Towards Accurate Multi-person Pose Estimation in the Wild

以往的Ground Truth的构建主要有两种思路，Coordinate和Heatmap。

（1）Coordinate将关键点坐标作为最后网络需要回归的目标，这种情况下可以直接得到每个坐标点的直接位置信息；

（2）Heatmap将**每一个关键点**用一个概率图来表示（如有17个关键点，则生成的概率图有17个通道，每个通道的概率图对应一个关键点），对概率图中的每个像素位置都给一个概率，表示该点属于对应类别关键点的概率。显然，距离关键点位置越近的像素点的概率越接近1，距离关键点越远的像素点的概率越接近0，可以用高斯函数进行模拟概率分布。

对于两种Ground Truth的差别，Coordinate网络在本质上来说，需要回归的是每个关键点的一个相对于图片的offset，而长距离offset在实际学习过程中是很难回归的，误差较大，同时在训练中的过程，提供的监督信息较少，整个网络的收敛速度较慢；Heatmap网络直接回归出每一类关键点的概率，在一定程度上每一个点都提供了监督信息，网络能够较快的收敛，同时对每一个像素位置进行预测能够提高关键点的定位精度，在可视化方面，Heatmap也要优于Coordinate，除此之外，实践证明，Heatmap确实要远优于Coordinate。



这篇论文将Heatmap和offset结合在一起构建Ground Truth。

这篇文章首先利用Faster-RCNN检测出包括人的box进行缩放裁减，将裁减之后的图像输入到网络ResNet101当中来生成Heatmaps和offset。假设关键点检测任务一共$K$个关键点，Heatmaps包含$K$个channel，一个关键点对应一个通道；offsets包含$2K$个通道，一个x方向上的偏移，一个y方向上的偏移。

Heatmaps构建：每个关键点对应一个通道，给定一个半径R，在距离关键点为R内，概率值为1，其余为0，将回归问题看成一个二分类问题；即：
$$
h_k(x_i)=1,||x_i-l_k||\leq R
$$


offset构建：每个关键点对应两个通道，分别记录像素点在x方向和y方向距离关键点的偏移量，即：
$$
F_k(x_i)=l_k-x_i
$$
卷积网络有两个输出，分别是Heatmaps损失和offsets损失：

Heatmaps损失通过sigmoid函数输出heatmap概率图，对应的损失函数是logistic losses，实际上就是交叉熵函数：
$$
L_h(\theta)=-\sum_{k=1}^{K}\sum_{i}[(h_k(x_i)ln(h'_k(x_i))+(1-h_k(x_i))ln(1-h'_k(x_i))]
$$
其中$h'_k(x_i)$是输出概率值，$h_k(x_i)$是真实标签值；

offsets损失函数如下：
$$
L_o(\theta)=\sum_{k=1}^{K}\sum_{i:||l_k-x_i||\leq R}H(||F'_k(x_i)-(l_k-x_i)||)
$$


其中$F'_k(x_i)$是实际输出，$F_k(x_i)=l_k-x_i$是真实标签值；

网络的损失函数是Heatmaps损失函数和offset损失函数的融合：
$$
L(\theta)=\lambda_hL_h(\theta)+\lambda_oL_o(\theta)
$$



##### 五、论文5

论文：Distribution-Aware Coordinate Representation for Human Pose Estimation  

repo：https://github.com/ilovepose/DarkPose

数据集：MSCOCO2017

这篇文章从关键点检测的编码和解码方法出发，提出DARK方法，有两个关键点：（1）基于泰勒展开的坐标解码；（2）无偏亚像素级别的坐标编码。



1.解码过程的改进

（1）标准的坐标解码方法（即HRNet或SimpleBaseline中的编码方法）

假设heatmap中最大值对应的坐标为$m$，次大值对应的坐标为$s$，则最终预测的结果为：
$$
p = m+0.25\frac{s-m}{||s-m||_2}\tag{1}
$$
意味着最终的预测结果是最大值的坐标向第二大值的坐标方向偏移0.25个像素（即亚像素级别）。但实际上代码只是取了相邻的四个像素来计算偏移方向，x方向上左边的像素值大于右边的像素值，则向左边偏移0.25个像素；y方向上面的像素值大于下面的像素值，则向上偏移0.25个像素。代码如下：

```python
def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px] - hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height])

    return preds, maxvals
```

然后再转换回原始图像空间。

公式（1）是为了补偿图像分辨率下采样造成的定量损失。也就是说，预测的heatmap中预测的最大值只是一个粗糙的位置，并不是关键点真正的坐标位置。这一标准方法在设计中缺乏直觉和解释，还没有专门的研究进行改进。这篇文章围绕这一点提出了一种偏移估计的理论方法，最终得到更精确的人体姿态估计。



（2）改进的解码方法

提出的解码方法通过探索预测的heatmap的**分布**来寻找潜在的最大值，这与上面依赖手工设计的偏移量预测的标准方法有很大的不同，后者几乎没有设计理由和基本原理。

假设离散的heatmap服从二维高斯分布：
$$
G(x,\mu,\Sigma)=\frac{1}{2\pi|\Sigma|^\frac{1}{2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
$$
同时协方差矩阵如下：
$$
\Sigma=
\left[\begin{matrix}
\sigma^2 & 0 \\
0 & \sigma^2
\end{matrix}\right]
$$
对高斯分布取对数：
$$
f(x;\mu,\Sigma)=ln(G)=-ln(2\pi)-\frac{1}{2}ln(|\Sigma|)-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
$$
对$f$求一阶导：
$$
f'(x)|_{x=\mu}=-\Sigma^{-1}(x-\mu)=0
$$
假设预测的heatmap上最大值的位置为$m$（$f'(m)=-\Sigma^{-1}(m-\mu)$），将$f$在$m$处多元泰勒展开，并代入真实中心$\mu$，得到：
$$
f(\mu)=f(m)+f'(m)^T(\mu-m)+\frac{1}{2}(\mu-m)^Tf''(m)(\mu-m)
$$
这里$f''(m)$是二维高斯分布的hessian阵，易求$f''(m)=-\Sigma^{-1}$，则可以推导出真实值$\mu$与预测heatmap的最大值$m$之间的偏移：
$$
\begin{align*}
\mu&=m-(m-\mu)\\
&=m-[-\Sigma f'(m)]\\
&=m-[f''(m)]^{-1}f'(m)
\end{align*}
$$
这里将$f$在$m$处进行多元泰勒展开得到$m$和$\mu$之间的关系，误差取决于$m$和$\mu$有多接近，并展开到进行几阶导。

**误差分析：二维高斯分布取对数后是一个二元二次函数，是一个凸函数，在使用牛顿法迭代求解极值时具有二次终止性质，即可以一步迭代到极值处。并且二元二次函数在最大值点$m$处做泰勒展开，只展开到第二项时是没有误差的（因为三阶导，即第三项为0）。整个过程的误差出现在利用像素离散值计算梯度和hessian的过程中。假设预测的heatmap都是非常准确的，那么在一个二元二次函数的某个方向上，利用左右同间距的像素点去计算中间一点的梯度，也是没有误差的（因为二次函数求导之后是线性的）。但当预测的值并不是完全准确的，则计算梯度会有误差，进而导致二阶导hessian矩阵也存在误差。这一步可能是后续改进的点。**





所以该解码方法分三步进行：

（1）预测的heatmap在最大值附近有很多“峰”，所以先使用一个高斯核进行平滑，再进行归一化得到处理后的预测heatmap；

（2）使用推导的偏移公式计算真正的高斯中心；

（3）将计算的高斯中心再返回到原始图像中；



2.编码过程的改进

（略）



测试指令（官方模型）：

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/w32_256×192.pth \
    TEST.USE_GT_BBOX False
```

官方模型测试结果：

```python
DONE (t=0.27s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.756
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.905
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.821
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.718
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.828
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.808
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.944
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.866
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.764
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.871
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hrnet | 0.756 | 0.905 | 0.821 | 0.718 | 0.828 | 0.808 | 0.944 | 0.866 | 0.764 | 0.871 |
```



测试指令（官方模型）：

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/w32_384×288.pth \
    TEST.USE_GT_BBOX False
```

官方模型测试结果：

```python
DONE (t=0.35s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.766
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.907
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.828
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.727
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.839
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.815
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.942
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.870
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.771
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.880
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hrnet | 0.766 | 0.907 | 0.828 | 0.727 | 0.839 | 0.815 | 0.942 | 0.870 | 0.771 | 0.880 |
```





训练：

```
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
```



测试指令：

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE output/coco/pose_hrnet/w32_256x192_adam_lr1e-3/model_best.pth \
    TEST.USE_GT_BBOX False
```

测试结果：

```python
DONE (t=0.32s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.755
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.903
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.822
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.717
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.827
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.807
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.942
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.867
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.763
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.871
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hrnet | 0.755 | 0.903 | 0.822 | 0.717 | 0.827 | 0.807 | 0.942 | 0.867 | 0.763 | 0.871 |
```

和官方给出的模型效果基本一致。







##### 六、论文6

论文：The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation  

repo：https://github.com/HuangJunJie2017/UDP-Pose

数据集：MSCOCO2017

这篇文章的方法是model-agnostic approach(即与模型无关的方法)，改进的是数据预处理时的方法。现阶段的预处理方法主要有以下三个方面：

（1）Data Transformation

Simple Baseline和HRNet在inference时，使用像素来衡量图像的尺寸，并使用flipping的策略，代码如下：

```python
if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy() # 第三维翻转，相当于对原图做水平翻转
                input_flipped = torch.from_numpy(input_flipped).cuda()
                # print(input_flipped.size()) # torch.size([size, 3, 256, 192])
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                # print(output_flipped.size()) # torch.size([size, 17, 64, 48])
                
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0
                    # print(output_flipped.size())

                output = (output + output_flipped) * 0.5
                #print(output.size()) # torch.size([size, 17, 64, 48])
                # 首先测试了原图a得到结果b，然后测试了翻转的原图a_，再把翻转的原图测试结果b_又翻转了回来得到b'，最后取了原图结果b和翻转结果b'的均值
```

```python
def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1] # 通道内进行水平翻转
    # 通道间根据matched_parts进行翻转
    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp
    return output_flipped
```

首先测试原图$a$的结果$b$，然后测试了翻转的原图$a\_$，再测试翻转的原图$a\_$得到测试结果$b\_$，再对$b\_$使用flip_back函数得到$b'$，最后对原图测试结果$b$以及翻转结果$b'$取均值得到inference的最终结果。值得注意的是flip_back函数，在flip_back函数中，不仅仅在17个关键点通道内进行水平翻转，还要对通道间的matched_parts进行对应的配对调换，比如右眼对应的通道和左眼对应的通道进行调换。

在水平翻转之后，Simple Baseline和HRNet还有个水平位移一个像素的操作，代码如下：

```python
# feature is not aligned, shift flipped heatmap for higher accuracy
if config.TEST.SHIFT_HEATMAP:
    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
```

（原因见下面的公式推导）



（2）Data Augmentation

数据增强操作，如旋转，翻转，规定尺寸等。



（3）Encoding-Decoding

Encoding-Decoding指的是关键点的坐标和heatmaps之间的转换。训练阶段，使用高斯分布将ground truth编码为heatmap；预测阶段将预测的hetamap结果解码为关键点坐标。这种编码解码方法比直接回归坐标效果要好，但是引入了系统误差。将回归和分类的编解码方式结合起来比单纯的分类要好（论文4）。

这篇文章从角度（1）（3）进行改进，提出无偏数据处理策略来提高姿态估计器的性能。



**标准深度学习方法（即SimpleBaseline和HRNet）的Data Augmentation：**

（见图Data_Augmentation，文章UDP的附录部分）

关键点检测的top-down方法的数据增强部分，会将目标的box旋转变换到网络的输入尺寸，如$^pw_i*^ph_i$。这一过程可以划分为几个步骤：

1）将原图的左上角坐标原点移到box的中心，即box中心作为原图新的坐标中心，坐标变换如下：
$$
\left[\begin{matrix}
   x_1\\
   y_1\\
   1
\end{matrix}\right]=
\left[\begin{matrix}
   1 & 0 & -^sx_b \\
   0 & -1 & ^sy_b \\
   0 & 0 & 1
\end{matrix}\right]*
\left[\begin{matrix}
   x\\
   y\\
   1
\end{matrix}\right]
$$
2）在box为坐标原点的坐标系下，将原图旋转$\theta$角度，坐标变换如下：
$$
\left[\begin{matrix}
   x_2\\
   y_2\\
   1
\end{matrix}\right]=
\left[\begin{matrix}
   cos\theta & sin\theta & 0 \\
   -sin\theta & cos\theta & 0 \\
   0 & 0 & 1
\end{matrix}\right]*
\left[\begin{matrix}
   x_1\\
   y_1\\
   1
\end{matrix}\right]
$$
3）将box中心的坐标原点移到box的左上角，坐标变换如下：
$$
\left[\begin{matrix}
   x_3\\
   y_3\\
   1
\end{matrix}\right]=
\left[\begin{matrix}
   1 & 0 & 0.5^sw_b \\
   0 & -1 & 0.5^sh_b \\
   0 & 0 & 1
\end{matrix}\right]*
\left[\begin{matrix}
   x_2\\
   y_3\\
   1
\end{matrix}\right]
$$
4）将box缩放到网络的输入尺寸大小，如$^pw_i*^ph_i$，坐标变换如下：
$$
\left[\begin{matrix}
   x_4\\
   y_4\\
   1
\end{matrix}\right]=
\left[\begin{matrix}
   \frac{^pw_i}{^sw_b} & 0 & 0 \\
   0 & \frac{^ph_i}{^sh_b} & 0 \\
   0 & 0 & 1
\end{matrix}\right]*
\left[\begin{matrix}
   x_3\\
   y_3\\
   1
\end{matrix}\right]
$$
综合以上4个步骤，得到UDP的附录部分的公式，即：

![Data_Augmentation](G:\Documents\sunzheng\Learning_SimpleBaseline_and_LightweightBaseling_for_Human_Pose_Estimation\code\Data_Augmentation.png)



数据增强代码（/lib/utils/transform.py）

```python
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])
    
    scale_tmp = scale * 200.0
    # print('scale_tmp:',scale_tmp)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180 # 角度转弧度
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift # 原图像第一个点选取的是box的中心
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5] # 目标图像第一个点选取的是输出256,192的中心
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    # 原图像和目标图像的第二个点个人认为是随便选取的
    # 原图像选取[0, src_w * -0.5]之后，逆时针旋转rot_rad，再加上中心坐标center
    # 目标图像直接选取[0, dst_w * -0.5]，再加上中心坐标[dst_w * 0.5, dst_h * 0.5]
    # 将这两个点作为原图像和目标图像对应的第二个点
    # 然后利用这两个点去计算第三个组成直角三角形的点
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    
    # 利用原图像和目标图像中的两个直角三角形来计算放射变换矩阵trans，2*3
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans
```

在原图坐标下以box为中心取了三个点，再在网络输入尺寸图的坐标下，以网络输入尺寸图为中心取了三个点，来计算放射变换矩阵，实际上就是上图中矩阵，只不过方式不一样。（待考证，需要查看cv2.getAffineTransform源码）



**标准深度学习方法（即SimpleBaseline和HRNet）的Data Transformation**

（下面的符号，凡是有$\hat{}$符号的均是预测的结果，如$\hat{k}$表示网络预测的关键点坐标，$k$表示标签的关键点坐标）

假设生成的ground truth heatmap尺寸为$(^pw_o,^ph_o)$，与网络输入尺寸之间的比例关系为：$s=\frac{^pwi}{^pw_o}=\frac{^ph_i}{^ph_o}$，且坐标存在以下关系：
$$
^ok=\frac{1}{s}*^ik\tag{3}
$$
其中，$^ok$是网络输出的标签关键点坐标，$^ik$是网络输入的关键点坐标。

现在有一个网络的预测的关键点在图像输出空间的坐标为$^o\hat{k}$，其在图像输入空间坐标为$^i\hat{k}$，在原始图像空间坐标为$^s\hat{k}$，则有以下公式：
$$
^s\hat{k}=
\left[\begin{matrix}
\frac{^sw_b}{^pw_o}cos\theta & \frac{^sh_b}{^ph_o}sin\theta & -0.5^sw_bcos\theta-0.5^sh_bsin\theta+^sx_b \\
-\frac{^sw_b}{^pw_o}cos\theta & \frac{^sh_b}{^ph_o}sin\theta & 0.5^sw_bsin\theta-0.5^sh_bcos\theta+^sy_b \\
0 & 0 & 1
\end{matrix}\right]*
^o\hat{k}\tag{4}
$$
公式（4）不知道如何推导来的，和前面的公式（2）如何联系？（以下套用公式（2）得到$^s\hat{k}$和$^i\hat{k}$的关系，但是$^i\hat{k}$和$^o\hat{k}$并不仅仅是差个s因子的关系）
$$
^s\hat{k}=
\left[\begin{matrix}
\frac{^sw_b}{^pw_i}cos\theta & \frac{^sh_b}{^ph_i}sin\theta & -0.5^sw_bcons\theta-0.5^sh_bsin\theta+^sx_b \\ -\frac{^sw_b}{^pw_i}cos\theta & \frac{^sh_b}{^ph_i}sin\theta & 0.5^sw_bsin\theta-0.5^sh_bcos\theta+^sy_b \\   0 & 0 & 1
\end{matrix}\right]*^i\hat{k}
$$
下面写上个人理解该公式的版本：

现在有一个ground truth关键点在图像输出空间的坐标为$^ok$，其在图像输入空间坐标为$^ik$，在原始图像空间坐标为$^sk$，则有以下公式：
$$
^sk=
\left[\begin{matrix}
\frac{^sw_b}{^pw_o}cos\theta & \frac{^sh_b}{^ph_o}sin\theta & -0.5^sw_bcos\theta-0.5^sh_bsin\theta+^sx_b \\
-\frac{^sw_b}{^pw_o}cos\theta & \frac{^sh_b}{^ph_o}sin\theta & 0.5^sw_bsin\theta-0.5^sh_bcos\theta+^sy_b \\
0 & 0 & 1
\end{matrix}\right]*
^ok\tag{4}
$$
联系前面的公式（2）得到$^sk$和$^ik$的关系，且$^ik$和$^ok$之间还差个s因子），所以有：
$$
\begin{align*}
^sk&=
\left[\begin{matrix}
\frac{^sw_b}{^pw_i}cos\theta & \frac{^sh_b}{^ph_i}sin\theta & -0.5^sw_bcons\theta-0.5^sh_bsin\theta+^sx_b \\ -\frac{^sw_b}{^pw_i}cos\theta & \frac{^sh_b}{^ph_i}sin\theta & 0.5^sw_bsin\theta-0.5^sh_bcos\theta+^sy_b \\   0 & 0 & 1
\end{matrix}\right]*^ik\\
&=\left[\begin{matrix}
\frac{^sw_b}{^pw_i}cos\theta & \frac{^sh_b}{^ph_i}sin\theta & -0.5^sw_bcons\theta-0.5^sh_bsin\theta+^sx_b \\ -\frac{^sw_b}{^pw_i}cos\theta & \frac{^sh_b}{^ph_i}sin\theta & 0.5^sw_bsin\theta-0.5^sh_bcos\theta+^sy_b \\   0 & 0 & 1
\end{matrix}\right]*\left[\begin{matrix}
\frac{^pw_i}{^pw_o} & 0 & 0\\
0 & \frac{^ph_i}{^ph_o} & 0\\
0 & 0 & 1
\end{matrix}\right]*^ok\\
&=\left[\begin{matrix}
\frac{^sw_b}{^pw_o}cos\theta & \frac{^sh_b}{^ph_o}sin\theta & -0.5^sw_bcos\theta-0.5^sh_bsin\theta+^sx_b \\
-\frac{^sw_b}{^pw_o}cos\theta & \frac{^sh_b}{^ph_o}sin\theta & 0.5^sw_bsin\theta-0.5^sh_bcos\theta+^sy_b \\
0 & 0 & 1
\end{matrix}\right]*
^ok
\end{align*}
$$

**（现在开始推导为何SimpleBaseline和HRNet的翻转策略方法中会有误差）**

假设一个关键点坐标在图像输入空间的坐标为$^ik$，在输入图像翻转之后的坐标为$^{if}k$，则有以下关系：
$$
^{if}k=
\left[\begin{matrix}
-1 & 0 & ^pw_i-1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*
^ik\tag{4}
$$
根据公式（3），翻转之后的输入图像的关键点坐标和对应的ground truth之间也存在一个s因子的关系，即：
$$
^{of}k = \frac{1}{s}*^{if}k
$$
再将翻转之后的输入图像对应的ground truth再翻转过来记作$^ok_f$，则有以下关系：
$$
\begin{align*}
^ok_f &=
\left[\begin{matrix}
-1 & 0 & ^pw_0-1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*
^{of}k\\
&=\left[\begin{matrix}
-1 & 0 & ^pw_0-1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*\frac{1}{s}*^{if}k\\
&=\left[\begin{matrix}
-1 & 0 & ^pw_0-1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*\left[\begin{matrix}
\frac{^pw_o}{^pw_i} & 0 & 0 \\
0 & \frac{^pw_o}{^pw_i} & 0 \\
0 & 0 & 1
\end{matrix}\right]*\left[\begin{matrix}
-1 & 0 & ^pw_i-1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*^ik\\
&=\left[\begin{matrix}
1 & 0 & -\frac{^pw_i-^pw_o}{^pw_i} \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*\left[\begin{matrix}
\frac{^pw_o}{^pw_i} & 0 & 0 \\
0 & \frac{^pw_o}{^pw_i} & 0 \\
0 & 0 & 1
\end{matrix}\right]*^ik\\
&=\left[\begin{matrix}
1 & 0 & -\frac{s-1}{s} \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*^ok

\tag{5}
\end{align*}
$$
其中倒数第二个等号并不是使用矩阵交换律将后两个矩阵交换顺序，直接和第一个矩阵相乘得到。矩阵交换律在某些条件下成立，如：

```latex
1.任何矩阵乘以数量阵或者零矩阵，可以交换顺序；
2.方阵A,B满足AB=A+B，则AB=BA;
```

公式（5）中两个矩阵不满足交换的条件，公式（5）中倒数第二个等号左边三个矩阵假设为$ABC$，需要将$C$穿过$B$变为$AC'B$。当$C$右乘$B$（相当于对矩阵$B$做初等列变换），需要得到一个矩阵$C'$，使得$C'$左乘$B$（即对矩阵$B$初等行变换）的结果与列变换相同，用这个思路可以得到倒数第二个等号。

由于矩阵$C$可以通过一个单位阵经过如下操作得到：
$$
\begin{align*}
&1.c1*(-1);\\
&2.c3+c1*[-(^pw_i-1)]
\end{align*}
$$
所以矩阵$C$右乘$B$，相当于对矩阵$B$使用同样的变换（使用矩阵乘法可以直接得到，这里为了方便解释，使用列变换），得到：
$$
\left[\begin{matrix}
-\frac{^pw_o}{^pw_i} & 0 & \frac{^pw_o}{^pw_i}(^pw_i-1)\\
0 & \frac{^pw_o}{^pw_i} & 0\\
0 & 0 & 1
\end{matrix}\right]
$$
现在观察通过对矩阵$B$做什么样的行变换可以得到上述结果，可以观察出来操作如下：
$$
\begin{align*}
&1.r1*(-1);\\
&2.r1+r3*[\frac{^pw_o}{^pw_i}(^pw_i-1)]
\end{align*}
$$
故矩阵$C'$为：
$$
\left[\begin{matrix}
-1 & 0 & \frac{^pw_o}{^pw_i}(^pw_i-1)\\
0 & 1 & 0\\
0 & 0 & 1
\end{matrix}\right]
$$
再将矩阵$AC'$相乘，得到公式（5）中倒数第二个等号右边的结果。

从公式（5）中可以看出，输入图像翻转之后的ground truth再翻转(即$^ok_f$)和原图像的ground truth(即$^ok$)之间会有一个x轴方向上的偏移量$-\frac{s-1}{s}$。

（当$s=4$时，偏移量为-0.75，比较接近1个像素的长度，此时将翻转的结果右移一个单位，再和原图像的ground truth取平均会得到更加准确的结果，这也是SimpleBaseline和HRNet中inference位移一个像素操作的原因。）

（1）不位移，直接取$^ok_f$和$^ok$的均值，则最终预测结果为$\frac{^ok_f+^ok}{2}$，误差为$^oe=|-\frac{s-1}{2s}|$；

（2）将$^ok_f$位移一个右移单位，则：
$$
\begin{align*}
^ok_{f+}&=\left[\begin{matrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*^ok_f\\
&=\left[\begin{matrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*\left[\begin{matrix}
1 & 0 & -\frac{s-1}{s} \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*^ok\\
&=\left[\begin{matrix}
1 & 0 & \frac{1}{s} \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*^ok
\end{align*}
$$
从公式中可以看出，偏移量变为$\frac{1}{s}$，此时再取$^ok_{f+}$和$^ok$的均值，误差为$^oe'=|\frac{1}{2s}|$。

当$s>2$时，有$^oe'<^oe$，并且当heatmap的尺寸$(^pw_o,^ph_o)$固定时，输入图像的尺寸$(pw_i,ph_i)$越大，则$s$越大，误差越小，和实际的网络表现一致。

注意：此处推导和原文中符号有一点不一样，个人认为这样写更便于理解位移的意义，因为在实际网络预测时，输入图像和输出heatmap的坐标不能再用因子$s$来联系。

**改进的Data Transformation解决翻转策略导致的偏移问题**

这一部分文章引入了一个概念：采用unit length作为图像尺寸的衡量标准，用两个相邻像素在特定空间中的距离来定义。个人认为这一概念并不重要，针对翻转策略导致的偏移问题，实际上改变了网络输入图像和对应的网络输出gt heatmap之间的坐标映射倍数，标准倍数因子是$s=\frac{^pwi}{^pw_o}=\frac{^ph_i}{^ph_o}$，这里采用了一个另外的因子$t=\frac{^pwi-1}{^pw_o-1}=\frac{^ph_i-1}{^ph_o-1}$，使得偏移消失，得到无偏的翻转策略。
$$
\begin{align*}
^ok_f
&=
\left[\begin{matrix}
-1 & 0 & ^pw_0-1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*
^{of}k\\
&=\left[\begin{matrix}
-1 & 0 & ^pw_0-1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*\frac{1}{t}*^{if}k\\
&=\left[\begin{matrix}
-1 & 0 & ^pw_0-1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*\left[\begin{matrix}
\frac{^pw_o-1}{^pw_i-1} & 0 & 0 \\
0 & \frac{^pw_o-1}{^pw_i-1} & 0 \\
0 & 0 & 1
\end{matrix}\right]*\left[\begin{matrix}
-1 & 0 & ^pw_i-1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix}\right]*^ik\\
&=\left[\begin{matrix}
\frac{^pw_o-1}{^pw_i-1} & 0 & 0 \\
0 & \frac{^pw_o-1}{^pw_i-1} & 0 \\
0 & 0 & 1
\end{matrix}\right]*^ik\\
&=^ok
\tag{6}
\end{align*}
$$
总结一下：文章提出诸多概念，个人并不是很理解，尤其是用像素间的距离来定义图像的长度这一点。文章在翻转策略产生偏移这一个问题上，改变了标签的映射倍数因子（由$s$到$t$），消除了偏移。



**标准深度学习方法（即SimpleBaseline和HRNet）的Data Encoding-decoding**

在标准深度学习方法（即SimpleBaseline和HRNet）中，编码过程是用真实的关键点坐标来生成heatmap，使用二维高斯函数来初始化heatmap，生成的时候有一个四舍五入的取值过程，代码为（/lib/dataset/JointDataset.py的generate_target函数）：

```python
for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size # 比例因子,(192,256)/(48,64)=(4,4)，这一点导致后续flip测试的时候需要位移一个像素点
                
                # (mu_x,mu_y)是关键点坐标的高斯中心
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5) # 四舍五入，小于0.5取0，大于等于0.5取1
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
```

标准的解码过程使用argmax函数计算预测结果的heatmap中最大的值对应的坐标即为关键点坐标。

**误差分析：**

假设真实的关键点坐标在图像平面中均匀分布，坐标为$(m,n)$，四舍五入取值为$(R(m),R(n))$，会产生误差，求该误差的平均期望，只考虑简单区间$[0,0.5]$，则$m$~$U(0,0.5)$，误差的平均期望取值如下：
$$
\begin{align*}
E[|m-R(m)|]&=E[m]\\
&=\int_0^{0.5}mf(m)dm\\
&=\int_0^{0.5}\frac{m}{0.5}dm\\
&=\frac{1}{4}
\end{align*}
$$
所以当真实的关键点横坐标在$[0,0.5)$之间时，应当取为0.25而不是0；当关键点横坐标在$[0.5,1]$时，应当取为0.75而不是1，即：
$$
\hat{m}=\left\{
\begin{array}{rcl}
F(m)+0.25 & &if\,\,m-F(m)<0.5\\
C(m)-0.25 & &otherwise
\end{array}
\right.
$$
**改进的Data Encoding-decoding方法**

编码时不再使用二维高斯函数来初始化heatmap，在一定的半径内，heatmap的值为1，其余为0；同时加上$x$轴和$y$轴两个方向上的偏移量，解码时使用一个二维高斯核先对预测结果做卷积取最大值得到heatmap的关键点坐标，同时使用偏移量进行修正。具体公式见论文。



**消融实验结果**

![Ablation Study Results](G:\Documents\sunzheng\Learning_SimpleBaseline_and_LightweightBaseling_for_Human_Pose_Estimation\code\Ablation Study Results.png)

消融实验有如下结果：

1.不使用翻转策略情况下，使用DT会降低AP，使用ED会提升1.1AP，DT和ED一起用AP提升1.2；

2.在使用翻转策略情况下，使用SimpleBaseline和HRNet中的位移一个像素的操作，相比较于不位移AP提升2.3，再加上偏移的补偿$^oe(x)'$，可以再提升0.2；不用偏移，单纯使用$DT$，结果只比位移操作高0.1（后续解释）；DT和ED一起用，可以得到最高的表现。

解释：

由于数据增强都会使用水平翻转，所以还是需要使用位移操作或者DT，但这两种方法表现基本一致，所以改进应当从ED部分来进行。

这里再对位移操作和DT做个简单的对比来表现二者表现基本一致。由于翻转导致的误差由文章中的证明方法，当$s=\frac{256}{64}$时，误差为0.75，**再加上**坐标四舍五入操作，误差刚好为1个像素，所以SimpleBaseline和HRNet中的位移一个像素的操作效果好，相当于没有误差；在DT中，改变映射的因子$t=\frac{256-1}{64-1}$，使得翻转之后没有误差，两个结果同时再取四舍五入相当于两个结果之间还是没有误差，所以位移操作和DT的结果表现基本一致。



##### 七、论文7

论文：Rethinking on Multi-Stage Networks for Human Pose Estimation  

repo：https://github.com/megvii-detection/MSPN  









#### domain adaptation用于姿态估计

##### 一、论文1

论文：Cross-Domain Adaptation for Animal Pose Estimation，ICCV2019

repo：

1.思路：

（1）使用两个大规模标注的数据集（人体姿态标注数据集，和动物目标框标注数据集（coco中有）），以及一个小规模数据集（人工标注的动物姿态数据集），来实现动物的姿态估计；

（2）从人类数据的预训练的模型开始，提出一种weakly- and semi-supervised cross-domain adaptation方法；

（3）包括三个部分：feature extractor, domain discriminator, keypoint estimator

**feature extractor从输入数据中提取特征，在此基础上domain discriminator来判断特征是来自于哪个域的，keypoint estimator来预测关键点。**

（4）在WS-CDA之后，模型已经具备了一些动物的姿态知识。但它在一个特定的没有见过的动物类上仍然表现不佳，因为没有从这个类获得监督知识。针对这种情况，我们提出了一种名为“渐进伪标签优化”(Progressive Pseudo-label-based optimization, PPLO)的模型优化机制。利用当前模型选择的预测输出产生的伪标签，对新物种动物的关键点预测进行优化。



2.Weakly- and Semi- supervised cross-domain adaptation(WS-CDA)

2.1. Network Design

（1）输入数据有三个来源，大量标记的人体姿态数据，第二个是少量标记的动物姿态数据，第三个是无标记的动物数据；

（2）WS-CDA包括四个部分：

1）所有数据被输入到一个CNN网络中生成特征图(feature maps)，称为特征提取器(feature extractor)；

2）所有特征图再输入到一个域判别器(domain discriminator)，用于区分输入的特征图来自于哪个域；

3）对于有姿态标签的特征图会被输入到关键点估计器(keypoint estimator)中，进行全监督学习；

4）中间插入一个域自适应网络(domain adaptation network)，为了对齐动物关键点的特征表示；

![WS-CDA](G:\Documents\sunzheng\Learning_SimpleBaseline_and_LightweightBaseling_for_Human_Pose_Estimation\code\WS-CDA.png)

将域判别器和关键点估计器的loss设为对抗性，域判别器用于混淆不同域提取的特征。

domain discrimination loss（DDL）设置如下：
$$
L_{DDL}=-w_1\sum^N_{i=1}(y_ilog(\hat{y}_i)+(1-y_i)log(1-\hat{y}_i))\\
-\sum^N_{i=1}y_i(z_ilog(\hat{z}_i)+(1-z_i)log(1-\hat{z}_i))
$$
其中，$y_i$表示$x_i$是否是一个人($y_i=0$)或者动物($y_i=1$)；$z_i$表示$x_i$是否来自于target domain($z_i=1$)；$\hat{y}_i$和$\hat{z}_i$是判别器预测值；$w_1$是一个权重。



姿态估计损失（包括动物全监督损失(APEL，记作$L_A$)和人全监督损失(HPEL，记作$L_H$)）设计如下：
$$
L_{pose}=\sum^N_{i=1}(w_2y_iL_A(I_i)+(1-y_i)L_H(I_i))
$$
$L_A,L_H$都是均方损失，$w_2$是权重。



网络的最终loss设计如下：
$$
L_{WS-CDA}=\alpha L_{DDL}+\beta L_{pose}
$$
其中，$\alpha\beta<0$，对域判别器和关键点估计器进行了对抗优化，既鼓励了域混淆，又提高了姿态估计性能。实验中取值：$\alpha=-1,\beta=500$。





3.Progressive Pseudo-label-based Optimization(PPLO)  



##### 二、论文2

论文：Learning from Synthetic Animals      

repo：https://github.com/JitengMu/Learning-from-Synthetic-Animals

**思路：**

（1）提出consistency-constrained semi-supervised learning method(CC-SSL，一致性约束半监督学习方法)来解决真实图像和合成图像之间的差距；

（2）我们的方法利用了空间和时间的一致性，用未标记的真实图像引导在合成数据上训练的弱模型；

（3）**在不使用任何真实图像的标签情况下，模型的表现和真实数据上训练的模型接近，在使用少部分真实图像的标签情况下，模型比真实数据上训练的模型表现好**；

（4）合成的数据集包括10多种动物，以及多种姿势和丰富的标签，使得我们可以使用多任务学习策略来进一步提高模型性能；

**总体来说，unsupervised domain adaptation的框架中有两个数据集，一个是合成数据称为源域$(X_s,Y_s)$，还有目标域数据集$X_t$，任务是学习一个函数$f$可以为目标域数据$X_t$预测标签。首先以全监督的方式使用源域数据对$(X_s,Y_s)$来学习源模型$f_s$，再使用目标域数据集以及一致性约束半监督学习方法对源模型$f_s$进行提升。**



**合成数据生成：**

合成数据集包括10+种动物，每只动物都来自一些动画序列。



**基本流程：**

1.在低维流形假设基础上建立统一的图像生成过程

使用一个生成器$G$来将姿势，形状，视角，纹理等转换为一副图像，$X=G(\alpha,\beta)$。其中$\alpha$表示与关键点检测任务相关的因子，如姿势，形状；$\beta$表示与任务无关的因子，如纹理，光照和背景。



2.定义三个一致性准则并考虑在伪标签生成过程如何利用

由于生成目标域数据集的伪标签是有噪声的，所以需要一些准则来判断标签的正确性。定义张量算子$T:\R^{H*W}\rightarrow\R^{H*W}$：
$$
T(X)=G(\tau_\alpha(\alpha),\tau_\beta(\beta))
$$
其中$\tau_\alpha$，$\tau_\beta$是影响$\alpha$和$\beta$的算子（个人理解 $T$ 就是图像的一个变换）。假设$f:\R^{H*W}\rightarrow\R^{H*W}$是一个完美的2D姿态估计模型，则应当满足下面三个准则：

（1）不变一致性（invariance consistency）

如果 $T$ 不改变与任务相关的因子$\alpha$（只改变$\beta$，比如在图像中加一些噪声或者干扰的颜色），则模型 $f$ 的预测结果应该不变，即：
$$
f(T_\beta(X))=f(X)
$$
这个一致性可以用来判断预测是否正确。



（2）等方差一致性（equivariance consistency）

如果 $T$ 改变与任务相关的因子$\alpha$（比如在图像的几何变换），则模型 $f$ 的预测结果应当有以下准则：
$$
f(T_\alpha(X))=T_\alpha(f(X))
$$


（3）时间一致性（temporal consistency）

视频中的相邻帧中的关键点检测应当遵循时间一致性：
$$
f(T_{\Delta}(X))=f(X)+\Delta
$$



3.提出伪标签生成算法，并使用一致性准则检查

![Pseudo_Label](G:\Documents\sunzheng\Learning_SimpleBaseline_and_LightweightBaseling_for_Human_Pose_Estimation\code\Pseudo_Label.png)

在第$n$次迭代中，使用了第$n-1$次迭代所获得的模型$f_{n-1}$，然后对Target dataset $X_t$中的图像$X^i_t$遍历：

（1）首先使用$f_{n-1}$，$T_\alpha$，$T_\beta$，来生成$X^i_t$的伪标签$\hat{Y}_t^{(n),i}$和伪标签置信度$C_t^{(n),i}$；

（2）使用$T_\Delta$来对第一步生成的$\hat{Y}_t^{(n),i}$和$C_t^{(n),i}$进行更新；（这一步可以选择是否进行更新，依据target dataset是否是视频的连续帧）；

Target dataset $X_t$遍历结束之后，会得到伪标签集合以及对应的置信度集合，然后执行：

（3）对置信度集合进行遍历，选择一个$C_{thresh}$，置信度大于这个阈值的$X_t^i$和对应的伪标签$\hat{Y}_i^{(n),i}$才会用来训练。

注意：随着模型训练epoch的增加，会使用越来越多的伪标签来训练，一开始是top-20%的伪标签，一直到80%，CCSSL/CCSSL.py中有这样一段代码：

```python
p = (1.0-0.02*(epoch+10)) # p的取值区间是[0.2,0.8]
if p>0.2:
    ccl_thresh = sorted_confidence[int( p * sorted_confidence.shape[0])]
else:
    p = 0.2
    ccl_thresh = sorted_confidence[int( p * sorted_confidence.shape[0])]
```

p的取值区间是[0.2,0.8]，当epoch=0时，p=0.8；当epoch=30时，p=0.2，此后随着epoch增大，p依旧为0.2，所以生成的伪标签伪标签最多可以用80%。



4.提出一致性约束半监督学习算法来迭代训练

损失函数设计如下：
$$
L^{(n)}=\sum_{i}L_{MSE}(f^{(n)}(X^i_s),Y_s^i)+\lambda \sum_{j}L_{MSE}(f^{(n)}(X_t^j),\hat{Y}_t^{(n-1),j})
$$




总体流程如下：

![CC-SSL](G:\Documents\sunzheng\Learning_SimpleBaseline_and_LightweightBaseling_for_Human_Pose_Estimation\code\CC-SSL.png)

训练流程如下：从初始模型$f^{(0)}$开始（仅使用合成数据训练），在第$n$次的迭代中，使用伪标签生成算法（涉及到$f^{(n-1)},X_t,T$）来生成伪标签$\hat{Y}_t^{(n)}$，随后使用$(X_s,Y_s)$和$(X_t,\hat{Y}_t^{(n)})$计算损失函数来训练模型得到$f^{(n)}$。



**实验部分：**

1.使用Stack Hourglass作为backbone，结构设计不是主要的，所以参数设置和原文章一样；

2.虚拟的相机像素为$640*480$，视角为$90^。$。对于每个合成的动物，生成10000张图像，其中5000张图像的纹理和背景来自于coco图像，另外5000张图像的纹理来自于CAD模型本来的纹理（背景应该还是coco的，从开源的数据大概可以看出来）；

3.数据集

（1）合成数据8000张作为训练集，2000张作为训练集；

（2）真实数据集TigDog：马有79个视频，8380帧作为训练，1772帧作为测试；老虎有96个视频，6523帧作为训练，1765帧作为测试。

4.实验结果：

以$PCK@0.05$对应的各个关键点的准确率作为评价指标：

（1）不使用真实数据的标签时：提出的CC-SSL方法的性能超过其他DA方法；

（2）使用真实数据的标签时：提出的CC-SSL-R方法的性能超过单独在真实数据上训练的结果。

（注：（2）中的CC-SSL-R是用（1）训好的CC-SSL模型在真实数据上finetune的结果）





**本地实验**

1.真实数据集TigDog上的训练和测试

训练指令：

```
python train/train.py --dataset real_animal -a hg --stacks 4 --blocks 1 --image-path ./animal_data/ --checkpoint ./checkpoint/real_animal/horse/horse_hourglass/ --animal horse
```

（可以在最后选择是否添加-d参数进行可视化）

测试指令：

```
python train/train.py --dataset real_animal -a hg --stacks 4 --blocks 1 --image-path ./animal_data/ --checkpoint ./checkpoint/real_animal/horse/horse_hourglass/ --animal horse --resume checkpoint/real_animal/horse/horse_hourglass/model_best.pth.tar --evaluate
```

测试结果：

```python
==> creating model 'hg', stacks=4, blocks=1
=> loading checkpoint 'checkpoint/real_animal/horse/horse_hourglass/model_best.pth.tar'
=> loaded checkpoint 'checkpoint/real_animal/horse/horse_hourglass/model_best.pth.tar' (epoch 60)
    Total params: 13.02M
init real animal stacked hourglass augmentation
split-by-video number of training images:  8380
split-by-video number of testing images:  1772
load mean file
    Mean: 0.5319, 0.5107, 0.4206
    Std:  0.1910, 0.1921, 0.1951
init real animal stacked hourglass augmentation
split-by-video number of training images:  8380
split-by-video number of testing images:  1772
load mean file
    Mean: 0.5319, 0.5107, 0.4206
    Std:  0.1910, 0.1921, 0.1951

Evaluation only
Eval  |################################| (296/296) Data: 0.000s | Batch: 0.161s | Total: 0:00:47 | ETA: 0:00:01 | Loss: 0.00070713 | Acc:  0.78983414
```



第1部分代码解读：

1./dataset/real_animal.py/load_animal函数

该函数中加载真实场景的数据，需要注意的是：在文件夹/animal_data.behaviorDiscovery2.0/ranges中记录的是shot_id（实际上就是video的编号）及对应帧范围，/animal_data.behaviorDiscovery2.0/landmarks中记录的是对应帧的标注信息。实际上标注信息只是标注了ranges中的一部分图片（以horse为例：99个id只标注了80个，15658帧标注了13545帧），load_animal函数也是加载的标注部分的图片。



2.合成数据上训练，真实数据TigDog上测试

训练指令：

```
python train/train.py --dataset synthetic_animal_sp -a hg --stacks 4 --blocks 1 --image-path ./animal_data/ --checkpoint ./checkpoint/synthetic_animal/horse/horse_spaug --animal horse
```

测试指令：

```
python ./evaluation/test.py --dataset1 synthetic_animal_sp --dataset2 real_animal_sp --arch hg --resume ./checkpoint/synthetic_animal/horse/horse_spaug/model_best.pth.tar --evaluate --animal horse
```

测试结果：

```python
==> creating model 'hg', stacks=4, blocks=1
=> loading checkpoint './checkpoint/synthetic_animal/horse/horse_spaug/model_best.pth.tar'
=> loaded checkpoint './checkpoint/synthetic_animal/horse/horse_spaug/model_best.pth.tar' (epoch 97)
    Total params: 13.02M
init synthetic animal super augmentation
10000
total number of images: 10000
train images: 8000
test images: 2000
load from mean file: ./data/synthetic_animal/horse_combineds5r5_texture/mean.pth.tar
load mean file
    Mean: 0.4038, 0.3956, 0.3925
    Std:  0.2611, 0.2452, 0.2299
/home/sunzheng/anaconda3/envs/pytorch16_cuda102_detectron2/lib/python3.7/site-packages/imgaug/imgaug.py:182: DeprecationWarning: Function `ContrastNormalization()` is deprecated. Use `imgaug.contrast.LinearContrast` instead.
  warn_deprecated(msg, stacklevel=3)
init real animal super augmentation
split-by-video number of training images:  8380
split-by-video number of testing images:  1772
load mean file
    Mean: 0.4038, 0.3956, 0.3925
    Std:  0.2611, 0.2452, 0.2299

Evaluation only
Eval  |################################| (296/296) Data: 0.000s | Batch: 0.147s | Total: 0:00:43 | ETA: 0:00:01 | Acc:  0.60842729

per joint PCK@0.05:
[0.7494758913914362, 0.7258620693765837, 0.8503521130433385, 0.5263721580246845, 0.5079651971658071, 0.5305452303039984, 0.5200870661481992, 0.7036821711664052, 0.5651399513692347, 0.6602564111768783, 0.6006278543133442, 0.5511309534311295, 0.48564814983142746, 0.6958333343888322, 0.5198412704325858, 0.6378070180353366, 0.6644591613123748, 0.5489010992613468]
```





**由于前面两个并未涉及Domain Adaptation，所以并没有进行训练，只进行了测试。**

3.使用CC-SSL进行合成数据上训练和真实数据TigDog上测试

训练指令：

```
python CCSSL/CCSSL.py --num-epochs 60 --checkpoint ./checkpoint/synthetic_animal/horse/horse_ccssl --resume ./checkpoint/synthetic_animal/horse/horse_spaug/model_best.pth.tar --animal horse
```

训练指令中：--checkpoint是训好的模型存储的地址，--resume是预训练的模型，即./checkpoint/synthetic_animal/horse/horse_spaug/model_best.pth.tar是在合成数据上训好模型，作为CC-SSL方法的预训练模型。可以设置该参数决定是否进行合成数据的预训练。



测试指令1：

自己训练的结果，不使用合成数据的模型（./checkpoint/synthetic_animal/horse/horse_spaug/model_best.pth.tar）进行预训练，即resume参数设置为空

```
python ./evaluation/test.py --dataset1 synthetic_animal_sp --dataset2 real_animal_sp --arch hg --resume ./checkpoint/synthetic_animal/horse/horse_ccssl/synthetic_animal_sp.pth.tar --evaluate --animal horse
```

测试指令中：--resume是要测试的模型，即上一步训练指令中用CC-SSL方法训好的模型，并在测试完毕之后，将路径改为：./checkpoint/synthetic_animal/horse/horse_ccssl_try/synthetic_animal_sp.pth.tar

测试结果：

```python
==> creating model 'hg', stacks=4, blocks=1
=> loading checkpoint './checkpoint/synthetic_animal/horse/horse_ccssl/synthetic_animal_sp.pth.tar'
=> loaded checkpoint './checkpoint/synthetic_animal/horse/horse_ccssl/synthetic_animal_sp.pth.tar' (epoch 58)
    Total params: 13.02M
init synthetic animal super augmentation
10000
total number of images: 10000
train images: 8000
test images: 2000
load from mean file: ./data/synthetic_animal/horse_combineds5r5_texture/mean.pth.tar
load mean file
    Mean: 0.4038, 0.3956, 0.3925
    Std:  0.2611, 0.2452, 0.2299
/home/sunzheng/anaconda3/envs/pytorch16_cuda102_detectron2/lib/python3.7/site-packages/imgaug/imgaug.py:182: DeprecationWarning: Function `ContrastNormalization()` is deprecated. Use `imgaug.contrast.LinearContrast` instead.
  warn_deprecated(msg, stacklevel=3)
init real animal super augmentation
split-by-video number of training images:  8380
split-by-video number of testing images:  1772
load mean file
    Mean: 0.4038, 0.3956, 0.3925
    Std:  0.2611, 0.2452, 0.2299

Evaluation only
Eval  |################################| (296/296) Data: 0.000s | Batch: 0.142s | Total: 0:00:42 | ETA: 0:00:01 | Acc:  0.41582018

per joint PCK@0.05:
[0.5406708596449978, 0.49568965568624695, 0.7482394362524362, 0.3702141916057671, 0.35829986871246355, 0.2669144994596566, 0.38700248862602815, 0.6056847553267035, 0.4281170511518726, 0.42150349800403303, 0.39303653120790444, 0.33089285860104223, 0.21296296330789724, 0.2593750008381903, 0.2670634934412582, 0.4085964911078152, 0.5264900660277992, 0.23901099048473023]
```



测试指令2：（官方模型训练结果，使用合成数据进行预训练./checkpoint/synthetic_animal/horse/horse_spaug/model_best.pth.tar）

```shell
python ./evaluation/test.py --dataset1 synthetic_animal_sp --dataset2 real_animal_sp --arch hg --resume ./checkpoint/synthetic_animal/horse/horse_ccssl/synthetic_animal_sp.pth.tar --evaluate --animal horse
```

测试指令中：--resume是要测试的模型，即上一步训练指令中用CC-SSL方法训好的模型。

测试结果：

```python
==> creating model 'hg', stacks=4, blocks=1
=> loading checkpoint './checkpoint/synthetic_animal/horse/horse_ccssl/synthetic_animal_sp.pth.tar'
=> loaded checkpoint './checkpoint/synthetic_animal/horse/horse_ccssl/synthetic_animal_sp.pth.tar' (epoch 48)
    Total params: 13.02M
init synthetic animal super augmentation
10000
total number of images: 10000
train images: 8000
test images: 2000
load from mean file: ./data/synthetic_animal/horse_combineds5r5_texture/mean.pth.tar
load mean file
    Mean: 0.4038, 0.3956, 0.3925
    Std:  0.2611, 0.2452, 0.2299
/home/sunzheng/anaconda3/envs/pytorch16_cuda102_detectron2/lib/python3.7/site-packages/imgaug/imgaug.py:182: DeprecationWarning: Function `ContrastNormalization()` is deprecated. Use `imgaug.contrast.LinearContrast` instead.
  warn_deprecated(msg, stacklevel=3)
init real animal super augmentation
split-by-video number of training images:  8380
split-by-video number of testing images:  1772
load mean file
    Mean: 0.4038, 0.3956, 0.3925
    Std:  0.2611, 0.2452, 0.2299

Evaluation only
Eval  |################################| (296/296) Data: 0.000s | Batch: 0.150s | Total: 0:00:44 | ETA: 0:00:01 | Acc:  0.70774686

per joint PCK@0.05:
[0.9014675047412608, 0.7905172412765438, 0.9025821594495169, 0.6113788509464647, 0.5686077654481413, 0.6128872376503112, 0.6601368172733642, 0.85885012941074, 0.6365139958171444, 0.7329254090473368, 0.6855022834792529, 0.6942261909799916, 0.61666666643901, 0.7770833331160247, 0.6498015875972453, 0.7169298244150062, 0.7400662259550284, 0.6363553123159723]
```

可以看出有合成数据模型预训练的结果和没有预训练的结果相差较多。





第3部分代码解读：

1.模型代码





2.数据加载部分

（1）/CCSSL/scripts/ssl_datasets/ssl_synthetic_animal_sp.py



（2）/CCSSL/scripts/ssl_datasets/ssl_real_animal_crop.py



（3）/CCSSL/scripts/ssl_datasets/ssl_real_animal_sp.py













#### 附录一、MSCOCO数据集介绍

1.简介

MSCOCO数据集适用于目标检测，图像分割，关键点检测以及图像描述等任务。相比于PASCAL VOC数据集，COCO数据集包含了生活中的常见图片，背景比较复杂，目标数量比较多并且尺寸更小，因此COCO数据集上的任务更难。MSCOCO数据集共91（后续测试索引文件只有80）个类别，远远小于ImageNet的1000个类别，但是每个类别的实例数量要比ImageNet多；



2.关于索引文件（以COCO2017为例）

（参考链接 https://zhuanlan.zhihu.com/p/29393415 ）

1.索引种类

COCO数据集有三种标注类型：object instances（目标实例），object keypoint（目标关键点），image captions（图像标注），都是使用json文件存储，每种类型都包含了训练集和验证集，共6个json文件。



2.索引内容

三种类型的索引文件json存储的都是字典，有基本键类型：info、image、license、annotation、categories，前三个是共享的，后两个根据具体任务会不一样。内容格式也不一样，而且image captions任务的索引文件是没有categories这一键的。

**（1）info、image、license键及对应的值**

'info'对应的值是字典：

```json
'info':{
  'description':'COCO 2017 Dataset',
  'url':'http://cocodataset.org',
  'version':'1.0',
  'year':'2017',
  'contributor':'COCO Consortium',
  'data_createed':'2017/09/01'
}
```

'image'对应的值是一个列表，列表中嵌套各个实例的字典：

```json
'image':[
  {'license':3,'filename':'COCO_val2014_000000074478.jpg','height':640,'width':428,'data_captured':'2013-11-25 20:09:45','flickr_url':'http://fram3.staticflickr.com/2753/4318988969_653bb58b14_z.jpg','id':74478},
  {},{},....]
```

'license'对应的值是列表，列表中嵌套各个实例的字典：

```json
'license':[
  {'url':'http://creativecommons.org/licenses/by-nc-sa/2.0/','id':1,'name':'Attribution-NonCommercial-ShareAlike License'},
  {},{},...
]
```



**（2）Object Instance的annotation和categories**

'annotation'对应的值是列表，列表中嵌套各个实例的字典：

```json
'annotation':[
  {'segmentation':[[239.97,260.24,...,271.34]],'area':2765.1486500,'iscrowd':0,'image_id':558840,'bbox':[190.84,200.46,77.71,70.88],'category_id':58,'id':156},
  {},{},...
]
```

（从instances_val2014.json文件中摘出的一个annotation的实例，这里的segmentation是polygon格式）

iscrowd=0，表示只有单个对象，此时segmentation是polygon格式；iscrowd=1，标注一组对象（比如一群人）的segmentation使用的就是RLE格式。

另外，每个对象（不管是iscrowd=0还是iscrowd=1）都会有一个矩形框bbox ，矩形框左上角的坐标和矩形框的长宽会以数组的形式提供，数组第一个元素就是左上角的横坐标值。area是area of encoded masks，是标注区域的面积。如果是矩形框，那就是高乘宽；如果是polygon或者RLE，那就复杂点，第（3）部分有代码测试，计算并显示分割区域的位置。

最后，annotation结构中的categories字段存储的是当前对象所属的category的id，以及所属的supercategory的name。

'categories'对应的值是列表，列表中嵌套各个实例的字典：

```json
'categories':[
  {'supercategory':'person','id':1,'name':'person'},
  {},{},...
]
```



**（3）Object Instance索引文件代码测试**

1）images列表元素数量等于训练集（测试集）的图片数量，annotations列表元素数量等于训练集（测试集）中bounding box的数量；categories列表元素数量为COCO数据集中的目标类别数，COCO2014和COCO2017都为80。

check_annotations.py

```python
#import coco
#import pycocotools.coco as coco
import json


ann_val_file = './annotations/instances_val2014.json'
#coco_val = coco(ann_val_file)

with open(ann_val_file,'r',encoding='utf8')as fp:
    ann_data = json.load(fp)

print(type(ann_data))
print(type(ann_data['images']))
print(len(ann_data['categories']))
print(len(ann_data['images']))
print(len(ann_data['annotations']))

```



2）polygon格式比较简单，这些数按照相邻的顺序两两组成一个点的xy坐标，如果有n个数（必定是偶数），那么就是n/2个点坐标。下面就是一段解析polygon格式的segmentation并且显示多边形的示例代码：

check_polygon.py

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
fig, ax = plt.subplots()
polygons = []
num_sides = 100
gemfield_polygons = [[246.9,404.24,261.04,378.46,269.35,316.1,269.35,297.8,290.98,252.07,299.29,227.12,
                    302.62,173.07,286.81,143.97,285.16,133.16,310.1,124.01,321.74,116.52,335.05,117.35,
                    351.68,119.02,359.99,128.16,348.35,137.31,341.7,152.28,361.66,163.93,384.94,190.53,
                    401.57,208.82,394.09,219.64,386.6,221.3,375.79,214.65,370.8,201.34,366.64,256.22,
                    360.83,270.36,369.14,284.5,367.47,320.25,383.28,350.19,401.57,386.78,409.05,408.4,
                    408.22,420.87,400.74,431.68,389.1,425.03,381.61,395.93,370.8,366.82,343.37,321.92,
                    329.22,288.65,310.1,313.6,299.29,312.77,285.98,332.73,283.49,362.66,276.0,392.6,
                    281.83,399.25,259.37,416.71,236.92,416.71,236.92,407.56]]
gemfield_polygon = gemfield_polygons[0]
max_value = max(gemfield_polygon) * 1.3
gemfield_polygon = [i * 1.0/max_value for i in gemfield_polygon]
poly = np.array(gemfield_polygon).reshape((int(len(gemfield_polygon)/2), 2))
polygons.append(Polygon(poly,True))
p = PatchCollection(polygons, cmap=matplotlib.cm.jet, alpha=0.4)
colors = 100*np.random.rand(1)
p.set_array(np.array(colors))

ax.add_collection(p)

plt.savefig('./polygon.png')
plt.show()
```

结果存为./result_image/polygon.png.

3）如果iscrowd=1，那么segmentation就是RLE格式(segmentation字段会含有counts和size数组)

COCO数据集的RLE都是uncompressed RLE格式（与之相对的是compact RLE）。RLE所占字节的大小和边界上的像素数量是正相关的。RLE格式带来的好处就是当基于RLE去计算目标区域的面积以及两个目标之间的unoin和intersection时会非常有效率。 上面的segmentation中的counts数组和size数组共同组成了这幅图片中的分割mask。其中size是这幅图片的宽高，然后在这幅图像中，每一个像素点要么在被分割（标注）的目标区域中，要么在背景中。很明显这是一个bool量：如果该像素在目标区域中为true那么在背景中就是False；如果该像素在目标区域中为1那么在背景中就是0。对于一个375x500的图片来说，一共有187500个像素点，根据每一个像素点在不在目标区域中，我们就有了187500个bit，比如像这样（随便写的例子）：00000111100111110...；但是这样写很明显浪费空间，可以写连续0,1的个数（Run-length encoding)，于是就成了54251...，这就是多个对象的分割结果标注方式。下面这个python代码片段直观的显示了这些bit：

check_rle.py

```python
import numpy as np
import matplotlib.pyplot as plt
rle = [139,3,20,2,20,11,313,15,1,17,1,13,3,14,310,48,2,16,308,69,304,72,302,74,300,76,298,78,
    296,79,295,80,295,80,295,80,295,80,295,80,292,83,290,85,290,85,289,73,1,12,289,72,2,12,289,
    71,3,12,288,87,287,88,286,89,286,89,286,89,286,44,1,44,286,43,3,43,286,42,5,42,286,42,6,41,
    286,42,7,39,288,41,7,39,289,21,1,19,9,3,2,31,291,3,2,34,14,31,296,34,14,31,296,35,13,31,296,
    35,13,31,297,34,13,30,298,35,12,17,1,12,298,35,12,16,2,11,299,35,12,16,3,3,306,35,12,15,314,
    33,14,14,315,32,15,13,314,34,15,10,315,35,15,9,316,35,15,8,316,36,16,6,316,37,16,5,316,38,17,
    2,318,38,336,39,335,40,334,27,1,13,334,18,10,13,334,17,11,13,333,18,12,12,332,19,12,12,331,20,
    12,12,331,21,11,12,330,22,12,10,331,23,12,8,332,23,14,3,334,24,350,26,349,26,348,28,347,28,345,
    30,344,32,343,32,342,33,341,34,340,35,339,36,338,37,338,37,337,38,337,38,337,38,336,39,335,40,
    335,40,335,40,334,41,334,17,1,23,333,16,2,24,332,17,2,24,332,16,3,24,332,44,331,45,330,46,329,
    46,329,47,328,47,329,46,328,47,327,48,326,49,326,49,326,49,326,49,326,48,327,47,328,46,329,45,
    330,45,330,46,328,35,3,9,328,24,1,8,6,6,329,24,2,7,8,2,331,24,2,8,9,1,330,24,2,9,340,23,2,10,
    340,22,2,11,340,21,3,10,341,20,3,11,341,20,3,13,339,20,2,18,336,19,1,23,333,18,1,24,334,16,1,
    23,339,36,339,35,340,35,340,34,341,33,342,40,335,41,334,41,334,41,334,41,334,42,334,41,334,41,
    334,41,334,42,334,41,334,41,334,42,333,42,332,43,332,43,331,45,330,46,329,47,328,48,325,50,325,
    50,324,51,324,36,2,14,323,35,3,13,324,35,4,11,324,37,3,10,325,37,3,10,324,38,4,8,324,40,5,5,325,
    40,6,4,325,40,7,3,325,40,8,2,1,4,320,40,9,5,321,40,10,1,324,40,335,40,335,42,333,43,332,47,328,
    48,327,50,326,50,325,51,324,52,323,52,323,52,322,53,322,53,322,53,322,53,322,53,322,53,322,53,
    322,52,323,51,324,44,1,3,327,44,331,44,331,43,332,42,334,38,338,35,342,33,344,31,343,32,342,33,
    342,33,342,33,341,34,341,34,341,34,341,20,1,12,342,17,7,8,342,18,10,2,345,17,357,18,357,19,356,
    19,356,21,354,25,350,26,349,27,348,28,347,29,346,30,345,31,344,30,345,30,345,30,346,29,346,28,
    347,28,348,27,348,28,347,32,344,33,343,37,339,37,339,38,339,39,338,38,337,38,337,39,336,40,335,
    41,334,42,333,42,332,43,332,43,332,43,332,43,331,44,331,43,332,42,333,41,334,40,335,39,336,38,
    336,38,337,38,337,37,338,37,337,37,338,36,339,36,339,35,340,34,341,34,341,33,342,32,344,31,345,
    29,348,27,349,25,350,25,350,25,350,24,351,24,351,23,353,21,355,19,358,3,3,8,20621,8,362,14,359,
    17,355,21,353,22,352,23,351,24,351,24,351,24,351,24,351,23,352,22,353,20,353,19,355,16,354,20,
    354,21,353,22,352,23,352,23,352,23,352,23,352,23,352,22,353,21,355,17,359,11,366,6,17629,8,366,
    8,366,10,364,11,364,11,364,11,364,11,3,9,352,11,1,14,349,27,348,27,349,26,350,25,352,4,2,17,360,
    15,363,12,364,10,366,8,369,3,7144,8,366,10,364,12,362,14,361,14,360,15,348,27,347,29,345,30,344,
    31,344,32,343,32,343,32,338,37,337,38,336,39,335,40,335,40,335,40,335,40,335,40,335,40,335,44,331,
    45,331,15,2,28,331,12,6,27,332,3,12,28,346,29,346,29,346,29,345,30,344,1,1,29,343,1,2,21,351,1,2,
    13,362,7,368,5,370,5,371,3,372,3,373,1,10358]
assert sum(rle) == 375*500
print(len(rle))
M = np.zeros(375*500)
N = len(rle)
n = 0
val = 1
for pos in range(N):
    #print(pos)
    val = not val
    for c in range(rle[pos]):
        M[n] = val
        #print(n)
        n += 1

GEMFIELD = M.reshape(([375, 500]), order='F')
plt.imshow(GEMFIELD)
plt.savefig('./rle.png')
plt.show()

```

结果存为./result_image/rle.png.



**（4）Object Keypoints的annotation和categories**

这个类型中的annotation结构体包含了Object Instance中annotation结构体的所有字段，再加上2个额外的字段。

新增的keypoints是一个长度为3*k的数组，其中k是该样本中keypoints的数量。每一个keypoint是一个长度为3的数组，第一和第二个元素分别是x和y坐标值，第三个元素是个标志位v，v为0时表示这个关键点没有标注（这种情况下x=y=v=0），v为1时表示这个关键点标注了但是不可见（被遮挡了），v为2时表示这个关键点标注了同时也可见。

```json
annotation{
    "keypoints": [x1,y1,v1,...],
    "num_keypoints": int,
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,}
```

例如：

```json
{
	"segmentation": [[125.12,539.69,140.94,522.43...]],
	"num_keypoints": 10,
	"area": 47803.27955,
	"iscrowd": 0,
	"keypoints": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,309,1,177,320,2,191,398...],
	"image_id": 425226,"bbox": [73.35,206.02,300.58,372.5],"category_id": 1,
	"id": 183126
}
```



对于每一个category结构体，相比Object Instance中的category新增了2个额外的字段，keypoints是一个长度为k的数组，包含了每个关键点的名字；skeleton定义了各个关键点之间的连接性（比如人的左手腕和左肘就是连接的，但是左手腕和右手腕就不是）。目前，COCO的keypoints只标注了person category （分类为人）。

```json
{
    "id": int,
    "name": str,
    "supercategory": str,
    "keypoints": [str],
    "skeleton": [edge]
}
```

例如：

```json
{
	"supercategory": "person",
	"id": 1,
	"name": "person",
	"keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
	"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
}
```





**（5）Image Caption的annotation和categories**

注：由于个人不做这个方向，所以可以自行参考链接。





#### 附录二、COCO数据集各任务评价指标

一、分类

参考链接 https://zhuanlan.zhihu.com/p/43068926 

混淆矩阵：

| （竖为预测，横为实际） | 1                  | 0                  |
| ---------------------- | ------------------ | ------------------ |
| 1                      | TP(True Positive)  | FP(False Positive) |
| 0                      | FN(False Negative) | TN(True Negative)  |

精度：在预测为正样本中实际为正样本的概率，P=TP/(TP+FP)

召回率：在实际为正样本中预测为正样本的概率，R=TP/(TP+FN)

当置信度阈值取不同的值时（这里的阈值是为了得到PR对，和之后计算AP的阈值区间没关系），会得到（P,R）对，用曲线拟合这些点就得到P-R曲线。一般认为召回率-精度曲线（P-R曲线）与坐标轴围成的面积是AP，然后各个类别的AP取平均值就是当前模型的mAP，但是实际计算会有一点差别。



二、VOC2010前后的mAP计算方法

mAP的计算方法在VOC2010前后是不一样的，VOC2010之前的方法实际上就是对于给定召回率区间，如Recall（召回率）$>0,0.1,0.2,...,0.9,1$（共11个区间）。计算当前类别在各个区间中最大的Pre（精度），将这些最大精度相加求平均，就得到当前类别的AP，最后求各个类别的AP，再求平均得到当前模型的mAP，公式如下：
$$
AP=\frac{1}{11}\sum_{r\in [0,0.1,...,1]}P(r),其中P(r)=max_{r'\geq r}p
$$
VOC2010以后的方法和求P-R曲线的面积很类似，根据真实的召回率来划分区间，然后算当前类别在当前区间中的最大精度和区间宽度差的乘积（区间宽度是相邻两个召回率的差值），所以实际上是在近似计算当前类别P-R曲线围成的面积。



三、目标检测

检测任务中AP默认就是mAP，假设一张图片中有m个box（标签），模型预测得到n个box+score。首先给定一些score阈值（类似分类任务中的概率置信度，用来生成PR对，score大于该阈值的box才会进入IoU的计算环节）。

检测任务多了一个IoU指标，使得检测任务需要在不同的IoU阈值下来计算mAP，比如数据集PASCAL VOC中设置阈值为0.5，当确定好$IoU$阈值之后，即可进行mAP的计算。此外，在目标检测的mAP计算中，没有TN，只有TP，FP，FN。当某两个框（一个标签框，一个预测框）之间的IoU大于给定的阈值0.5时，这两个框视为TP，如果这两个框之间的IoU小于阈值0.5，则将标签框视为FN（正确的没有识别出来），预测框视为FP(错误的识别为正确的)，即可计算精度和召回。后续和分类任务一样的计算方法。

举例说明，对于某张图片：

标签有2个box：

| id   | box               |
| ---- | ----------------- |
| 0    | [480,457,515,529] |
| 1    | [637,435,676,536] |

预测有6个box和相应的score（已经按照score进行排序）：

| id   | score  | box                  |
| ---- | ------ | -------------------- |
| 0    | 0.972  | [641, 439, 670, 532] |
| 1    | 0.9113 | [484, 455, 514, 519] |
| 2    | 0.8505 | [649, 479, 670, 531] |
| 3    | 0.2452 | [357, 458, 382, 485] |
| 4    | 0.1618 | [468, 435, 520, 521] |
| 5    | 0.1342 | [336, 463, 362, 496] |

首先规定一个IoU的阈值，比如阈值为0.7，则该情况下计算得到的AP记作$AP_{0.7}$。(在这个例子中只有标签id=0和预测id=1的框之间IoU大于这个阈值）

按照mAP的计算流程，首先要设定score阈值来产生（P,R）对，计算不同score阈值下的精度和召回率：

（1）score阈值为0.972，则：

| TP   | FP   | FN   | precision | recall |
| ---- | ---- | ---- | --------- | ------ |
| 0    | 1    | 2    | 0         | 0      |

（2）score阈值为0.9113，则：

| TP   | FP   | FN   | precision | recall |
| ---- | ---- | ---- | --------- | ------ |
| 1    | 1    | 1    | 0.5       | 0.5    |

（3）score阈值为0.8505，则：

| TP   | FP   | FN   | precision | recall |
| ---- | ---- | ---- | --------- | ------ |
| 1    | 2    | 1    | 0.33      | 0.5    |

（4）score阈值为0.2452，则：

| TP   | FP   | FN   | precision | recall |
| ---- | ---- | ---- | --------- | ------ |
| 1    | 3    | 1    | 0.25      | 0.5    |

（5）score阈值为0.1618，则：

| TP   | FP   | FN   | precision | recall |
| ---- | ---- | ---- | --------- | ------ |
| 1    | 4    | 1    | 0.2       | 0.5    |

（6）score阈值为0.1342，则：

| TP   | FP   | FN   | precision | recall |
| ---- | ---- | ---- | --------- | ------ |
| 1    | 5    | 1    | 0.16      | 0.5    |

产生（P,R）对之后，可以使用11点法或者VOC2010之后的方法来计算$AP_{0.7}$。COCO中$AP_{0.5:0.05:0.95}$，就是依次计算IoU为0.5,0.55,0.6,...,0.95时，对应的AP，再求均值得到$AP_{0.5:0.05:0.95}$。因为COCO还是多类别，所以再对类别求平均就是mAP，但是COCO中统一为了AP。





四、关键点检测

关键点检测性能度量指标(OKS,Object Keypoint Similarity)

Keypoint detection度量方法的核心思想就是模仿Object detection的度量方法：average precision(AP),average recall(AR),F1-score等，这些度量方法的核心都是去检测真实目标和预测目标之间的相似度。在Object detection中，IoU是作为一种相似度度量，它定义了在真实目标和预测值目标之间的匹配程度并允许计算precision-recall曲线。为了将AP/AR应用到关节点检测，定义了一个类似IoU的OKS(object keypoint similarity)方法。(目标检测的指标前面有简单叙述，这里略去)

Object Keypoint Similarity(OKS)

真实关键点的格式：$[x_1,y_1,v_1,x_2,y_2,v_2,...,x_k,y_k,v_k]$

其中$x,y$为Keypoint的坐标，$v$为可见标志：$v=0$，未标注点；$v=1$，标注了但是图像不可见（例如遮挡）；$v=2$，标注了并图像可见。
$$
OKS=\frac{\sum_i[exp(\frac{-d_i^2}{2s^2k_i^2})\delta(v_i>0)]}{\sum_i\delta(v_i>0)}
$$
对某个人来计算OKS：

$i$表示Keypoint的id；

$d_i$是标注和预测关节点之间的欧式距离；

$s$表示当前人的尺度因子，这个值等于此人在groundtruth中所占的面积的平方根，即$\sqrt{(x_2-x_1)(y_2-y_1)}$；

$k_i$表示第$i$个关键点的归一化因子(通过对数据集中的所有groundtruth计算的标准差而得到的，反映当前骨骼点标注时候的标准差，$k_i$越大表示这个点越难标注)；

$v_i$表示第$i$个关键点是否可见；

$\delta$用于将可见点选出来进行计算的函数，即脉冲$\delta$函数；

每个关键点相似度都在$[0,1]$之间，完美的预测将会得到$OKS=1$，预测值与真实值差距太大将会有$OKS->0$。



OKS矩阵：

当一个图像中有多个人的时候，需要构造一个OKS矩阵。假设一张图中，一共有$M$个人(Groudtruth)，现在算法预测出了$N$个人，可以构造一个$M*N$的矩阵，矩阵中的位置$(i,j)$代表groundtruth中的第$i$个人和算法预测出的第$j$个人的OKS相似度，找到矩阵中每一行的最大值，作为对应于第$i$个人的OKS相似度。



$PCK$指标：

$PCK$指标是在$OKS$指标出现之前广泛使用的关键点检测评价指标，是以关键点为单位计算的指标，可以输出每个关键点对应的准确率。对于一个检测器$d_0$来说，其$PCK$为：
$$
PCK^p_\sigma(d_0)=\frac{1}{T}\sum_{t\in A}\delta(||x_p^f-y_p^f||<\sigma)
$$
$T$表示测试集合中样本的个数，$\sigma$表示欧式距离的阈值，$A$表示测试集合，$x_p^f$表示检测器的预测位置，$y_p^f$表示真实位置。可以通过卡不同的阈值来计算AP。

对于$OKS$和$PCK$指标，区别实际上很明显，一个是以人为单位，计算每个人的$OKS$（每个人的$OKS$又和这个人的所有关键点有关），再对$OKS$设置阈值，来计算当前batch_size所有人中大于给定阈值的比例；一个以关键点为单位，计算每个关键点归一化之后的预测坐标和真实坐标之间的欧式距离，再对距离设置阈值，计算当前关键点在batch_size中小于给定阈值的比例，再求该阈值下所有关键点（比如17个）的平均值比例。





#### 附录三、框架代码解读

由于上述几篇论文都是使用同一套论文框架，所以有必要对框架代码进行学习，这里记录SimpleBaseline和HRNet+UDP的代码。

##### 1.Simple Baseline框架代码

（1）parser.parse_known_args()函数

```python
args, rest = parser.parse_known_args()
```

parse_known_args()比parse_args()功能强大，它在接受到多余的命令行参数时不报错。相反的，返回一个tuple类型的命名空间和一个保存着余下的命令行字符的list，如：

```python
import argparse 
parser = argparse.ArgumentParser() 
parser.add_argument( 
    '--flag_int', 
    type=float, 
    default=0.01) 
args, unparsed = parser.parse_known_args() 
print(args) 
print(unparsed)

$ python test.py --flag_int 0.02 --double 0.03 --a 1
Namespace(flag_int=0.02)
['--double', '0.03', '--a', '1']
```

（2）config和yaml文件读取方式

train.py中的config在环境编译的时候已经在/lib/core/config.py中定义好了，是一个EasyDict。作用是可以使得以属性的方式去访问字典的值：

```python
from easydict import EasyDict as edict
d = edict({'foo':3, 'bar':{'x':1, 'y':2}})
print(d.foo) # 3
print(d.bar.x) # 1
```

```python
import yaml
from easydict import EasyDict as edict

config_file = './256x192_d256x3_adam_lr1e-3_lpn.yaml'
with open(config_file) as f:
    exp_config = edict(yaml.load(f))
print(exp_config) # 是一个EasyDict
'''
{'GPUS': '0,1,2,3', 'DATA_DIR': '', 'OUTPUT_DIR': 'output', 'LOG_DIR': 'log', 'WORKERS': 4, 'PRINT_FREQ': 10, 'DATASET': {'DATASET': 'monkey', 'ROOT': 'data/monkey_demo/', 'TEST_SET': 'val', 'TRAIN_SET': 'train', 'FLIP': True, 'ROT_FACTOR': 40, 'SCALE_FACTOR': 0.3}, 'MODEL': {'NAME': 'lpn', 'PRETRAINED': 'output/coco/lpn_50/256x192_d256x3_adam_lr1e-3_lpn/model_best.pth.tar', 'IMAGE_SIZE': [192, 256], 'NUM_JOINTS': 17, 'EXTRA': {'ATTENTION': 'GC', 'TARGET_TYPE': 'gaussian', 'HEATMAP_SIZE': [48, 64], 'SIGMA': 2, 'FINAL_CONV_KERNEL': 1, 'DECONV_WITH_BIAS': False, 'NUM_DECONV_LAYERS': 2, 'NUM_DECONV_FILTERS': [256, 256], 'NUM_DECONV_KERNELS': [4, 4], 'NUM_LAYERS': 50}}, 'LOSS': {'USE_TARGET_WEIGHT': True}, 'TRAIN': {'BATCH_SIZE': 16, 'SHUFFLE': True, 'BEGIN_EPOCH': 0, 'END_EPOCH': 150, 'RESUME': False, 'OPTIMIZER': 'adam', 'LR': 0.001, 'LR_FACTOR': 0.1, 'LR_STEP': [90, 120], 'WD': 0.0001, 'GAMMA1': 0.99, 'GAMMA2': 0.0, 'MOMENTUM': 0.9, 'NESTEROV': False}, 'TEST': {'BATCH_SIZE': 16, 'COCO_BBOX_FILE': 'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json', 'BBOX_THRE': 1.0, 'FLIP_TEST': False, 'IMAGE_THRE': 0.0, 'IN_VIS_THRE': 0.2, 'MODEL_FILE': '', 'NMS_THRE': 1.0, 'OKS_THRE': 0.9, 'USE_GT_BBOX': True}, 'DEBUG': {'DEBUG': True, 'SAVE_BATCH_IMAGES_GT': True, 'SAVE_BATCH_IMAGES_PRED': True, 'SAVE_HEATMAPS_GT': True, 'SAVE_HEATMAPS_PRED': True, 'TEST': ''}}
'''
```

（3）eval函数

eval函数的作用使用字符串代替执行字符串里的内容，这个作用在读取yaml参数的时候非常管用，如：

```python
x = 7
eval( '3 * x' ) # 21
eval('pow(2,2)') # 4
eval('2 + 2') # 4
model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(config)
```

（4）python类中的\__len\__以及\__getitem\__函数

定义了\__len\__函数可以直接len(s)来求类实例s的长度，定义了\__getitem\__函数可以直接s[index]，通过索引来取类实例s中的元素，如：

```python
class Student(object):
    def __init__(self, *args):
        self.name = args
    def __len__(self):
        return len(self.name)
    def __getitem__(self, idx):
        return self.name[idx]

s = Student('Bob', 'Alice', 'JR_Smith')
print(s.name) # ('Bob', 'Alice', 'JR_Smith')
print(len(s)) # 3
print(s[1]) # Alice
```



（5）损失函数的计算

损失函数有三个参数，output(当前样本的模型输出)，target(当前样本的标签)，target_weight(当前样本的标签权重)；通过print中间变量的信息，模型输出output和标签target都是规模为$batch\_size*17*64*48$的tensor，target_weight是$batch\_size*17*1$，给17个通道分别赋权值0，1，权值0表示该关键点没有进行标注或者该关键点经过坐标映射，$256*192$的原图关键点坐标映射到$64*48$的target关键点坐标之后，该关键点对应的高斯半径脱离了heatmap的范围（实际上就是该关键点经过变换脱离了heatmap的范围，并且该点对应的高斯圆也脱离了heatmap的范围），1表示关键点仍然存在（数据加载部分生成target_weight）。

损失函数loss.py：

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True) # 计算结果取平均
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        #print('loss calculated:')
        batch_size = output.size(0)
        num_joints = output.size(1)
        #print(batch_size)
        #print(num_joints)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        #print(len(heatmaps_pred)) # 17
        #print(len(heatmaps_gt)) # 17
        #print(heatmaps_pred[0].size()) # torch.size([size,1,3072])
        #print(heatmaps_gt[0].size()) # torch.size([size,1,3072])
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            # print(heatmap_pred.size()) # torch.size([size,3072])
            # print(target_weight[:,idx])
            # print(target_weight[:,idx].size()) # torch.size([size,1])
            # print(heatmap_pred.mul(target_weight[:, idx]))
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            
            #if idx==0:
                #break

        return loss / num_joints
```

从代码中可以看出，在JointsMSELoss中实际上还是nn.MSELoss(size_average=True)来计算标签值和预测值之间的差距，测试该函数：

```python
import torch

loss_fn1 = torch.nn.MSELoss(size_average=False) # 计算欧式距离之后相加，不计算平均值
loss_fn2 = torch.nn.MSELoss(size_average=True) # 计算欧式距离之后相加，计算平均值

a = torch.Tensor([[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10]])
b = torch.Tensor([[3, 3, 4, 5], [4, 5, 6, 7], [8, 9, 10, 11]])

loss1 = loss_fn1(a.float(), b.float())
loss2 = loss_fn2(a.float(), b.float())

print(loss1) # 15
print(loss2) # 1.25=15/12，12是4*3的结果，即batch_size为3，每个样本是4*1的tensor
```

将$batch\_size*17*64*48$维度的output和target变为$17*batch\_size*3072$的tensor，target_weight是$batch\_size*17*1$的tensor，然后对17个关键点遍历，idx从1到17，每次计算乘以权重（$batch\_size*1$）之后的output（$batch\_size*3072$）和target的MSE，即：

```python
loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx]))
```

最后将17个关键点的loss相加求均值得到每个batch的loss结果。



（5）加载数据集类（代码中最重要的一部分）

coco.py

```python
class COCODataset(JointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform) # 初始化父类JointsDataset
        .......
```

在加载数据进行训练时，会用到数据类中的\__getitem\__函数：

```python
def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        #print(db_rec)
        #print('here')
        #print(idx)
        image_file = db_rec['image'] # 获取图片路径
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        
        # print(self.data_format) # 'jpg',/lib/core/config.py里面指定该参数
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # print(type(data_numpy)) # <class 'numpy.ndarray'>
        #print(data_numpy.shape) # monkey:960*1280*3
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            #print('s:',s) # scale因子
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
            #print('r:',r) # 旋转度数
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :] # ::-1表示水平翻转
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
            #print('c:',c)

        # 使用cv2.getAffineTransform获取仿射变换矩阵trans
        trans = get_affine_transform(c, s, r, self.image_size)
        # print(type(trans)) # <class 'numpy.ndarray'>
        # print(trans.shape) # (2,3)
        
        # 利用放射变换矩阵trans以及cv2.warpAffine函数来变换原图像
        # 根据放射变换矩阵，原图的box区域会变换为256*192的目标图像
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR) # 
        # print(type(input)) # <class 'numpy.ndarray'>
        # print(input.shape) # torch.Size([3,256,192])
        if self.transform:
            input = self.transform(input)
        # print(type(input)) # <class 'torch.Tensor'>
        # print(input.size()) # torch.Size([3, 256, 192])
        for i in range(self.num_joints):
            # 如果关键点可见(v = 1 or 2)，则将关键点坐标也进行放射变换
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)
        
        # 生成的numpy格式的target和target_weight转为tensor
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta
```

target使用标准差为2的高斯函数来计算高斯核，具体计算过程见lib/dataset/JointDataset.py（coco.py和mpii.py都继承了JointDataset.py）中的generate_heatmap函数，在加载数据集的时候计算的标签target。

```python
def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32) # 初始化全部置1
        target_weight[:, 0] = joints_vis[:, 0] # 用关键点的可视标签joints_vis来初始化target_weight

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32) # 标签尺寸为(48,64)

            tmp_size = self.sigma * 3 # 高斯核半径,2*3=6

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size # 比例因子,(192,256)/(48,64)=(4,4)
                
                # (mu_x,mu_y)是关键点坐标的高斯中心
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                #print('mu_x:',mu_x) # 22,,,3
                #print('nu_y',mu_y) # 35,,,57
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                #print('ul:',ul) # [16,29],,,[-3,51]
                #print('br:',br) # [29,42],,,[10,64]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0 # 关键点的高斯核半径脱离了heatmap，则target_weight[joint_id]=0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1 # 两个高斯半径加中心点,2*6+1=13
                x = np.arange(0, size, 1, np.float32) # [0,1,2,...,12]
                y = x[:, np.newaxis] # 在newaxis处增加一个维度，(13,1),[[0],[1],[2],...,[12]]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
                #print(g)
                #print(g.shape) # (13,13)
                # gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                #print('g_x:',g_x) # (0,13),,,(3,13)
                #print('g_y:',g_y) # (0,13),,,(0,13)
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])
                #print('img_x:',img_x) # (16,29),,,(0,10)
                #print('img_y:',img_y) # (29,42),,,(51,64)

                v = target_weight[joint_id]
                # 若当前关键点可见，则target为高斯核，否则target为全0
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target, target_weight
```



（6）准确率指标的计算

```python
def accuracy(output, target, hm_type='gaussian', thr=0.5):
    # output和target在进入accuracy之前被转换成cpu上的numpy
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1])) # [0,1,2,...,16]
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output) # (size, 17, 2),batch_size个样本每个通道对应预测关键点和横纵坐标
        target, _ = get_max_preds(target) # (size, 17, 2),batch_size个样本每个通道对应真实关键点的横纵坐标
        h = output.shape[2] # 64
        w = output.shape[3] # 48
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        #print(norm.shape) # (size, 2)
        #print(norm) #[[6.4, 4.8], [6.4, 4.8], [6.4, 4.8],...]
    dists = calc_dists(pred, target, norm) # (17, size),(i,j)对应的值代表第j个样本在第i个关键点上的欧式距离（预测坐标和真实坐标）

    acc = np.zeros((len(idx) + 1)) # 18个值，第一个值为17个关键点的平均准确率，其余为17个关键点各自对应的准确率
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]]) # 返回第i+1个关键点在当前的batch_size中的准确率
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0 # 关键点个数
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred
```

准确率的计算代码写的非常复杂，涉及多个函数，需要慢慢debug去理解各个变量的含义。简单总结就是：训练过程中的每个batch_size输出的准确率是17个关键点的平均准确率，而每个关键点的准确率是batch_size个样本中真实关键点坐标和预测关键点坐标之间的欧式距离小于某个阈值的比例（代码中设置为0.5）。比如一个batch_size设置为32，在计算左眼关键点的准确率时，需要计算32个样本中每个样本的左眼预测坐标和真实坐标的欧式距离小于0.5的比例，假设有8个样本的预测坐标和真实坐标欧式距离小于0.5，则当前这个batch_size中左眼关键点的准确率为8/32=25%，再依次计算其余16个关键点，再求平均得到当前batch_size的准确率。

代码中的这个计算方法和$PCK$指标非常相似。



（7）COCODataset和MonkeyDataset数据类中的evaluate函数

在evaluate函数中调用了_do_python_keypoint_eval函数，该函数中使用cocoapi计算了数据集的各种指标：

```python
def _do_python_keypoint_eval(self, res_file, res_folder):
        # res_file:output/monkey/lpn_50/256x192_d256x3_adam_lr1e-3_lpn/results/keypoints_val_results.json
        # process_id = os.path.basename(res_file).split('_')[0]
        # os.path.basename返回路径的最后一个文件名，即keypoints_val_results.json
        # print(process_id) # keypoints
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        # print(info_str) # info_str中已经计算出各项指标

        eval_file = os.path.join(
            res_folder, 'keypoints_%s_results.pkl' % self.image_set)

        with open(eval_file, 'wb') as f:
            pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
        logger.info('=> coco eval results saved to %s' % eval_file)

        return info_str
```



##### 2.HRNet+UDP

1.损失函数

在训练代码的主函数中，有两种损失函数类型可以选择，代码如下：

```python
# define loss function (criterion) and optimizer
    if cfg.MODEL.TARGET_TYPE == 'gaussian':
        criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    elif cfg.MODEL.TARGET_TYPE == 'offset':
        criterion = JointsMSELoss_offset(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
        # criterion = JointsL1Loss_offset(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,reduction=cfg.LOSS.REDUCTION).cuda()
```

由于损失函数设计涉及到target heatmap如何生成，所以需要先读懂Dataset类。









#### 附录四、关键问题及解答

1.目标检测评价指标AR如何计算？



2.关键点检测中AP的计算是否用kpt_score来划分阈值生成（P,R）对？



