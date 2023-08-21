import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """ 

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        #parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='JSCC_OFDM', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='JSCCOFDM', help='chooses which model to use. [JSCCOFDM | JSCC ]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--max_ngf', type=int, default=256, help='maximal # of gen filters in the last conv layer')
        parser.add_argument('--gan_mode', type=str, default='none', help='choose from [wgangp | lsgan | vanilla | none]')
        parser.add_argument('--label_smooth', type=int, default=1, help='label smoothing factor for lsgan and vanilla gan')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if gan_mode != none')
        parser.add_argument('--norm_D', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--n_downsample', type=int, default=2, help='number of downsampling')
        parser.add_argument('--n_blocks', type=int, default=2, help='number of residual blocks in either encoder or generator')
        parser.add_argument('--first_kernel', type=int, default=5, help='kernal size of the first conv layer in encoder')
        parser.add_argument('--C_channel', type=int, default=12, help='output channels for the latent vector')
        parser.add_argument('--activation', type=str, default='sigmoid', help='output activation, choose from [sigmoid | tanh]')
        parser.add_argument('--norm_EG', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # OFDM parameters
        parser.add_argument('--P', type=int, default=1, help='number of packets for each transmitted image')
        parser.add_argument('--S', type=int, default=6, help='number of OFDM symbols per packet')
        parser.add_argument('--M', type=int, default=64, help='number of subcarriers per symbol')
        parser.add_argument('--K', type=int, default=16, help='length of cyclic prefix')
        parser.add_argument('--L', type=int, default=8, help='length of multipath channel')
        parser.add_argument('--decay', type=int, default=4, help='decay constant for the multipath channel')
        parser.add_argument('--is_clip', action='store_true', help='whether to include clipping')
        parser.add_argument('--CR', type=float, default=1.0, help='clipping ratio')
        parser.add_argument('--N_pilot', type=int, default=2, help='number of pilot symbols for channel estimation')
        parser.add_argument('--pilot', type=str, default='QPSK', help='type of pilots, choose from [QPSK | ZadoffChu]')
        parser.add_argument('--CE', type=str, default='LMMSE', help='channel estimation method, choose from [LS | LMMSE | TRUE]')
        parser.add_argument('--EQ', type=str, default='MMSE', help='equalization method, choose from [ZF | MMSE]')
        parser.add_argument('--SNR', type=float, default=20.0, help='SNR')
        parser.add_argument('--is_feedback', action='store_true', help='Wether to provide CSI feedback to the encoder')
        parser.add_argument('--feedforward', type=str, default='EXPLICIT-RES', help='which decoder design to use, choose from [IMPLICIT | EXPLICIT-CE | EXPLICIT-CE-EQ | EXPLICIT-RES]')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='CIFAR10', help='chooses how datasets are loaded. [CIFAR10 | CelebA]')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        #model_name = opt.model
        #model_option_setter = models.get_option_setter(model_name)
        #parser = model_option_setter(parser, self.isTrain)
        #opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        #dataset_name = opt.dataset_mode
        #dataset_option_setter = data.get_option_setter(dataset_name)
        #parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
