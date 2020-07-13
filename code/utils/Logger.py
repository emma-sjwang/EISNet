from time import time
from os.path import join, dirname
from torch.utils.tensorboard import SummaryWriter
import yaml
_log_path = join(dirname(__file__), '../logs')


# high level wrapper for tf_logger.TFLogger
class Logger():
    def __init__(self, args, update_frequency=10):
        folder, logname = self.get_name_from_args(args)

        log_path = join(_log_path, folder, logname)
        self.writer = SummaryWriter(log_path)
        print("Saving to %s" % log_path)

        with open(join(log_path, 'config.yaml'), 'w') as f:
            yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    
    @staticmethod
    def get_name_from_args(args):
        folder_name = "%s_to_%s" % ("-".join(sorted(args.source)), args.target)
        name = ""
        if args.moco:
            name += "moco%f_" % args.alpha

        if args.folder_name:
            folder_name = join(args.folder_name, folder_name)
        name += "eps%d_bs%d_lr%g_jigw%g_Mocow%g_ncek%g_margin%g_tripletk%g" % (args.epochs, args.batch_size,
                                                                    args.learning_rate,
                                                   args.jig_weight, args.moco_weight, args.nce_k, args.margin, args.k_triplet)


        name += "_%d" % int(time() % 1000)
        return folder_name, name

