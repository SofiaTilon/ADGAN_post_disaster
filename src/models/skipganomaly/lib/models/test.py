"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model

##
def main():
    """ Training
    """
    """ Training
        """
    opt = Options().parse()
    data = load_data(opt)
    model = load_model(opt, data)
    res, scores = model.test()
    model.save_anomaly_scores(scores, savedir="test", fname=opt.anom_file_name)

if __name__ == '__main__':
    main()
