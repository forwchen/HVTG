import config
from utils import *
from trainer_mgpu import Trainer

logger = get_logger()



def main(args):
    set_gpu(args)
    prepare_dirs(args)

    save_args(args)
    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test()
    else:
        trainer.infer()



if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
