from config import get_config
from trainer import Trainer


def main(args):
    # trainer
    trainer = Trainer(args)

    # train
    if not args.test:
        trainer.train()
    # test
    else:
        trainer.test()


if __name__ == '__main__':
    args = get_config()
    print(args.automatic_entropy_tuning)
    # main(args)
