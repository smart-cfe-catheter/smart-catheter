import argparse

models = ['fcn', 'rnn', 'transformer']


def get_model_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, choices=models)
    return parser


def add_train_args(parser):
    group = parser.add_argument_group('train')
    group.add_argument('--seed', type=int, default=7777,
                       help="Random seed.")
    group.add_argument('--epochs', type=int, default=10,
                       help="Number of epochs for training.")
    group.add_argument('--batch_size', type=int, default=32,
                       help="Number of instances in a batch.")
    group.add_argument('--lr', type=float, default=1e-5,
                       help="Learning rate.")
    group.add_argument('--log_interval', type=int, default=10,
                       help="Log interval.")
    group.add_argument('--model', type=str, choices=models,
                       help="Task name for training.")
    group.add_argument('--train_data', type=str,
                       help="Root directory of train data.")
    group.add_argument('--valid_data', type=str,
                       help="Root directory of validation data.")
    group.add_argument('--device', type=str,
                       help="Device going to use for training.")
    group.add_argument('--save_dir', type=str, default='checkpoints/',
                       help="Folder going to save model checkpoints.")
    group.add_argument('--log_dir', type=str, default='logs/',
                       help="Folder going to save logs.")


def add_test_args(parser):
    group = parser.add_argument_group('test')
    group.add_argument('--batch_size', type=int, default=32,
                       help="Number of instances in a batch.")
    group.add_argument('--test_data', type=str,
                       help="Root directory of test data.")
    group.add_argument('--device', type=str,
                       help="Device going to use for training.")
    group.add_argument('--ckpt_dir', type=str,
                       help="Directory which contains the checkpoint and args.json.")
