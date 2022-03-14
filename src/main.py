from argparse import ArgumentParser

from data import create_loaders

parser = ArgumentParser()
parser.add_argument('--data', type=str, default='../data/train.npy', help='Training data path')
parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='Testing batch size')
parser.add_argument('--loader_threads', type=int, default=8, help='Number of threads of the data loader')
args = parser.parse_args()

train_loader, test_loader = create_loaders(args.data, args.train_batch_size, args.test_batch_size, args.loader_threads)
