import os

random_seed = 20200804
os.environ['PYTHONHASHSEED'] = str(random_seed)
import argparse
import collections
import random

random.seed(random_seed)
import numpy as np

np.random.seed(random_seed)

import paddle

paddle.seed(random_seed)
import paddle.optimizer as optim
from paddle.vision import transforms

import model
from test import run_from_train
from dataloader import CSVDataset, collater, AspectRatioBasedSampler, Augmenter, Normalizer, \
    PhotometricDistort, RandomSampleCrop
from paddle.io import DataLoader

# assert torch.__version__.split('.')[1] == '4'


print('CUDA available: {}'.format(paddle.is_compiled_with_cuda()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a CTracker network.')

    parser.add_argument('--dataset', default='csv', type=str, help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--model_dir', default='./ctracker/', type=str, help='Path to save the model.')
    parser.add_argument('--root_path', default=r'', type=str,
                        help='Path of the directory containing both label and images')
    parser.add_argument('--csv_train', default='train_annots.csv', type=str,
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='train_labels.csv', type=str,
                        help='Path to file containing class list (see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=60)

    parser = parser.parse_args(args)
    print(parser)

    print(parser.model_dir)
    if not os.path.exists(parser.model_dir):
        os.makedirs(parser.model_dir)

    # Create the data loaders
    if parser.dataset == 'csv':
        if (parser.csv_train is None) or (parser.csv_train == ''):
            raise ValueError('Must provide --csv_train when training on COCO,')

        if (parser.csv_classes is None) or (parser.csv_classes == ''):
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(parser.root_path, train_file=os.path.join(parser.root_path, parser.csv_train),
                                   class_list=os.path.join(parser.root_path, parser.csv_classes),
                                   transform=transforms.Compose(
                                       [RandomSampleCrop(), PhotometricDistort(), Augmenter(), Normalizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=4,
                                      drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler)

    # Create the model
    if parser.depth == 50:
        retinanet = model.resnext50_32x4d(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 50, 101')

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.to()

    retinanet = paddle.DataParallel(retinanet).to()

    print('Number of models parameters: {}'.format(
        sum([p.data.size() for p in retinanet.parameters()])))

    retinanet.training = True

    optimizer = optim.Adam(parameters=retinanet.parameters(), learning_rate=5e-5)

    scheduler = optim.lr.ReduceOnPlateau(optimizer.get_lr(), patience=3, verbose=True)

    loss_hist = collections.deque(
        maxlen=500)

    retinanet.train()
    retinanet._layers.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    total_iter = 0
    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet._layers.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                total_iter = total_iter + 1
                optimizer.clear_gradients()

                (classification_loss, regression_loss), reid_loss = retinanet([data['img'].cuda().astype(dtype='float32'), data['annot'], data['img_next'].cuda().astype(dtype='float32'), data['annot_next']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                reid_loss = reid_loss.mean()

                # loss = classification_loss + regression_loss + track_classification_losses
                loss = classification_loss + regression_loss + reid_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                paddle.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iter: {} | Cls loss: {:1.5f} | Reid loss: {:1.5f} | Reg loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(reid_loss), float(regression_loss),
                        np.mean(loss_hist)))
            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

    retinanet.eval()

    paddle.save(retinanet, os.path.join(parser.model_dir, 'model_final.pt'))
    run_from_train(parser.model_dir, parser.root_path)


if __name__ == '__main__':
    main()
