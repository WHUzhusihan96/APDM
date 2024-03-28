from utils.config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler


def load_data(args, dataset):
    a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
    c = c - a - b
    common_classes = [i for i in range(a)]
    source_private_classes = [i + a for i in range(b)]
    target_private_classes = [i + a + b for i in range(c)]

    source_classes = common_classes + source_private_classes
    args.source_classes = source_classes
    target_classes = common_classes + target_private_classes
    args.target_classes = target_classes

    train_transform = Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor()
    ])

    test_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor()
    ])
    # FileListDataset labels and datas are labels and file_path for each sample
    # And the __getitem__ can get the im and label for corresponding index.
    source_train_ds = FileListDataset(list_path=args.source_file,
                                      path_prefix=dataset.prefixes[args.data.dataset.source],
                                      transform=train_transform, filter=(lambda x: x in source_classes))
    source_test_ds = FileListDataset(list_path=args.source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                     transform=test_transform, filter=(lambda x: x in source_classes))
    target_train_ds = FileListDataset(list_path=args.target_file,
                                      path_prefix=dataset.prefixes[args.data.dataset.target],
                                      transform=train_transform, filter=(lambda x: x in target_classes))
    target_test_ds = FileListDataset(list_path=args.target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                     transform=test_transform, filter=(lambda x: x in target_classes))

    classes = source_train_ds.labels
    freq = Counter(classes)
    class_weight = {x: 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

    source_weights = [class_weight[x] for x in source_train_ds.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

    source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                                 sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
    source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                                num_workers=1, drop_last=False)
    target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size, shuffle=True,
                                 num_workers=args.data.dataloader.data_workers, drop_last=True)
    target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                                num_workers=1, drop_last=False)

    return source_train_dl, source_test_dl, target_train_dl, target_test_dl

