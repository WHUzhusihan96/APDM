import yaml
import easydict
from os.path import join


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


def load_config_file(config_file):
    args = yaml.safe_load(open(config_file))
    args = easydict.EasyDict(args)
    save_config = yaml.safe_load(open(config_file))
    dataset = None
    if args.data.dataset.name == 'office':
        dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['amazon', 'dslr', 'webcam'],
        files=[
            'amazon.txt',
            'dslr.txt',
            'webcam.txt'
        ],
        prefix=args.data.dataset.root_path)
    elif args.data.dataset.name == 'DG':
        dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['AID', 'CLRS', 'MLRSN', 'RSSCN7'],
        files=[
            'AID.txt',
            'CLRS.txt',
            'MLRSN.txt',
            'RSSCN7.txt'
        ],
        prefix=args.data.dataset.root_path)
    elif args.data.dataset.name == 'CSC':
        dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['AID', 'CLRS', 'MLRSN', 'OPTIMAL-31'],
        files=[
            'AID.txt',
            'CLRS.txt',
            'MLRSN.txt',
            'OPTIMAL-31.txt'
        ],
        prefix=args.data.dataset.root_path)
    elif args.data.dataset.name == 'officehome':
        dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['Art', 'Clipart', 'Product', 'Real_World'],
        files=[
            'Art.txt',
            'Clipart.txt',
            'Product.txt',
            'Real_World.txt'
        ],
        prefix=args.data.dataset.root_path)
    elif args.data.dataset.name == 'visda2017':
        dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['train', 'validation'],
        files=[
            'train/image_list.txt',
            'validation/image_list.txt',
        ],
        prefix=args.data.dataset.root_path)
        dataset.prefixes = [join(dataset.path, 'train'), join(dataset.path, 'validation')]
    else:
        raise Exception(f'dataset {args.data.dataset.name} not supported!')

    source_domain_name = dataset.domains[args.data.dataset.source]
    target_domain_name = dataset.domains[args.data.dataset.target]
    source_file = dataset.files[args.data.dataset.source]
    target_file = dataset.files[args.data.dataset.target]

    args.source_domain_name = source_domain_name
    args.target_domain_name = target_domain_name
    args.source_file = source_file
    args.target_file = target_file
    print("load config successful")
    return args, dataset, save_config
