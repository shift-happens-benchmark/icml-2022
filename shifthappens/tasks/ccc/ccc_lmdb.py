"""
Code copied from: https://github.com/rmccorm4/PyTorch-LMDB
"""
import os

import lmdb
import pyarrow as pa
import six
import torch.utils.data as data
import tqdm
from PIL import Image
from torch.utils.data import DataLoader


class ImageFolderLMDB(data.Dataset):
    """
    Saves a Dataset object as LMDB files
    """

    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=os.path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b"__len__"))
            self.keys = pa.deserialize(txn.get(b"__keys__"))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def dset2lmdb(dataset, outpath, subset_size):
    """
    Saves a given dataset in LMDB format

    Parameters
    ----------
    dataset :
        DataSet object that you want to save
    outpath : str
        path to save generated files
    subset_size: int
        amount of images in dataset
    """
    data_loader = DataLoader(dataset, num_workers=0, collate_fn=lambda x: x)

    lmdb_path = os.path.expanduser(outpath)
    isdir = os.path.isdir(lmdb_path)

    if subset_size == 50000:
        map_size = 2048 * 2048 * 512  # allocates 2GB
    elif subset_size == 5000:
        map_size = 2048 * 2048 * 128  # allocates 512MB
    else:
        map_size = 2048 * 2048 * 2048 * 256  # allocates 1TB

    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = db.begin(write=True)
    for idx, sample in tqdm.tqdm(
        enumerate(data_loader),
        total=len(dataset),
        desc="Generate LMDB to %s" % lmdb_path,
    ):
        image, label = sample[0]
        txn.put(f"{idx}".encode("ascii"), dumps_pyarrow((image, label)))

    # finish iterating through dataset
    txn.commit()
    keys = [f"{k}".encode("ascii") for k in range(len(data_loader))]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dumps_pyarrow(keys))
        txn.put(b"__len__", dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
    print("Closing")
