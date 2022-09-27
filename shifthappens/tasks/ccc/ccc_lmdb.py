"""
Code copied from: https://github.com/rmccorm4/PyTorch-LMDB
"""
import os

import lmdb
import pyarrow as pa
import six
import torch.utils.data as data
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


def dset2lmdb(dataset, outpath, write_frequency=5000):
    data_loader = DataLoader(dataset, collate_fn=lambda x: x)
    """
        Saves a given dataset in LMDB format

        Parameters
        ----------
        dataset :
            DataSet object that you want to save
        outpath : str
            path to save generated files
        write_frequency: int
            write frequency
        Returns
        -------
    """
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = os.path.expanduser(outpath)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=1099511627776 * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = db.begin(write=True)
    for idx, sample in enumerate(data_loader):
        image, label = sample[0]
        txn.put("{}".format(idx).encode("ascii"), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(len(data_loader))]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dumps_pyarrow(keys))
        txn.put(b"__len__", dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
    print("Closing")
