import gzip
import os
import struct
import urllib.request

import jax
import jax.numpy as jnp

SEED = 42
BATCH_SIZE = 5123

def mnist():
    filename = "train-images-idx3-ubyte.gz"
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    target_dir = os.getcwd() + "/data/mnist"
    url = f"{url_dir}/{filename}"
    target = f"{target_dir}/{filename}"

    if not os.path.exists(target):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {url} to {target}")

    with gzip.open(target, "rb") as fh:
        _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
        shape = (batch, 1, rows, cols)
        return jnp.frombuffer(fh.read(), dtype=jnp.uint8).reshape(shape)

def infinite_dataloader(data, batch_size, *, rng):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        rng, perm_rng = jax.random.split(rng)
        perm_indices = jax.random.permutation(perm_rng, indices)

        start, end = 0, batch_size
        while start < dataset_size:
            batch_indices = perm_indices[start:min(end, dataset_size)]
            start, end = end, end + batch_size
            yield data[batch_indices]


if __name__ == "__main__":

    data = mnist()
    dataset_size = data.shape[0]
    n_batches_per_epoch = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE

    rng = jax.random.PRNGKey(SEED)
    for batch_idx, batch in enumerate(infinite_dataloader(data, BATCH_SIZE, rng=rng)):
        print("EPOCH: {0:2d} | BATCH: {1:2d} | SHAPE: {2}".format(
            batch_idx // n_batches_per_epoch,
            batch_idx % n_batches_per_epoch,
            batch.shape,
        ))