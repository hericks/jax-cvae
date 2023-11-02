import gzip
import os
import struct
import urllib.request

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

# GENERAL
SEED = 42

# NETWORK ARCHITECTURE
HIDDEN_DIM = 32
LATENT_DIM = 2

# TRAINING
N_EPOCHS = 10
BATCH_SIZE = 500


class Encoder(eqx.Module):
    hidden: eqx.nn.Linear
    logvar: eqx.nn.Linear
    mean: eqx.nn.Linear

    def __init__(self, *, rng):
        hidden_rng, mean_rng, logvar_rng = jax.random.split(rng, 3)
        self.hidden = eqx.nn.Linear(
            in_features=28*28, out_features=HIDDEN_DIM, key=hidden_rng
        )
        self.logvar = eqx.nn.Linear(
            in_features=HIDDEN_DIM, out_features=LATENT_DIM, key=logvar_rng
        )
        self.mean = eqx.nn.Linear(
            in_features=HIDDEN_DIM, out_features=LATENT_DIM, key=mean_rng
        )

    def __call__(self, x):
        x = jnp.ravel(x)
        x = self.hidden(x)
        x = jax.nn.sigmoid(x)
        return self.mean(x), self.logvar(x)

class Decoder(eqx.Module):
    hidden: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(self, *, rng):
        hidden_rng, output_rng = jax.random.split(rng)
        self.hidden = eqx.nn.Linear(
            in_features=LATENT_DIM, out_features=HIDDEN_DIM, key=hidden_rng
        )
        self.output = eqx.nn.Linear(
            in_features=HIDDEN_DIM, out_features=28*28, key=output_rng
        )

    def __call__(self, z):
        z = self.hidden(z)
        z = jax.nn.sigmoid(z)
        z = self.output(z)
        z = jax.nn.sigmoid(z)
        return jnp.reshape(z, (1, 28, 28))
    
class VAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, *, rng):
        encoder_rng, decoder_rng = jax.random.split(rng)
        self.encoder = Encoder(rng=encoder_rng)
        self.decoder = Decoder(rng=decoder_rng)

    def __call__(self, x, *, rng):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar, rng=rng)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

    def reparameterize(self, mean, logvar, *, rng):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, mean.shape)
        return mean + eps * std
    
def mnist():
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    target_dir = os.getcwd() + "/data/mnist"
    filenames = {
        "images": "train-images-idx3-ubyte.gz",
        "labels": "train-labels-idx1-ubyte.gz",
    }

    # download images and labels into data folder
    for _, filename in filenames.items():
        url = f"{url_dir}/{filename}"
        target = f"{target_dir}/{filename}"
        if not os.path.exists(target):
            os.makedirs(target_dir, exist_ok=True)
            urllib.request.urlretrieve(url, target)
            print(f"Downloaded {url} to {target}")

    # load images into memory
    target = f"{target_dir}/{filenames['images']}"
    with gzip.open(target, "rb") as fh:
        _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
        shape = (batch, 1, rows, cols)
        images = jnp.frombuffer(fh.read(), dtype=jnp.uint8).reshape(shape)

    # load labels into memory 
    target = f"{target_dir}/{filenames['labels']}"
    with gzip.open(target, "rb") as fh:
        _, batch = struct.unpack(">II", fh.read(8))
        shape = (batch, 1,)
        labels = jnp.frombuffer(fh.read(), dtype=jnp.uint8).reshape(shape)
    
    return images, labels

def infinite_dataloader(images, labels, batch_size, *, rng):
    dataset_size = images.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        rng, perm_rng = jax.random.split(rng)
        perm_indices = jax.random.permutation(perm_rng, indices)

        start, end = 0, batch_size
        while start < dataset_size:
            batch_indices = perm_indices[start:min(end, dataset_size)]
            start, end = end, end + batch_size
            yield images[batch_indices], labels[batch_indices]

def reconstruction_error(mean, sample):
    # TODO. Add variance term
    return 0.5 * jnp.mean(jnp.square(mean - sample))

def kullback_leibler_divergence(mean, logvar):
    return - 0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def vae_loss(model, x, *, rng):
    rng_batch = jax.random.split(rng, x.shape[0])

    x_recon, mean, logvar = jax.vmap(model)(x, rng=rng_batch)

    kl = jax.vmap(kullback_leibler_divergence)(mean, logvar)
    neg_recon_error = - jax.vmap(reconstruction_error)(x_recon, x)
    elbo = - kl + neg_recon_error

    return - jnp.mean(elbo)


if __name__ == "__main__":

    images, labels = mnist()
    dataset_size = images.shape[0]
    n_batches_per_epoch = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE
    n_batches = n_batches_per_epoch * N_EPOCHS

    rng = jax.random.PRNGKey(SEED)
    rng, model_rng, dataloader_rng = jax.random.split(rng, 3)

    # initialize model and optimizer state
    vae = VAE(rng=model_rng)
    optim = optax.adam(learning_rate=3e-4)
    opt_state = optim.init(eqx.filter(vae, eqx.is_array))

    # initialize dataloader
    train_dataloader = infinite_dataloader(images / 255, labels, BATCH_SIZE, rng=dataloader_rng)

    for batch_idx, (image_batch, label_batch) in zip(range(n_batches), train_dataloader):
        # split PRNG key
        rng, loss_rng = jax.random.split(rng)

        # compute loss value and gradient
        loss, grads = eqx.filter_value_and_grad(vae_loss)(vae, image_batch, rng=loss_rng)

        # transform gradient and apply updates to model
        updates, opt_state = optim.update(grads, opt_state, vae)
        vae = eqx.apply_updates(vae, updates)

        # log metrics
        epoch, batch = divmod(batch_idx, n_batches_per_epoch)
        logging_components = [
            "EPOCH: {0:2d}".format(epoch),
            "BATCH: {0:3d}".format(batch),
            "LOSS: {0:.4f}".format(loss),
        ]
        print(" | ".join(logging_components))
