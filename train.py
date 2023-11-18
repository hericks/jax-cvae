import gzip
import os
import struct
import urllib.request

import configargparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


class Encoder(eqx.Module):
    hidden: eqx.nn.Linear
    logvar: eqx.nn.Linear
    mean: eqx.nn.Linear

    def __init__(self, hidden_dim, latent_dim, *, rng):
        hidden_rng, mean_rng, logvar_rng = jax.random.split(rng, 3)
        self.hidden = eqx.nn.Linear(
            in_features=28*28, out_features=hidden_dim, key=hidden_rng
        )
        self.logvar = eqx.nn.Linear(
            in_features=hidden_dim, out_features=latent_dim, key=logvar_rng
        )
        self.mean = eqx.nn.Linear(
            in_features=hidden_dim, out_features=latent_dim, key=mean_rng
        )

    def __call__(self, x):
        x = jnp.ravel(x)
        x = self.hidden(x)
        x = jax.nn.sigmoid(x)
        return self.mean(x), self.logvar(x)

class Decoder(eqx.Module):
    hidden: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(self, hidden_dim, latent_dim, *, rng):
        hidden_rng, output_rng = jax.random.split(rng)
        self.hidden = eqx.nn.Linear(
            in_features=latent_dim, out_features=hidden_dim, key=hidden_rng
        )
        self.output = eqx.nn.Linear(
            in_features=hidden_dim, out_features=28*28, key=output_rng
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

    def __init__(self, hidden_dim, latent_dim, *, rng):
        encoder_rng, decoder_rng = jax.random.split(rng)
        self.encoder = Encoder(hidden_dim, latent_dim, rng=encoder_rng)
        self.decoder = Decoder(hidden_dim, latent_dim, rng=decoder_rng)

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

def gaussian_log_likelihood(mean, sample):
    # TODO. Add variance term
    return 0.5 * jnp.sum(jnp.square(mean - sample))

def bernoulli_log_likelihood(logits, labels):
    # with soft labels
    return - jnp.sum(labels * jnp.log(logits) + (1.0 - labels) * jnp.log(1 - logits))

def reconstruction_error(mean, sample):
    return bernoulli_log_likelihood(logits=mean, labels=sample)

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
    # initialize argparser
    p = configargparse.ArgParser()

    # seed
    p.add_argument('--seed', type=int, default=42)

    # architecture
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--latent_dim', type=int, default=6)

    # training / optimisation
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--n_epochs', default=10)
    p.add_argument('--batch_size', default=128)

    # obtain and log arguments to console
    args = p.parse_args()
    print(p.format_values())

    # prepare training parameters
    images, labels = mnist()
    dataset_size = images.shape[0]
    n_batches_per_epoch = (dataset_size + args.batch_size - 1) // args.batch_size
    n_batches = n_batches_per_epoch * args.n_epochs

    # obtain keys for pseudo-random number generators
    rng = jax.random.PRNGKey(args.seed)
    rng, model_rng, dataloader_rng = jax.random.split(rng, 3)

    # initialize model and optimizer state
    vae = VAE(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, rng=model_rng)
    optim = optax.adam(learning_rate=args.learning_rate)
    opt_state = optim.init(eqx.filter(vae, eqx.is_array))

    # initialize dataloader
    train_dataloader = infinite_dataloader(images / 255, labels, args.batch_size, rng=dataloader_rng)

    # define jitted training step function
    @eqx.filter_jit
    def make_step(vae, opt_state, x, *, rng):
        loss, grads = eqx.filter_value_and_grad(vae_loss)(vae, x, rng=rng)
        updates, opt_state = optim.update(grads, opt_state, vae)
        vae = eqx.apply_updates(vae, updates)
        return vae, opt_state, loss

    # training loop
    for batch_idx, (image_batch, label_batch) in zip(range(n_batches), train_dataloader):
        # split PRNG key
        rng, step_rng = jax.random.split(rng)

        # perform training step
        vae, opt_state, loss = make_step(vae, opt_state, image_batch, rng=step_rng)

        # log metrics
        epoch, batch = divmod(batch_idx, n_batches_per_epoch)

        if batch == 0:
            losses = np.zeros((n_batches_per_epoch,))

        losses[batch] = loss

        if batch == n_batches_per_epoch - 1:
            logging_components = [
                "EPOCH: {0:2d}".format(epoch),
                "BATCH: {0:3d}".format(batch),
                "LOSS: {0:.4f}".format(losses.mean()),
            ]
            print(" | ".join(logging_components))
