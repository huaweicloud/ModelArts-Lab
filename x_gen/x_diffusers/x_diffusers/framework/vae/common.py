import torch
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput


def encode_with_slicing(autoencoder, x: torch.Tensor, return_dict: bool = True):
    if autoencoder.use_slicing and x.shape[0] > 1:
        encoded_slices = [autoencoder._encode(x_slice) for x_slice in x.split(1)]
        h = torch.cat(encoded_slices)
    else:
        h = autoencoder._encode(x)

    posterior = DiagonalGaussianDistribution(h)

    if not return_dict:
        return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)


def decode_with_slicing(autoencoder, z: torch.Tensor, return_dict: bool = True):
    if autoencoder.use_slicing and z.shape[0] > 1:
        decoded_slices = [autoencoder._decode(z_slice).sample for z_slice in z.split(1)]
        decoded = torch.cat(decoded_slices)
    else:
        decoded = autoencoder._decode(z).sample

    if not return_dict:
        return (decoded,)
    return DecoderOutput(sample=decoded)
