from pyod.models.vae import VAE, VAEModel


class ContinualVAE(VAE):
    """VAE that warm-starts from existing weights on repeated fit() calls.

    Identical to PyOD's VAE except build_model() skips model creation if a
    model is already present, enabling continual fine-tuning across concepts.
    The optimizer is still reset each call (fresh Adam state), but the network
    weights carry over.
    """

    def build_model(self):
        if getattr(self, 'model', None) is not None:
            return
        self.model = VAEModel(
            self.feature_size,
            encoder_neuron_list=self.encoder_neuron_list,
            decoder_neuron_list=self.decoder_neuron_list,
            latent_dim=self.latent_dim,
            hidden_activation_name=self.hidden_activation_name,
            output_activation_name=self.output_activation_name,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
            logvar_clip=self.logvar_clip,
        )
