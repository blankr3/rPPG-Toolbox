import numpy as np


class PhysicsSkinAugmentation:
    """Physics-based skin tone augmentation using a two-component reflection model.

    Models how melanin concentration affects the rPPG signal by separately
    treating surface reflection (unaffected by melanin) and deep tissue
    reflection (attenuated by double-pass through the melanin-bearing epidermis).

    The key insight: DiffNormalized = (f[t+1]-f[t]) / (f[t+1]+f[t]) is invariant
    to uniform multiplicative attenuation (it cancels in the ratio). Real skin has
    two reflection components:
      - Surface (Fresnel, ~4-7% of total): unaffected by melanin
      - Deep (dermis, carries the pulsatile signal): attenuated by a² (double-pass)

    This produces a physically-motivated scale factor per channel:
        scale_c = a_c² / (r + a_c²·(1-r))

    where a_c = exp(-μ_melanin(λ_c) · Δf · l_eff) and r is the surface reflection
    fraction. As melanin increases, DiffNormalized amplitude drops — matching the
    real-world SNR degradation observed in darker skin tones.

    References:
        Jacques, S.L. (2013). "Optical properties of biological tissues: a review."
            Physics in Medicine & Biology. https://pubmed.ncbi.nlm.nih.gov/23666068/
        Meglinski, I.V. & Matcher, S.J. (2002). "Quantitative assessment of skin
            layers absorption and skin reflectance spectra simulation in the visible
            and near-infrared spectral regions." Physiological Measurement.
    """

    def __init__(self, p=0.5, delta_f_range=(0.0, 0.3), surface_reflection=0.10):
        """
        Args:
            p: Probability of applying augmentation per sample.
            delta_f_range: Range for melanin volume fraction shift.
                (0.0, 0.3) roughly covers Fitzpatrick I → VI.
            surface_reflection: Fraction of reflected light from the skin surface
                (Fresnel reflection at stratum corneum, typically 4-7%).
        """
        self.p = p
        self.delta_f_range = delta_f_range
        self.r = surface_reflection

        # Melanin absorption coefficients from Jacques (2013):
        #   μ_a(λ) = 1.70 × 10¹² · λ^(-3.48)  [cm⁻¹, λ in nm]
        # Center wavelengths for typical RGB camera (e.g., Logitech C920):
        lambdas = np.array([660.0, 530.0, 450.0])  # R, G, B in nm
        self.mu_a_mel = 1.70e12 * np.power(lambdas, -3.48)  # [cm⁻¹]

        # Effective pathlength: melanin is in the stratum basale (~50 μm).
        # Double-pass is accounted for by using a² in the scale formula.
        self.l_eff = 0.005  # 50 μm = 0.005 cm

        # Log scale factor range so users can verify augmentation strength
        scale_min = self._compute_diffnorm_scale(self._compute_attenuation(delta_f_range[1]))
        print(f"[PhysicsSkinAugmentation] p={p}, r={surface_reflection}, "
              f"delta_f={delta_f_range}")
        print(f"  Scale at max delta_f: R={scale_min[0]:.3f}, "
              f"G={scale_min[1]:.3f}, B={scale_min[2]:.3f}")

    def _sample_delta_f(self):
        """Sample a random melanin volume fraction shift."""
        return np.random.uniform(*self.delta_f_range)

    def _compute_attenuation(self, delta_f):
        """Compute per-channel single-pass and double-pass attenuation.

        Returns:
            a_sq: Double-pass attenuation, shape (3,).
        """
        a = np.exp(-self.mu_a_mel * delta_f * self.l_eff)
        return (a ** 2).astype(np.float32)

    def _compute_diffnorm_scale(self, a_sq):
        """Compute DiffNormalized scale factor from double-pass attenuation.

        scale_c = a_c² / (r + a_c²·(1-r))

        When a=1 (no melanin): scale=1.
        When a→0 (max melanin): scale→0 (signal vanishes).
        When r=0 (no surface reflection): scale=1 (uniform attenuation cancels).
        """
        return a_sq / (self.r + a_sq * (1.0 - self.r))

    def __call__(self, diff_norm_data, raw_data):
        """Apply consistent physics-based augmentation to both branches.

        Both branches receive the same sampled melanin shift so they represent
        the same simulated skin tone.

        Args:
            diff_norm_data: (T, H, W, 3) DiffNormalized video data.
            raw_data: (T, H, W, 3) raw RGB video data (for Standardized branch).

        Returns:
            aug_diff_norm: (T, H, W, 3) augmented DiffNormalized.
            aug_raw: (T, H, W, 3) attenuated raw RGB (to be standardized by caller).
        """
        if np.random.rand() > self.p:
            return diff_norm_data, raw_data

        delta_f = self._sample_delta_f()
        a_sq = self._compute_attenuation(delta_f)  # (3,)

        # DiffNormalized: apply the two-component scale factor per channel
        scale = self._compute_diffnorm_scale(a_sq)  # (3,)
        aug_diff_norm = diff_norm_data * scale  # broadcasts over (T, H, W, 3)

        # Raw/Standardized: uniform double-pass attenuation teaches the
        # appearance branch color invariance across skin tones
        aug_raw = (raw_data * a_sq).astype(np.float32)

        return aug_diff_norm, aug_raw
