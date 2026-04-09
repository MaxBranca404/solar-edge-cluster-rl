import numpy as np

# Cache per profili nuvole giornalieri
_cloud_cache = {}


def solar_power_realistic(t_sec: float, peak_power: float) -> float:
    """
    Modello realistico di potenza solare con nuvole.
    Alba: 6:00
    Tramonto: 18:00
    Potenza max: peak_power
    """

    DAY = 86400

    # Ora del giorno
    hours = (t_sec % DAY) / 3600.0

    # Notte
    if hours < 6.0 or hours > 18.0:
        return 0.0

    # =============================
    # Curva solare base
    # =============================
    solar_base = np.sin(np.pi * (hours - 6.0) / 12.0)

    # =============================
    # Giorno corrente
    # =============================
    day_index = int(t_sec // DAY)

    # =============================
    # Genera nuvole giornaliere
    # =============================
    if day_index not in _cloud_cache:

        samples = 144  # ogni 10 minuti

        noise = np.random.normal(0, 1, samples)

        # Smussatura → nuvole realistiche
        kernel = np.ones(15) / 15
        smooth = np.convolve(noise, kernel, mode="same")

        # Normalizzazione 0-1
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())

        # Attenuazione: 40% - 100%
        cloud_profile = 0.4 + 0.6 * smooth

        _cloud_cache[day_index] = cloud_profile

    cloud_profile = _cloud_cache[day_index]

    # =============================
    # Interpolazione temporale
    # =============================
    minutes = (t_sec % DAY) / 60.0
    idx = minutes / (24 * 60) * len(cloud_profile)

    i0 = int(idx)
    i1 = min(i0 + 1, len(cloud_profile) - 1)

    alpha = idx - i0

    cloud_factor = (
        cloud_profile[i0] * (1 - alpha)
        + cloud_profile[i1] * alpha
    )

    # =============================
    # Potenza finale
    # =============================
    power = peak_power * solar_base * cloud_factor

    return max(power, 0.0)