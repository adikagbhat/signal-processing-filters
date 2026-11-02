"""
Digital Filter Design for Tactical Audio Denoising
Butterworth Bandpass and Lowpass filtering via SciPy / NumPy.
"""
import numpy as np
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5):
    """Generates IIR Butterworth bandpass filter coefficients (b, a)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5
) -> np.ndarray:
    """Filters 1D audio or communication waveform to remove high/low band interference."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


if __name__ == "__main__":
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0
    t = np.linspace(0, 0.05, int(fs * 0.05), endpoint=False)
    # Synthetic test signal: 600 Hz (passband) + 2000 Hz (noise)
    sig = np.sin(2 * np.pi * 600 * t) + 1.5 * np.cos(2 * np.pi * 2000 * t)
    filtered = apply_bandpass_filter(sig, lowcut, highcut, fs, order=3)
    print("Filter execution completed successfully.")
    print("Filtered output sample count:", len(filtered))
