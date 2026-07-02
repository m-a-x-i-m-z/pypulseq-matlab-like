from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from pypulseq.utils.siemens.asc_to_hw import asc_to_acoustic_resonances
from pypulseq.utils.siemens.readasc import readasc


def calculate_gradient_spectrum(
    obj,
    max_frequency: float = 3000.0,
    window_width: float = 0.05,
    frequency_oversampling: float = 3.0,
    time_range: Union[List[float], None] = None,
    plot: bool = True,
    combine_mode: str = 'rss',
    use_derivative: bool = False,
    acoustic_resonances: Union[List[dict], str, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if acoustic_resonances is None:
        acoustic_resonances = []

    if isinstance(acoustic_resonances, str):
        asc, _ = readasc(acoustic_resonances)
        acoustic_resonances = asc_to_acoustic_resonances(asc)

    dt = obj.system.grad_raster_time
    nwin = int(round(window_width / dt))
    if nwin <= 0:
        nwin = 1
    os = int(round(frequency_oversampling))
    if os <= 0:
        os = 1

    faxis = np.arange(0, nwin / 2) / nwin / dt / os
    nfmax = int(np.sum(faxis <= max_frequency))

    wave_src = obj.waveforms_and_times() if time_range is None else obj.waveforms(time_range=time_range)
    wave_data = wave_src[0] if isinstance(wave_src, tuple) else wave_src
    ng = len(wave_data)
    tmax = 0.0
    for i in range(ng):
        if wave_data[i] is not None and wave_data[i].shape[1] > 0:
            tmax = max(tmax, wave_data[i][0, -1])
    if tmax == 0:
        raise ValueError('Empty sequence passed to gradSpectrum()')

    nt = int(np.ceil(tmax / dt))
    gw = np.zeros((ng, nt))
    t = (np.arange(1, nt + 1) - 0.5) * dt
    for i in range(ng):
        if wave_data[i] is not None and wave_data[i].shape[1] > 0:
            gw[i, :] = np.interp(t, wave_data[i][0, :], wave_data[i][1, :], left=0.0, right=0.0)

    if use_derivative:
        gw = np.diff(gw, axis=1)

    gs = []
    for g in range(ng):
        x = gw[g, :]
        nx = len(x)
        nx = int(np.ceil(nx / nwin) * nwin)
        if nx > len(x):
            x = np.concatenate((x, np.zeros(nx - len(x))))

        nseg1 = nx // nwin
        xseg = np.zeros((nseg1 * 2 - 1, int(nwin * os)))
        xseg[0::2, :nwin] = x.reshape((nseg1, nwin))
        if nseg1 > 1:
            xseg[1::2, :nwin] = x[nwin // 2 : len(x) - nwin // 2].reshape((nseg1 - 1, nwin))

        xseg_dc = np.mean(xseg, axis=1, keepdims=True)
        xseg = xseg - xseg_dc

        if nseg1 > 1:
            cwin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, nwin + 1) / nwin))
            xseg[:, :nwin] = xseg[:, :nwin] * cwin[None, :]

        fseg = np.abs(np.fft.fft(xseg, axis=1))
        fseg = fseg[:, : fseg.shape[1] // 2]

        if nseg1 > 1:
            gs.append(np.sqrt(np.mean(fseg**2, axis=0)))
        else:
            gs.append(np.abs(fseg[0, :]))

    gs = np.asarray(gs)
    F = faxis[:nfmax]
    Rax = gs[:, :nfmax]
    R = np.sqrt(np.sum(gs[:, :nfmax] ** 2, axis=0))

    if not plot:
        return R, Rax, F

    plt.figure()
    for i in range(Rax.shape[0]):
        plt.plot(F, Rax[i, :])
    plt.plot(F, R)
    plt.xlabel('frequency / Hz')

    for res in acoustic_resonances:
        freq = res['frequency'] if 'frequency' in res else res['freq']
        bw = res['bandwidth'] if 'bandwidth' in res else res['bw']
        plt.axvline(freq, color='k', linestyle='-')
        plt.axvline(freq - bw / 2, color='k', linestyle='--')
        plt.axvline(freq + bw / 2, color='k', linestyle='--')

    plt.legend(['Gx', 'Gy', 'Gz', 'Gtot'])
    return R, Rax, F
