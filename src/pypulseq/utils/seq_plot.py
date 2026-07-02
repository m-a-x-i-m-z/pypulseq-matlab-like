from __future__ import annotations

import contextlib
import itertools
import math
import typing
from copy import deepcopy

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.rotate_3d import rotate_3d
from pypulseq.Sequence import parula
from pypulseq.supported_labels_rf_use import get_supported_labels
from pypulseq.utils.cumsum import cumsum

try:
    import mplcursors

    _MPLCURSORS_AVAILABLE = True
except ImportError:
    _MPLCURSORS_AVAILABLE = False

if typing.TYPE_CHECKING:
    from pypulseq.Sequence.sequence import Sequence


class SeqPlot:
    """
    Interactive plotter for a Pulseq `Sequence` object.

    Parameters
    ----------
    seq : Sequence
        The Pulseq sequence object to plot.
    label : str, default=str()
        Plot label values for ADC events. Valid labels are accepted as a comma-separated list.
    show_blocks : bool, default=False
        Boolean flag to indicate if grid and tick labels at the block boundaries are to be plotted.
    time_range : iterable, default=(0, np.inf)
        Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
    block_range : iterable or None, default=None
        Optional block range [first, last] (1-based, inclusive), MATLAB-compatible.
    time_disp : str, default='s'
        Time display type, must be one of `s`, `ms` or `us`.
    grad_disp : str, default='kHz/m'
        Gradient display unit, must be one of `kHz/m` or `mT/m`.
    show_guides : bool, default=True
        If True, enable dynamic vertical hairline guides that follow the cursor. Requires `mplcursors`.
    stacked : bool, default=False
        If True, stack the six axes vertically, matching MATLAB Pulseq SeqPlot stacked mode.
    hide : bool, default=False
        If True, prepare the plot without showing it.

    Attributes
    ----------
    f : matplotlib.figure.Figure
        MATLAB-style single plot figure containing all six Pulseq axes.
    ax : tuple of matplotlib.axes.Axes
        Axes in MATLAB order: ADC/lbl/trig, RF mag, RF/ADC phase, Gx, Gy, Gz.
    vlines : dict (axis -> Line2D) if show_guides enabled
    """

    def __init__(
        self,
        seq: Sequence,
        label: str = str(),
        show_blocks: bool = False,
        time_range=(0, np.inf),
        block_range=None,
        time_disp: str = 's',
        grad_disp: str = 'kHz/m',
        show_guides: bool = True,
        stacked: bool = False,
        hide: bool = False,
    ):
        # Handle optional dependencies
        if _MPLCURSORS_AVAILABLE is False:
            show_guides = False

        self.seq = seq
        self._cursors = []
        self._vlines = None  # populated if show_guides enabled
        self._guide_cids = []  # mpl_connect IDs for motion events
        self._show_guides = show_guides
        self._time_disp = time_disp
        self._time_format = {'us': '{:.1f}', 'ms': '{:.4f}', 's': '{:.7f}'}.get(time_disp, '{:.7f}')

        self.f, self.ax = _seq_plot(
            seq,
            label=label,
            show_blocks=show_blocks,
            time_range=time_range,
            block_range=block_range,
            time_disp=time_disp,
            grad_disp=grad_disp,
            stacked=stacked,
            hide=hide,
        )
        self.fig1 = self.f
        self.ax1 = self.ax
        self.fig2 = None
        self.ax2 = ()

        if _MPLCURSORS_AVAILABLE:
            self._setup_cursor(self.f)

        if self._show_guides:
            self._setup_guides()

        if not hide:
            self.show()

    def show(self):
        backend = str(mpl.get_backend()).lower()
        if 'agg' in backend:
            return
        plt.show()

    def _setup_cursor(self, fig):
        if fig is None:
            return
        for ax in fig.axes:
            lines = ax.get_lines()
            for line in lines:
                with contextlib.suppress(Exception):
                    cursor = mplcursors.cursor(line, multiple=True)
                    cursor.connect('add', lambda sel: self._on_datatip(sel))
                    cursor.connect('remove', lambda sel: self._hide_datatip_guides(sel))  # new
                    self._cursors.append(cursor)

    def _setup_guides(self):
        unique_axes = []
        for ax in self.ax:
            if ax not in unique_axes:
                unique_axes.append(ax)
        self._vlines = {}
        for ax in unique_axes:
            ln = ax.axvline(0.0, color='r', linestyle='--', linewidth=1.0, visible=False, zorder=1000)
            self._vlines[ax] = ln

        def _motion(event):
            if event.inaxes in unique_axes and event.xdata is not None:
                x = event.xdata
                for ln in self._vlines.values():
                    ln.set_xdata([x])
                    ln.set_visible(True)
            else:
                for ln in self._vlines.values():
                    ln.set_visible(False)
            with contextlib.suppress(Exception):
                self.f.canvas.draw_idle()

        cid = self.f.canvas.mpl_connect('motion_notify_event', _motion)
        self._guide_cids.append((self.f.canvas, cid))

    def _on_datatip(self, sel):
        """
        Called when a datatip is created (user clicks via mplcursors).
        Populate annotation text and, if guides exist, move them to the selected x position.
        """
        artist = sel.artist
        ax = artist.axes
        x, y = sel.target
        ylabel = ax.get_ylabel().lower()

        if ylabel.startswith('adc') or (
            ylabel.startswith('rf/adc') and artist.get_linestyle() == 'none' and artist.get_marker() == '.'
        ):
            field = 'adc'
        else:
            field = ylabel[:2]

        # Convert the displayed x coordinate back to sequence time units.
        fig = artist.axes.figure
        t_factor = getattr(fig, '_seq_t_factor', 1.0)
        seq_time = x / t_factor

        # MATLAB behavior: for trapezoids, the last drawn point may belong to the next block.
        if hasattr(artist, 'get_xdata') and hasattr(artist, 'get_linestyle') and artist.get_linestyle() != 'none':
            x_data = artist.get_xdata(orig=False)
            if x_data is not None and len(x_data) > 0:
                seq_time = float(x_data[0]) / t_factor

        try:
            block_id = self.seq.find_block_by_time(seq_time)
            rb = self.seq.get_raw_block_content_IDs(block_id) if block_id is not None else None
        except Exception:
            block_id = None
            rb = None

        lines_txt = [
            f"t: {self._time_format.format(x)} {self._time_disp}",
            f'Y: {y:.6g}',
        ]

        if rb is not None and block_id is not None:
            val = getattr(rb, field, None)
            if val is not None:
                try:
                    if field[0] == 'a':
                        name = self.seq.adc_id2name_map[val]
                    elif field[0] == 'r':
                        name = self.seq.rf_id2name_map[val]
                    else:
                        name = self.seq.grad_id2name_map[val]

                    lines_txt.append(f"blk: {block_id} {field}_id: {val} '{name}'")
                except Exception:
                    lines_txt.append(f'blk: {block_id} {field}_id: {val}')
            else:
                lines_txt.append(f'blk: {block_id}')

        sel.annotation.set_text('\n'.join(lines_txt))

        # If we have dynamic guides, move them to the datatip x position and show them
        if getattr(self, '_vlines', None):
            x_coord = x
            for ln in self._vlines.values():
                ln.set_xdata([x_coord])
                ln.set_visible(True)
            for fig in {self.fig1, self.fig2}:
                if fig is not None:
                    with contextlib.suppress(Exception):
                        fig.canvas.draw_idle()

        self._update_guides()

    def _hide_datatip_guides(self, sel):  # noqa
        # Hide guides when datatip removed (connected to mplcursors 'remove' event)
        if getattr(self, '_vlines', None):
            for ln in self._vlines.values():
                ln.set_visible(False)
            for fig in {self.fig1, self.fig2}:
                if fig is not None:
                    with contextlib.suppress(Exception):
                        fig.canvas.draw_idle()

    def _update_guides(self):
        # Redraw figures after guide updates.
        for fig in (self.fig1, self.fig2):
            if fig is not None:
                with contextlib.suppress(Exception):
                    fig.canvas.draw_idle()


def _seq_plot(
    seq,
    label,
    show_blocks,
    time_range,
    block_range,
    time_disp,
    grad_disp,
    stacked,
    hide,
):
    mpl.rcParams['lines.linewidth'] = 0.75  # Set default Matplotlib linewidth
    if label is None:
        label = ''

    valid_time_units = ['s', 'ms', 'us']
    valid_grad_units = ['kHz/m', 'mT/m']
    valid_labels = get_supported_labels()
    if not all(isinstance(x, (int, float)) for x in time_range) or len(time_range) != 2:
        raise ValueError('Invalid time range')
    if time_disp not in valid_time_units:
        raise ValueError('Unsupported time unit')

    if grad_disp not in valid_grad_units:
        raise ValueError('Unsupported gradient unit. Supported gradient units are: ' + str(valid_grad_units))

    # MATLAB SeqPlot creates one figure with six axes, ordered as
    # ADC/RF/phase on the left and gradients on the right.
    fig = plt.figure()
    axes_grid = [fig.add_subplot(3, 2, i + 1) for i in range(6)]
    sp11, sp12, sp13, sp21, sp22, sp23 = [axes_grid[i] for i in [0, 2, 4, 1, 3, 5]]
    axes = (sp11, sp12, sp13, sp21, sp22, sp23)
    for sp in axes:
        sp.grid(True)

    t_factor_list = [1, 1e3, 1e6]
    t_factor = t_factor_list[valid_time_units.index(time_disp)]

    g_factor_list = [1e-3, 1e3 / seq.system.gamma]
    g_factor = g_factor_list[valid_grad_units.index(grad_disp)]

    t0 = 0
    label_defined = False
    label_idx_to_plot = []
    label_legend_to_plot = []
    label_store = {}
    for i in range(len(valid_labels)):
        label_store[valid_labels[i]] = 0
        if valid_labels[i] in label.upper():
            label_idx_to_plot.append(i)
            label_legend_to_plot.append(valid_labels[i])

    if len(label_idx_to_plot) != 0:
        p = parula.main(len(label_idx_to_plot) + 1)
        label_colors_to_plot = p(np.arange(len(label_idx_to_plot)))
        label_colors_to_plot = np.vstack([label_colors_to_plot[-1, :], label_colors_to_plot[:-1, :]])
        cycler = mpl.cycler(color=label_colors_to_plot)
        sp11.set_prop_cycle(cycler)

    # MATLAB-compatible timeRange + blockRange handling
    time_range = np.asarray(time_range, dtype=float).reshape(-1)
    if time_range.size != 2:
        raise ValueError('Invalid time range')
    if block_range is None:
        block_range = [1, np.inf]
    if len(block_range) != 2:
        raise ValueError("block_range must contain exactly two numbers")
    br_start = int(max(1, int(block_range[0])))
    br_stop = block_range[1]
    block_ids = list(seq.block_events.keys())
    n_blocks = len(block_ids)
    if n_blocks == 0:
        block_edges = np.array([0.0], dtype=float)
    else:
        block_durs = np.asarray([float(seq.block_durations[bid]) for bid in block_ids], dtype=float)
        block_edges = np.concatenate(([0.0], np.cumsum(block_durs)))
    if br_start > 1 and br_start <= n_blocks and block_edges[br_start - 1] > time_range[0]:
        time_range[0] = block_edges[br_start - 1]
    if np.isfinite(br_stop):
        br_stop_i = int(br_stop)
        if br_stop_i < n_blocks and block_edges[br_stop_i] < time_range[1]:
            time_range[1] = block_edges[br_stop_i]

    # Block timings
    block_edges_in_range = block_edges[(block_edges >= time_range[0]) * (block_edges <= time_range[1])]
    if show_blocks:
        for sp in axes:
            sp.set_xticks(t_factor * block_edges_in_range)
            sp.set_xticklabels(sp.get_xticklabels(), rotation=90)
    if time_disp == 'us':
        for sp in axes:
            with contextlib.suppress(Exception):
                sp.ticklabel_format(style='plain', axis='x', useOffset=False)

    for block_counter in seq.block_events:
        block = seq.get_block(block_counter)
        if getattr(block, 'rotation', None) is not None:
            # MATLAB v7 SeqPlot behavior: apply rotation extension before plotting gradients.
            rotated_events = rotate_3d(block.rotation.rot_quaternion, block, 'system', seq.system)
            rotated_block = deepcopy(block)
            rotated_block.gx = None
            rotated_block.gy = None
            rotated_block.gz = None
            for event in rotated_events:
                if hasattr(event, 'type') and hasattr(event, 'channel') and event.type in ['grad', 'trap']:
                    setattr(rotated_block, 'g' + event.channel, event)
            block = rotated_block
        if t0 <= time_range[1] and getattr(block, 'label', None) is not None:
            for i in range(len(block.label)):
                if block.label[i].type == 'labelinc':
                    label_store[block.label[i].label] += block.label[i].value
                else:
                    label_store[block.label[i].label] = block.label[i].value
            label_defined = True

        is_valid = time_range[0] <= t0 + seq.block_durations[block_counter] and t0 <= time_range[1]
        if is_valid:
            trig_events = getattr(block, 'trig', None)
            if trig_events is not None:
                for trig in trig_events:
                    if trig.type == 'output':
                        sp11.plot(t_factor * (t0 + trig.delay), 0, marker='D', color=(0.0, 0.5, 0.0))
                        sp11.plot(
                            t_factor * (t0 + trig.delay + np.array([0.0, trig.duration])),
                            np.array([0.0, 0.0]),
                            '-',
                            marker='.',
                            color=(0.0, 0.5, 0.0),
                        )
                    elif trig.type == 'trigger':
                        sp11.plot(t_factor * (t0 + trig.delay), 0, '>b')
                        sp11.plot(t_factor * (t0 + trig.delay), 0, '.b')

            if getattr(block, 'adc', None) is not None:  # ADC
                adc = block.adc
                # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                # is the present convention - the samples are shifted by 0.5 dwell
                t = adc.delay + (np.arange(int(adc.num_samples)) + 0.5) * adc.dwell
                sp11.plot(t_factor * (t0 + t), np.zeros(len(t)), 'rx')

                if adc.phase_modulation is None or len(adc.phase_modulation) == 0:
                    phase_modulation = 0.0
                else:
                    phase_modulation = adc.phase_modulation

                full_freq_offset = np.atleast_1d(adc.freq_offset + adc.freq_ppm * 1e-6 * seq.system.gamma * seq.system.B0)
                full_phase_offset = np.atleast_1d(
                    adc.phase_offset + adc.phase_ppm * 1e-6 * seq.system.gamma * seq.system.B0 + phase_modulation
                )

                sp13.plot(
                    t_factor * (t0 + t),
                    np.angle(np.exp(1j * full_phase_offset) * np.exp(1j * 2 * math.pi * t * full_freq_offset)),
                    'b.',
                    markersize=1.0,
                )

                if label_defined and len(label_idx_to_plot) != 0:
                    arr_label_store = list(label_store.values())
                    lbl_vals = np.take(arr_label_store, label_idx_to_plot)
                    t = t0 + adc.delay + (adc.num_samples - 1) / 2 * adc.dwell
                    _t = [t_factor * t] * len(lbl_vals)
                    # Plot each label individually to retrieve each corresponding Line2D object
                    p = itertools.chain.from_iterable(
                        [sp11.plot(__t, _lbl_vals, '.') for __t, _lbl_vals in zip(_t, lbl_vals, strict=True)]
                    )
                    if len(label_legend_to_plot) != 0:
                        sp11.legend(list(p), label_legend_to_plot, loc='upper left')
                        label_legend_to_plot = []

            if getattr(block, 'rf', None) is not None:  # RF
                rf = block.rf
                time_center, _ = calc_rf_center(rf)
                time_full = np.asarray(rf.t, dtype=float).reshape(-1)
                signal_full = np.asarray(rf.signal).reshape(-1)
                sc_real = np.interp(float(time_center), time_full, np.real(signal_full))
                sc_imag = np.interp(float(time_center), time_full, np.imag(signal_full))
                sc = sc_real + 1j * sc_imag
                time = time_full.copy()
                signal = signal_full.copy()

                if time.size > 100:
                    dtime = np.diff(time)
                    if np.max(np.abs(dtime - dtime[0])) < 1e-9:
                        st = max(1, int(round(float(seq.system.grad_raster_time) / float(dtime[0]))))
                        time = time[::st]
                        signal = signal[::st]
                        if time[-1] != rf.t[-1]:
                            time = np.concatenate([time, [rf.t[-1]]])
                            signal = np.concatenate([signal, [rf.signal[-1]]])

                if signal.shape[0] == 2 and rf.freq_offset != 0:
                    num_samples = min(int(abs(rf.freq_offset)), 256)
                    time = np.linspace(time[0], time[-1], num_samples)
                    signal = np.linspace(signal[0], signal[-1], num_samples)

                if abs(signal[0]) != 0:
                    signal = np.concatenate(([0], signal))
                    time = np.concatenate(([time[0]], time))

                if abs(signal[-1]) != 0:
                    signal = np.concatenate((signal, [0]))
                    time = np.concatenate((time, [time[-1]]))

                signal_is_real = max(np.abs(np.imag(signal))) / max(np.abs(np.real(signal))) < 1e-6

                full_freq_offset = rf.freq_offset + rf.freq_ppm * 1e-6 * seq.system.gamma * seq.system.B0
                full_phase_offset = rf.phase_offset + rf.phase_ppm * 1e-6 * seq.system.gamma * seq.system.B0

                # If off-resonant and rectangular (2 samples), interpolate the pulse
                if len(signal) == 2 and full_freq_offset != 0:
                    num_interp = min(int(abs(full_freq_offset)), 256)
                    time = np.linspace(time[0], time[-1], num_interp)
                    signal = np.linspace(signal[0], signal[-1], num_interp)

                # Compute time vector with delay applied
                time_with_delay = t_factor * (t0 + time + rf.delay)
                time_center_with_delay = t_factor * (t0 + time_center + rf.delay)

                # Choose plot behavior based on realness of signal
                if signal_is_real:
                    # Plot real part of signal
                    sp12.plot(time_with_delay, np.real(signal))

                    # Include sign(real(signal)) factor like MATLAB
                    phase_corrected = (
                        signal
                        * np.sign(np.real(signal))
                        * np.exp(1j * full_phase_offset)
                        * np.exp(1j * 2 * math.pi * time * full_freq_offset)
                    )
                    sc_corrected = (
                        sc
                        * np.exp(1j * full_phase_offset)
                        * np.exp(1j * 2 * math.pi * time_center * full_freq_offset)
                    )

                    sp13.plot(
                        time_with_delay,
                        np.angle(phase_corrected),
                        time_center_with_delay,
                        np.angle(sc_corrected),
                        'xb',
                    )
                else:
                    # Plot magnitude of complex signal
                    sp12.plot(time_with_delay, np.abs(signal))

                    # Plot angle of complex signal
                    phase_corrected = (
                        signal * np.exp(1j * full_phase_offset) * np.exp(1j * 2 * math.pi * time * full_freq_offset)
                    )
                    sc_corrected = (
                        sc
                        * np.exp(1j * full_phase_offset)
                        * np.exp(1j * 2 * math.pi * time_center * full_freq_offset)
                    )

                    sp13.plot(
                        time_with_delay,
                        np.angle(phase_corrected),
                        time_center_with_delay,
                        np.angle(sc_corrected),
                        'xb',
                    )

            grad_channels = ['gx', 'gy', 'gz']
            for x in range(len(grad_channels)):  # Gradients
                if getattr(block, grad_channels[x], None) is not None:
                    grad = getattr(block, grad_channels[x])
                    if grad.type == 'grad':
                        # We extend the shape by adding the first and the last points in an effort of making the
                        # display a bit less confusing...
                        time = grad.delay + np.array([0, *grad.tt, grad.shape_dur])
                        waveform = g_factor * np.array((grad.first, *grad.waveform, grad.last))
                    else:
                        time = np.array(
                            cumsum(
                                0,
                                grad.delay,
                                grad.rise_time,
                                grad.flat_time,
                                grad.fall_time,
                            )
                        )
                        waveform = g_factor * grad.amplitude * np.array([0, 0, 1, 1, 0])
                    [sp21, sp22, sp23][x].plot(t_factor * (t0 + time), waveform)

        t0 += seq.block_durations[block_counter]

    # Set axis labels
    sp11.set_ylabel('ADC/lbl/trig')
    sp12.set_ylabel('RF mag (Hz)')
    sp13.set_ylabel('RF/ADC phase (rad)')
    sp21.set_ylabel(f'Gx ({grad_disp})')
    sp22.set_ylabel(f'Gy ({grad_disp})')
    sp23.set_ylabel(f'Gz ({grad_disp})')
    sp13.set_xlabel(f't ({time_disp})')
    sp23.set_xlabel(f't ({time_disp})')

    # Setting display limits
    disp_range = t_factor * np.array([time_range[0], min(t0, time_range[1])])
    for sp in axes:
        sp.set_xlim(disp_range)

    sp13.set_ylim([-np.pi, np.pi])
    for sp in [sp12, sp13, sp21, sp22, sp23]:
        y0, y1 = sp.get_ylim()
        yr = y1 - y0
        if yr > 0:
            sp.set_ylim([y0 - 0.03 * yr, y1 + 0.03 * yr])

    # Store the t_factor on the figures so interactive callbacks can convert displayed x back to sequence time
    fig._seq_t_factor = t_factor
    if stacked:
        _apply_stacked_layout(fig, axes)
    else:
        fig.tight_layout()

    if not hide:
        with contextlib.suppress(Exception):
            fig.set_visible(True)

    return fig, axes


def _apply_stacked_layout(fig, axes):
    margin = 6.0
    my1 = 45.0
    mx1 = 70.0
    mx2 = 5.0
    fig.set_size_inches(10.0, 8.0, forward=True)

    width_px, height_px = fig.get_size_inches() * fig.dpi
    n_axes = len(axes)
    if width_px <= 0 or height_px <= 0 or n_axes <= 0:
        return

    ax_height = (height_px - (n_axes - 1) * margin - my1) / n_axes
    ax_width = width_px - mx1 - mx2
    if ax_height <= 0 or ax_width <= 0:
        fig.tight_layout()
        return

    for idx, ax in enumerate(axes, start=1):
        left = mx1 / width_px
        bottom = (height_px - idx * ax_height - (idx - 1) * margin) / height_px
        ax.set_position([left, bottom, ax_width / width_px, ax_height / height_px])
        ax.tick_params(axis='both', labelsize=8)
        ax.yaxis.label.set_size(9)
        ax.xaxis.label.set_size(9)
        ax.yaxis.labelpad = 2
        ax.xaxis.labelpad = 2
        ax.yaxis.set_label_coords(-0.055, 0.5)
        if idx != n_axes:
            ax.set_xlabel("")
            ax.set_xticklabels([])
