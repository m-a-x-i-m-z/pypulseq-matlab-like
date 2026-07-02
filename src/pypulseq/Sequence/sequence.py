import math
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, List, Tuple, Union
from warnings import warn

try:
    from typing import Self
except ImportError:
    from typing import TypeVar

    Self = TypeVar('Self', bound='Sequence')

import numpy as np
from scipy.interpolate import PPoly

from pypulseq import __version__, eps
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_power import calc_rf_power as ext_calc_rf_power
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.check_timing import check_timing as ext_check_timing
from pypulseq.check_timing import print_error_report
from pypulseq.decompress_shape import decompress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.opts import Opts
from pypulseq.rotate_3d import rotate_3d
from pypulseq.Sequence import block
from pypulseq.Sequence.auto_label import auto_label
from pypulseq.Sequence.calc_grad_spectrum import calculate_gradient_spectrum
from pypulseq.Sequence.calc_moments_b_tensor import calc_moments_b_tensor
from pypulseq.Sequence.calc_pns import calc_pns
from pypulseq.Sequence.ext_test_report import ext_test_report
from pypulseq.Sequence.install import detect_scanner
from pypulseq.Sequence.read_binary import read_binary
from pypulseq.Sequence.read_seq import read
from pypulseq.Sequence.write_binary import write_binary
from pypulseq.Sequence.write_seq import write as write_seq
from pypulseq.Sequence.write_seq import write_v141 as write_seq_v141
from pypulseq.restore_additional_shape_samples import restore_additional_shape_samples
from pypulseq.utils.cumsum import cumsum
from pypulseq.utils.paper_plot import paper_plot as ext_paper_plot
from pypulseq.utils.seq_plot import SeqPlot
from pypulseq.utils.tracing import format_trace, trace, trace_enabled

major, minor, revision = __version__.split('.')[:3]


class Sequence:
    """
    Generate sequences and read/write sequence files. This class defines properties and methods to define a complete MR
    sequence including RF pulses, gradients, ADC events, etc. The class provides an implementation of the open MR
    sequence format defined by the Pulseq project. See http://pulseq.github.io/.

    See also `demo_read.py`, `demo_write.py`.
    """

    version_major = int(major)
    version_minor = int(minor)
    version_revision = revision

    def __init__(self, system: Union[Opts, None] = None, use_block_cache: bool = True):
        if system is None:
            system = Opts()
        if not hasattr(system, 'flag_trid') or getattr(system, 'flag_trid') is None:
            system.flag_trid = True
        system.flag_trid = bool(system.flag_trid)

        # =========
        # EVENT LIBRARIES
        # =========
        self.adc_library = EventLibrary(numpy_data=True)
        self.delay_library = EventLibrary(numpy_data=True)
        self.extensions_library = EventLibrary(numpy_data=True)
        self.grad_library = EventLibrary(numpy_data=True)
        self.label_inc_library = EventLibrary(numpy_data=True)
        self.label_set_library = EventLibrary(numpy_data=True)
        self.rf_library = EventLibrary(numpy_data=True)
        self.shape_library = EventLibrary(numpy_data=True)
        self.trigger_library = EventLibrary(numpy_data=True)
        self.soft_delay_library = EventLibrary(numpy_data=True)
        self.rotation_library = EventLibrary(numpy_data=True)  # [NEW] Rotation extension
        self.rf_shim_library = EventLibrary(numpy_data=True)   # [NEW] RF Shim extension

        # =========
        # OTHER
        # =========
        self.system = system

        self.block_events = OrderedDict()
        self.block_trace = OrderedDict()
        self.use_block_cache = use_block_cache
        self.block_cache = {}
        self.next_free_block_ID = 1
        self.definitions = {}

        self.rf_raster_time = self.system.rf_raster_time
        self.grad_raster_time = self.system.grad_raster_time
        self.adc_raster_time = self.system.adc_raster_time
        self.block_duration_raster = self.system.block_duration_raster
        self.set_definition('AdcRasterTime', self.adc_raster_time)
        self.set_definition('BlockDurationRaster', self.block_duration_raster)
        self.set_definition('GradientRasterTime', self.grad_raster_time)
        self.set_definition('RadiofrequencyRasterTime', self.rf_raster_time)
        self.signature_type = ''
        self.signature_file = ''
        self.signature_value = ''
        self.rf_id_to_name_map = {}
        self.adc_id_to_name_map = {}
        self.grad_id_to_name_map = {}

        self.block_durations = {}
        # Extension IDs are dynamically allocated on first use.
        self.extension_numeric_idx = []
        self.extension_string_idx = []
        self.soft_delay_hints = {}
        # Soft delay hint string <-> ID mapping.
        self.soft_delay_hint_ids = {}  # hint string -> hintID (1-based)
        self.soft_delay_hints2 = []    # list of hint strings by hintID-1
        self.grad_check_data = {'validForBlockNum': 0, 'lastGradVals': np.array([0.0, 0.0, 0.0])}
        self.trid_name_to_id = {}
        self.trid_id_to_name = []
        self.trid_history = []

    def __str__(self) -> str:
        s = 'Sequence:'
        s += '\nshape_library: ' + str(self.shape_library)
        s += '\nrf_library: ' + str(self.rf_library)
        s += '\ngrad_library: ' + str(self.grad_library)
        s += '\nadc_library: ' + str(self.adc_library)
        s += '\ndelay_library: ' + str(self.delay_library)
        s += '\nextensions_library: ' + str(self.extensions_library)
        s += '\nrf_raster_time: ' + str(self.rf_raster_time)
        s += '\ngrad_raster_time: ' + str(self.grad_raster_time)
        s += '\nblock_events: ' + str(len(self.block_events))
        return s

    def copy_definitions(self, other_seq) -> None:
        self.definitions = other_seq.definitions

    def get_or_create_trid_id(self, label_name: str) -> int:
        if isinstance(label_name, np.str_):
            label_name = str(label_name)
        if not isinstance(label_name, str) or label_name == '':
            raise ValueError('TRID label_name must be a non-empty char/string.')
        if label_name in self.trid_name_to_id:
            trid_id = self.trid_name_to_id[label_name]
        else:
            trid_id = len(self.trid_id_to_name) + 1
            self.trid_name_to_id[label_name] = trid_id
            self.trid_id_to_name.append(label_name)
        self.trid_history.append(label_name)
        return trid_id

    def add_trid(self, label_name: str) -> None:
        if hasattr(self.system, 'flag_trid') and not self.system.flag_trid:
            return
        trid_id = self.get_or_create_trid_id(label_name)
        from pypulseq.make_label import make_label

        self.add_block(make_label('TRID', 'SET', float(trid_id)))

    def adc_times(self, time_range: Union[List[float], None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return time points of ADC sampling points.

        Returns
        -------
        t_adc: np.ndarray
            Contains times of all ADC sample points.
        fp_adc : np.ndarray
            Contains frequency and phase offsets of each ADC object (not samples).
        """
        # Collect ADC timing data
        t_adc = []
        fp_adc = []

        curr_dur = 0
        if time_range is None:
            blocks = self.block_events
        else:
            if len(time_range) != 2:
                raise ValueError('Time range must be list of two elements')
            if time_range[0] > time_range[1]:
                raise ValueError('End time of time_range must be after begin time')

            # Calculate end times of each block
            bd = np.array(list(self.block_durations.values()))
            t = np.cumsum(bd)
            # Search block end times for start of time range
            begin_block = np.searchsorted(t, time_range[0])
            # Search block begin times for end of time range
            end_block = np.searchsorted(t - bd, time_range[1], side='right')
            blocks = list(self.block_durations.keys())[begin_block:end_block]
            curr_dur = t[begin_block] - bd[begin_block]

        for block_counter in blocks:
            block = self.get_block(block_counter)

            if block.adc is not None:
                t_adc.append((np.arange(block.adc.num_samples) + 0.5) * block.adc.dwell + block.adc.delay + curr_dur)
                fp_adc.append([block.adc.freq_offset, block.adc.phase_offset])

            curr_dur += self.block_durations[block_counter]

        if t_adc == []:
            # If there are no ADCs, make sure the output is the right shape
            t_adc = np.zeros(0)
            fp_adc = np.zeros((0, 2))
        else:
            t_adc = np.concatenate(t_adc)
            fp_adc = np.array(fp_adc)

        return t_adc, fp_adc

    def add_block(self, *args: SimpleNamespace) -> None:
        """
        Add a new block/multiple events to the sequence.

        Parameters
        ----------
        *args : SimpleNamespace
            Event objects to be added as a block to the sequence. For delays,
            `pypulseq.make_delay()` is recommended, but numeric duration values
            are also accepted for compatibility with Matlab Pulseq behavior.

        See Also
        --------
        pypulseq.make_delay : Create delay events
        pypulseq.make_adc : Create ADC events
        pypulseq.make_trapezoid : Create trapezoid gradient events
        pypulseq.make_sinc_pulse : Create sinc RF pulse events
        pypulseq.make_soft_delay : Create soft delay events
        """
        if trace_enabled():
            self.block_trace[self.next_free_block_ID] = SimpleNamespace(block=trace())

        block.set_block(self, self.next_free_block_ID, *args)
        self.next_free_block_ID += 1

    def calculate_gradient_spectrum(
        self,
        max_frequency: float = 3000.0,
        window_width: float = 0.05,
        frequency_oversampling: float = 3.0,
        time_range: Union[List[float], None] = None,
        plot: bool = True,
        combine_mode: str = 'max',
        use_derivative: bool = False,
        acoustic_resonances: Union[List[dict], str, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the gradient spectrum of the sequence.

        Parameters
        ----------
        max_frequency : float, optional
            Maximum frequency to include in spectrograms. The default is 3000.
        window_width : float, optional
            Window width (in seconds). The default is 0.05.
        frequency_oversampling : float, optional
            Oversampling in the frequency dimension, higher values make
            smoother spectrograms. The default is 3.
        time_range : List[float], optional
            Time range over which to calculate the spectrograms as a list of
            two timepoints (in seconds) (e.g. [1, 1.5])
            The default is None.
        plot : bool, optional
            Whether to plot the spectrograms. The default is True.
        combine_mode : str, optional
            Additional selector retained by the current Python interface.
        use_derivative : bool, optional
            Whether to use the derivative of the gradient waveforms.
        acoustic_resonances : List[dict] or str, optional
            Acoustic resonances as dictionaries with 'frequency' and 'bandwidth'
            elements, or an ASC file path to derive them from.

        Returns
        -------
        R : np.ndarray
            Root-sum-of-squares spectrum over all gradient channels.
        Rax : np.ndarray
            Spectrum for the individual gradient axes.
        F : np.ndarray
            Frequency axis.

        """
        if acoustic_resonances is None:
            acoustic_resonances = []

        return calculate_gradient_spectrum(
            self,
            max_frequency=max_frequency,
            window_width=window_width,
            frequency_oversampling=frequency_oversampling,
            time_range=time_range,
            plot=plot,
            combine_mode=combine_mode,
            use_derivative=use_derivative,
            acoustic_resonances=acoustic_resonances,
        )

    def calculate_kspace(
        self,
        trajectory_delay: Union[float, List[float], np.ndarray] = 0.0,
        gradient_offset: Union[float, List[float], np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the k-space trajectory of the entire pulse sequence.

        Parameters
        ----------
        trajectory_delay : float or list or numpy.ndarray, default=0
            Compensation factor in seconds (s) to align ADC and gradients in the reconstruction.
            If trajectory_delay is a single value, this value will be used for all gradient channels.
            If trajectory_delay is a list or array, it is expected to have the same length as the number of gradient
            channels and the first element is applied to the first gradient channel, the second to the second, and so on.
        gradient_offset : float or list or numpy.ndarray, default=0
            Simulates background gradients (specified in Hz/m)
            If gradient_offset is a single value, this value will be used for all gradient channels.
            If gradient_offset is a list or array, it is expected to have the same length as the number of gradient
            channels and the first element is applied to the first gradient channel, the second to the second, and so on.
        Returns
        -------
        k_traj_adc : numpy.array
            K-space trajectory sampled at `t_adc` timepoints.
        k_traj : numpy.array
            K-space trajectory of the entire pulse sequence.
        t_excitation : numpy.array
            Excitation timepoints.
        t_refocusing : numpy.array
            Refocusing timepoints.
        t_adc : numpy.array
            Sampling timepoints.
        """
        if np.any(np.abs(trajectory_delay) > 100e-6):
            warn(f'trajectory delay of ({np.asarray(trajectory_delay) * 1e6}) us is suspiciously high')

        traj_recon_delay = trajectory_delay

        num_blocks = len(self.block_events)
        c_excitation = 0
        c_refocusing = 0
        c_adc_samples = 0
        for block_index in self.block_events:
            block = self.get_block(block_index)
            if block.rf is not None:
                if not hasattr(block.rf, 'use') or block.rf.use in ['excitation', 'undefined']:
                    c_excitation += 1
                elif block.rf.use == 'refocusing':
                    c_refocusing += 1
            if block.adc is not None:
                c_adc_samples += block.adc.num_samples

        t_excitation = np.zeros(c_excitation, dtype=float)
        t_refocusing = np.zeros(c_refocusing, dtype=float)
        ktime = np.zeros(c_adc_samples, dtype=float)
        current_dur = 0.0
        c_excitation = 0
        c_refocusing = 0
        k_counter = 0

        for block_index in self.block_events:
            block = self.get_block(block_index)
            if block.rf is not None:
                rf = block.rf
                t = rf.delay + calc_rf_center(rf)[0]
                if not hasattr(rf, 'use') or rf.use in ['excitation', 'undefined']:
                    t_excitation[c_excitation] = current_dur + t
                    c_excitation += 1
                elif rf.use == 'refocusing':
                    t_refocusing[c_refocusing] = current_dur + t
                    c_refocusing += 1
            if block.adc is not None:
                num_samples = block.adc.num_samples
                ktime[k_counter : k_counter + num_samples] = (
                    (np.arange(num_samples, dtype=float) + 0.5) * block.adc.dwell
                    + block.adc.delay
                    + current_dur
                    + traj_recon_delay
                )
                k_counter += num_samples
            current_dur += self.block_durations[block_index]

        gw = self._discrete_gradient_waveforms(gradient_offset=gradient_offset)
        i_excitation = np.round(t_excitation / self.grad_raster_time).astype(int)
        i_refocusing = np.round(t_refocusing / self.grad_raster_time).astype(int)

        i_periods = np.sort(
            np.concatenate(
                (
                    np.array([1], dtype=int),
                    i_excitation + 1,
                    i_refocusing + 1,
                    np.array([gw.shape[1] + 1], dtype=int),
                )
            )
        )
        ii_next_excitation = min(len(i_excitation), 1)
        ii_next_refocusing = min(len(i_refocusing), 1)
        k_traj = np.zeros_like(gw)
        k = np.zeros(3, dtype=float)

        for period_index in range(len(i_periods) - 1):
            i_period_start = int(i_periods[period_index])
            i_period_end = int(i_periods[period_index + 1] - 1)
            k_period = np.cumsum(
                np.hstack((k[:, None], gw[:, i_period_start - 1 : i_period_end] * self.grad_raster_time)),
                axis=1,
            )
            k_traj[:, i_period_start - 1 : i_period_end] = k_period[:, 1:]
            k = k_period[:, -1]

            if ii_next_excitation > 0 and i_excitation[ii_next_excitation - 1] == i_period_end:
                k[:] = 0.0
                k_traj[:, i_period_end - 1] = np.nan
                ii_next_excitation = min(len(i_excitation), ii_next_excitation + 1)
            if ii_next_refocusing > 0 and i_refocusing[ii_next_refocusing - 1] == i_period_end:
                k = -k
                ii_next_refocusing = min(len(i_refocusing), ii_next_refocusing + 1)

        sample_times = (np.arange(1, gw.shape[1] + 1, dtype=float)) * self.grad_raster_time
        if ktime.size == 0 or sample_times.size == 0:
            k_traj_adc = np.zeros((3, 0))
        else:
            k_traj_adc = np.zeros((3, ktime.size), dtype=float)
            for axis in range(3):
                k_traj_adc[axis, :] = np.interp(ktime, sample_times, k_traj[axis, :], left=np.nan, right=np.nan)

        return k_traj_adc, k_traj, t_excitation, t_refocusing, ktime

    def _discrete_gradient_waveforms(
        self,
        gradient_offset: Union[float, List[float], np.ndarray] = 0.0,
    ) -> np.ndarray:
        total_duration = float(sum(self.block_durations.values()))
        n_grad_samples = int(round(total_duration / self.grad_raster_time))
        gw = np.zeros((3, n_grad_samples), dtype=float)

        if n_grad_samples == 0:
            return gw

        sample_times = (np.arange(1, n_grad_samples + 1, dtype=float)) * self.grad_raster_time
        wave_data = self.waveforms(append_RF=False)
        gradient_offsets = (
            np.full(3, float(gradient_offset))
            if np.isscalar(gradient_offset)
            else np.asarray(gradient_offset, dtype=float)
        )
        if gradient_offsets.size == 1:
            gradient_offsets = np.full(3, float(gradient_offsets[0]))
        elif gradient_offsets.size != 3:
            raise ValueError('gradient_offset must be scalar or length 3')

        for axis in range(min(3, len(wave_data))):
            if wave_data[axis].size == 0:
                if abs(gradient_offsets[axis]) > eps:
                    gw[axis, :] = gradient_offsets[axis]
                continue
            gw[axis, :] = np.interp(
                sample_times,
                wave_data[axis][0, :],
                wave_data[axis][1, :],
                left=0.0,
                right=0.0,
            ) + gradient_offsets[axis]

        return gw

    def calculate_kspacePP(
        self,
        trajectory_delay: Union[float, List[float], np.ndarray] = 0,
        gradient_offset: Union[float, List[float], np.ndarray] = 0,
        blockRange: Union[List[int], Tuple[int, int], None] = None,
        externalWaveformsAndTimes: Any = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[PPoly], np.ndarray]:
        """
        Calculate the k-space trajectory using the piecewise-polynomial gradient representation.
        """
        if np.any(np.abs(trajectory_delay) > 100e-6):
            warn(f'trajectory delay of ({np.asarray(trajectory_delay) * 1e6}) us is suspiciously high')

        if blockRange is None:
            blockRange = [1, np.inf]
        block_start = max(int(blockRange[0]), 1)
        block_stop = len(self.block_durations) if not np.isfinite(blockRange[1]) else int(blockRange[1])
        if block_stop < block_start:
            raise ValueError('blockRange end must not be smaller than blockRange start')

        total_duration = float(sum(list(self.block_durations.values())[block_start - 1 : block_stop]))

        if externalWaveformsAndTimes is None:
            gw_data, tfp_excitation, tfp_refocusing, t_adc, _, pm_adc = self._waveforms_and_times_impl(
                append_RF=False, blockRange=[block_start, block_stop]
            )
        else:
            gw_data = self._get_external_field(externalWaveformsAndTimes, 'gw_data')
            tfp_excitation = self._get_external_field(externalWaveformsAndTimes, 'tfp_excitation')
            tfp_refocusing = self._get_external_field(externalWaveformsAndTimes, 'tfp_refocusing')
            t_adc = self._get_external_field(externalWaveformsAndTimes, 't_adc')
            pm_adc = self._get_external_field(externalWaveformsAndTimes, 'pm_adc', default=np.array([]))

        ng = len(gw_data)
        if np.isscalar(trajectory_delay):
            gradient_delays = np.full(ng, float(trajectory_delay))
        else:
            gradient_delays = np.asarray(trajectory_delay, dtype=float)
            if len(gradient_delays) != ng:
                raise ValueError('trajectory_delay must match the number of gradient channels')

        if np.isscalar(gradient_offset):
            gradient_offsets = np.full(ng, float(gradient_offset))
        else:
            gradient_offsets = np.asarray(gradient_offset, dtype=float)
            if len(gradient_offsets) != ng:
                raise ValueError('gradient_offset must match the number of gradient channels')

        gw_pp = [None] * ng
        teps = 1e-12
        for j in range(ng):
            wave_cnt = gw_data[j].shape[1]
            if wave_cnt == 0:
                if abs(gradient_offsets[j]) <= eps:
                    continue
                gw = np.array([[0.0, total_duration], [0.0, 0.0]], dtype=float)
            else:
                gw = np.array(gw_data[j], dtype=float, copy=True)

            if abs(gradient_delays[j]) > eps:
                gw[0, :] = gw[0, :] - gradient_delays[j]
            if not np.all(np.isfinite(gw)):
                warn('Not all elements of the generated waveform are finite.')

            starts_after_zero = gw[0, 0] > teps
            ends_before_total = gw[0, -1] < total_duration - 1e-9
            if starts_after_zero and ends_before_total:
                gw = np.hstack(
                    (
                        np.array([[-teps, gw[0, 0] - teps], [0.0, 0.0]]),
                        gw,
                        np.array([[gw[0, -1] + teps, total_duration + teps], [0.0, 0.0]]),
                    )
                )
            elif starts_after_zero:
                gw = np.hstack((np.array([[-teps, gw[0, 0] - teps], [0.0, 0.0]]), gw))
            elif ends_before_total:
                gw = np.hstack((gw, np.array([[gw[0, -1] + teps, total_duration + teps], [0.0, 0.0]])))

            if abs(gradient_offsets[j]) > eps:
                gw[1, :] = gw[1, :] + gradient_offsets[j]

            time_diff = np.diff(gw[0, :])
            unique_time_mask = np.concatenate(([True], time_diff > 1e-9))
            if not np.all(unique_time_mask):
                if np.any(time_diff <= 0.0):
                    warn('Warning: not all elements of the generated time vector are unique and sorted in accending order!')
                gw = gw[:, unique_time_mask]

            gw[1, :][gw[1, :] == -0.0] = 0.0
            gw_pp[j] = PPoly(np.vstack((np.diff(gw[1, :]) / np.diff(gw[0, :]), gw[1, :-1])), gw[0, :], extrapolate=True)

        if tfp_excitation.size > 0:
            slicepos = np.zeros((len(gw_data), tfp_excitation.shape[1]))
            for j in range(len(gw_data)):
                if gw_pp[j] is None:
                    slicepos[j, :] = np.nan
                else:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        slicepos[j, :] = tfp_excitation[1, :] / gw_pp[j](tfp_excitation[0, :])
            slicepos[~np.isfinite(slicepos)] = 0.0
            t_slicepos = tfp_excitation[0, :]
        else:
            slicepos = np.array([])
            t_slicepos = np.array([])

        gm_pp = [None] * ng
        tc = []
        for i in range(ng):
            if gw_pp[i] is None:
                continue
            gm_pp[i] = gw_pp[i].antiderivative()
            tc.append(gm_pp[i].x)
            ramp_idx = np.flatnonzero(np.abs(gm_pp[i].c[0, :]) > eps)
            for j in ramp_idx:
                tc.append(
                    np.arange(
                        np.floor(gm_pp[i].x[j] / self.grad_raster_time),
                        np.ceil(gm_pp[i].x[j + 1] / self.grad_raster_time) + 1,
                    )
                    * self.grad_raster_time
                )

        if tfp_excitation.size == 0:
            t_excitation = np.array([])
        else:
            t_excitation = tfp_excitation[0, :]
        if tfp_refocusing.size == 0:
            t_refocusing = np.array([])
        else:
            t_refocusing = tfp_refocusing[0, :]

        tacc = 1e-10
        taccinv = 1.0 / tacc
        t_ktraj = tacc * np.unique(
            np.round(
                taccinv
                * np.concatenate(
                    [
                        *(tc if tc else [np.array([])]),
                        np.array([0.0]),
                        np.asarray(t_excitation) - 2 * self.rf_raster_time,
                        np.asarray(t_excitation) - self.rf_raster_time,
                        np.asarray(t_excitation),
                        np.asarray(t_refocusing) - self.rf_raster_time,
                        np.asarray(t_refocusing),
                        np.asarray(t_adc),
                        np.array([total_duration]),
                    ]
                )
            )
        )

        rounded_t_ktraj = np.round(taccinv * t_ktraj).astype(np.int64)
        t_index_map = {value: idx for idx, value in enumerate(rounded_t_ktraj)}

        def _lookup_indices(values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return np.zeros(0, dtype=int)
            rounded_values = np.round(taccinv * np.asarray(values)).astype(np.int64)
            return np.array([t_index_map[value] for value in rounded_values], dtype=int)

        i_excitation = _lookup_indices(np.asarray(t_excitation))
        i_refocusing = _lookup_indices(np.asarray(t_refocusing))
        i_adc = _lookup_indices(np.asarray(t_adc))

        i_periods = np.unique(np.concatenate((np.array([0]), i_excitation, i_refocusing, np.array([len(t_ktraj) - 1]))))
        ii_next_excitation = 0
        ii_next_refocusing = 0

        ktraj = np.zeros((3, len(t_ktraj)))
        for i in range(min(ng, 3)):
            if gw_pp[i] is None:
                continue
            it = np.where(
                (t_ktraj >= tacc * np.round(taccinv * gm_pp[i].x[0]))
                & (t_ktraj <= tacc * np.round(taccinv * gm_pp[i].x[-1]))
            )[0]
            if len(it) == 0:
                continue
            ktraj[i, it] = gm_pp[i](t_ktraj[it])
            if t_ktraj[it[-1]] < t_ktraj[-1]:
                ktraj[i, it[-1] + 1 :] = ktraj[i, it[-1]]

        dk = -ktraj[:, 0]
        i_period_end = 0
        for i in range(len(i_periods) - 1):
            i_period = i_periods[i]
            i_period_end = i_periods[i + 1]
            if ii_next_excitation < len(i_excitation) and i_excitation[ii_next_excitation] == i_period:
                if abs(t_ktraj[i_period] - t_excitation[ii_next_excitation]) > tacc:
                    warn(
                        f'abs(t_ktraj[i_period]-t_excitation[ii_next_excitation])<{tacc} failed for '
                        f'ii_next_excitation={ii_next_excitation} error={t_ktraj[i_period] - t_excitation[ii_next_excitation]}'
                    )
                dk = -ktraj[:, i_period]
                if i_period > 0:
                    ktraj[:, i_period - 1] = np.nan
                ii_next_excitation += 1
            elif ii_next_refocusing < len(i_refocusing) and i_refocusing[ii_next_refocusing] == i_period:
                dk = -2 * ktraj[:, i_period] - dk
                ii_next_refocusing += 1

            ktraj[:, i_period:i_period_end] = ktraj[:, i_period:i_period_end] + dk[:, None]

        ktraj[:, i_period_end] = ktraj[:, i_period_end] + dk
        ktraj_adc = ktraj[:, i_adc]
        return ktraj_adc, np.asarray(t_adc), ktraj, t_ktraj, t_excitation, t_refocusing, slicepos, t_slicepos, gw_pp, pm_adc

    def calculate_pns(
        self,
        hardware: SimpleNamespace,
        time_range: Union[List[float], None] = None,
        do_plots: bool = True,
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate PNS using safe model implementation by Szczepankiewicz and Witzel
        See http://github.com/filip-szczepankiewicz/safe_pns_prediction

        Returns pns levels due to respective axes (normalized to 1 and not to 100#)

        Parameters
        ----------
        hardware : SimpleNamespace
            Hardware specifications. See safe_example_hw() from
            the safe_pns_prediction package. Alternatively a text file
            in the .asc format (Siemens) can be passed, e.g. for Prisma
            it is MP_GPA_K2309_2250V_951A_AS82.asc (we leave it as an
            exercise to the interested user to find were these files
            can be acquired from)
        do_plots : bool, optional
            Plot the results from the PNS calculations. The default is True.

        Returns
        -------
        ok : bool
            Boolean flag indicating whether peak PNS is within acceptable limits
        pns_norm : numpy.array [N]
            PNS norm over all gradient channels, normalized to 1
        pns_components : numpy.array [Nx3]
            PNS levels per gradient channel
        t_pns : np.array [N]
            Time axis for the pns_norm and pns_components arrays
        """
        return calc_pns(self, hardware, time_range=time_range, do_plots=do_plots)

    def check_timing(self, print_errors=False):
        """
        Sequence timing check.
        """
        is_ok = True
        error_report = []
        total_duration = 0.0
        grad_book = {}
        soft_delay_state = {}
        last_ev = []
        last_block_id = 0

        for i_b in self.block_events:
            b = self.get_block(i_b)
            field_order = ['rf', 'gx', 'gy', 'gz', 'adc', 'delay', 'trig', 'label', 'soft_delay', 'rf_shim', 'rotation']
            ev = []
            for field in field_order:
                if not hasattr(b, field):
                    continue
                value = getattr(b, field)
                if value is None:
                    continue
                if isinstance(value, dict):
                    ev.extend([v for v in value.values() if v is not None])
                    continue
                if isinstance(value, list):
                    ev.extend([v for v in value if v is not None])
                else:
                    ev.append(value)
            last_ev = ev
            last_block_id = i_b

            res, rep, dur = self._check_timing_block(b, ev)
            is_ok = is_ok and res

            if abs(dur - self.block_durations[i_b]) > eps:
                rep += ' inconsistency between the stored block duration and the duration of the block content'
                is_ok = False
                dur = self.block_durations[i_b]

            bd = self.block_durations[i_b] / self.block_duration_raster
            bdr = round(bd)
            if abs(bdr - bd) >= 1e-6:
                rep += ' block duration is not aligned to the block_duration_raster'
                is_ok = False

            if getattr(b, 'rf', None) is not None:
                if b.rf.delay - b.rf.dead_time < -eps:
                    rep += (
                        f' delay of {b.rf.delay * 1e6}us is smaller than the RF dead time '
                        f'{b.rf.dead_time * 1e6}us'
                    )
                    is_ok = False
                if b.rf.delay + b.rf.t[-1] + b.rf.ringdown_time - dur > eps:
                    rep += (
                        f' time between the end of the RF pulse at {(b.rf.delay + b.rf.t[-1]) * 1e6}us and the end '
                        f'of the block at {dur * 1e6}us is shorter than rf_ringdown_time'
                    )
                    is_ok = False
                rf_freq_offset = getattr(b.rf, 'freq_offset', 0.0)
                rf_freq_ppm = getattr(b.rf, 'freq_ppm', 0.0)
                rf_ppm_offset = rf_freq_ppm * 1e-6 * self.system.gamma
                max_freq_offset = getattr(self.system, 'max_freq_offset', np.inf)
                if (
                    abs(rf_freq_offset) > max_freq_offset
                    or abs(rf_ppm_offset) > max_freq_offset
                    or abs(rf_freq_offset + rf_ppm_offset) > max_freq_offset
                ):
                    rep += f' frequency offset of the RF pulse exceeds the maximum allowed value of {max_freq_offset}Hz'
                    is_ok = False

            if getattr(b, 'adc', None) is not None:
                if b.adc.delay - self.system.adc_dead_time < -eps:
                    rep += ' adc.delay<system.adc_dead_time'
                    is_ok = False
                if b.adc.delay + b.adc.num_samples * b.adc.dwell + self.system.adc_dead_time - dur > eps:
                    rep += ' adc: system.adc_dead_time (post-adc) violation'
                    is_ok = False
                if abs(b.adc.dwell / self.system.adc_raster_time - round(b.adc.dwell / self.system.adc_raster_time)) > 1e-10:
                    rep += ' adc: dwell time is not an integer multiple of system.adc_raster_time'
                    is_ok = False
                if hasattr(self.system, 'adc_samples_divisor'):
                    if abs(b.adc.num_samples / self.system.adc_samples_divisor - round(b.adc.num_samples / self.system.adc_samples_divisor)) > eps:
                        rep += ' adc: num_samples is not an integer multiple of system.adc_samples_divisor'
                        is_ok = False
                adc_freq_offset = getattr(b.adc, 'freq_offset', 0.0)
                adc_freq_ppm = getattr(b.adc, 'freq_ppm', 0.0)
                adc_ppm_offset = adc_freq_ppm * 1e-6 * self.system.gamma
                max_freq_offset = getattr(self.system, 'max_freq_offset', np.inf)
                if (
                    abs(adc_freq_offset) > max_freq_offset
                    or abs(adc_ppm_offset) > max_freq_offset
                    or abs(adc_freq_offset + adc_ppm_offset) > max_freq_offset
                ):
                    rep += f' frequency offset of the ADC object exceeds the maximum allowed value of {max_freq_offset}Hz'
                    is_ok = False

            if rep:
                error_report.append(f'   Block:{i_b} {rep}\n')

            grad_book_curr = {}
            for event in ev:
                if not (isinstance(event, SimpleNamespace) and getattr(event, 'type', None) == 'grad'):
                    continue
                if event.first != 0:
                    if event.delay != 0:
                        error_report.append(
                            f'   Block:{i_b} {event.channel} gradient starts at a non-zero value but defines a delay\n'
                        )
                        is_ok = False
                    if event.channel not in grad_book or abs(grad_book[event.channel] - event.first) > 1:
                        error_report.append(
                            f"   Block:{i_b} {event.channel} gradient's start value {event.first} differs from the previous block end value\n"
                        )
                        is_ok = False
                    else:
                        grad_book[event.channel] = 0
                if abs(event.last) > eps:
                    if abs(event.delay + event.shape_dur - dur) > eps:
                        error_report.append(
                            f'   Block:{i_b} {event.channel} gradient ends at a non-zero value but does not last until the end of the block\n'
                        )
                        is_ok = False
                    grad_book_curr[event.channel] = event.last

            if getattr(b, 'soft_delay', None) is not None:
                if b.soft_delay.factor == 0:
                    error_report.append(
                        f"   Block:{i_b} soft delay {b.soft_delay.hint}/{b.soft_delay.numID} has factor parameter of 0 which is invalid\n"
                    )
                    is_ok = False
                def_del = (self.block_durations[i_b] - b.soft_delay.offset) * b.soft_delay.factor
                soft_key = int(b.soft_delay.numID)
                if soft_key >= 0:
                    if soft_key not in soft_delay_state:
                        soft_delay_state[soft_key] = {'def': def_del, 'hint': b.soft_delay.hint, 'blk': i_b}
                    else:
                        prev = soft_delay_state[soft_key]
                        if abs(def_del - prev['def']) > 1e-7:
                            error_report.append(
                                f"   Block:{i_b} soft delay {b.soft_delay.hint}/{soft_key}: default duration derived from this block ({def_del * 1e6}us) is inconsistent with the previous default ({prev['def'] * 1e6}us) that was derived from block {prev['blk']}\n"
                            )
                            is_ok = False
                        if b.soft_delay.hint != prev['hint']:
                            error_report.append(
                                f"   Block:{i_b} soft delay {b.soft_delay.hint}/{soft_key}: soft delays with the same numeric ID are expected to share the same text hint but previous hint recorded in block {prev['blk']} is {prev['hint']}\n"
                            )
                            is_ok = False
                else:
                    error_report.append(
                        f'   Block:{i_b} contains a soft delay {b.soft_delay.hint} with an invalid numeric ID{soft_key}\n'
                    )
                    is_ok = False

            if dur != 0:
                if any(abs(v) > 0 for v in grad_book.values()):
                    error_report.append(
                        f'   Block:{i_b} some gradients in the previous non-empty block are ending at non-zero values but are not continued here\n'
                    )
                    is_ok = False
                grad_book = grad_book_curr

            total_duration += dur

        for event in last_ev:
            if isinstance(event, SimpleNamespace) and getattr(event, 'type', None) == 'grad':
                if abs(event.last) > eps:
                    error_report.append(
                        f'   Block:{last_block_id} gradients do not ramp to 0 at the end of the sequence\n'
                    )
                    is_ok = False

        prev_total_duration = self.get_definition('TotalDuration')
        if prev_total_duration != '':
            try:
                prev_total_duration_float = float(prev_total_duration)
            except (TypeError, ValueError):
                prev_total_duration_float = None
            if prev_total_duration_float is None or abs(prev_total_duration_float - total_duration) > 1e-9:
                prev_total_duration_str = (
                    f'{prev_total_duration_float:.9g}' if prev_total_duration_float is not None else str(prev_total_duration)
                )
                error_report.append(
                    f'   TotalDuration definition of {prev_total_duration_str}s was present in the sequence, but was incorrect. It is now {total_duration:.9g}s\n'
                )
                is_ok = False
        self.set_definition('TotalDuration', total_duration)

        if print_errors and error_report:
            for line in error_report:
                print(line, end='')

        return is_ok, error_report

    def _check_timing_block(self, block, events):
        if len(events) == 0:
            total_dur = calc_duration(block)
            if total_dur <= eps:
                return False, 'empty or damaged block detected', 0.0
            is_ok = self._div_check(total_dur, self.system.block_duration_raster)
            text_error = '' if is_ok else f'total duration:{total_dur * 1e6}us'
            return is_ok, text_error, total_dur

        total_dur = calc_duration(block)
        is_ok = self._div_check(total_dur, self.system.block_duration_raster)
        text_error = '' if is_ok else f'total duration:{total_dur * 1e6}us'

        for event in events:
            ok = True
            if getattr(event, 'type', None) in ('adc', 'rf', 'output'):
                raster = self.system.rf_raster_time
            else:
                raster = self.system.grad_raster_time

            if hasattr(event, 'delay'):
                if event.delay < -eps or not self._div_check(event.delay, raster):
                    ok = False
            if hasattr(event, 'duration') and not self._div_check(event.duration, raster):
                ok = False
            if hasattr(event, 'dwell'):
                if event.dwell < self.system.adc_raster_time or abs(round(event.dwell / self.system.adc_raster_time) * self.system.adc_raster_time - event.dwell) > 1e-10:
                    ok = False
            if getattr(event, 'type', None) == 'rf':
                if not self._div_check(event.shape_dur, self.system.rf_raster_time):
                    ok = False
                if len(event.t) >= 4:
                    rt = np.asarray(event.t) / self.system.rf_raster_time
                    drt = np.diff(rt)
                    if np.all(np.abs(drt[1:] - drt[0]) < 1e-9 / self.system.rf_raster_time):
                        dwell = event.t[1] - event.t[0]
                        if not self._div_check(dwell, min(self.system.adc_raster_time, self.system.rf_raster_time)):
                            ok = False
                    elif np.any(np.abs(rt - np.round(rt)) > 1e-6):
                        ok = False
            if getattr(event, 'type', None) == 'trap':
                if not (
                    self._div_check(event.rise_time, self.system.grad_raster_time)
                    and self._div_check(event.flat_time, self.system.grad_raster_time)
                    and self._div_check(event.fall_time, self.system.grad_raster_time)
                ):
                    ok = False
            if not ok:
                is_ok = False
                if text_error:
                    text_error += ' '
                text_error += '[ '
                if hasattr(event, 'type'):
                    text_error += f'type:{event.type} '
                if hasattr(event, 'delay'):
                    text_error += f'delay:{event.delay * 1e6}us '
                if hasattr(event, 'duration'):
                    text_error += f'duration:{event.duration * 1e6}us '
                if hasattr(event, 'shape_dur'):
                    text_error += f'shape_dur:{event.shape_dur * 1e6}us '
                if hasattr(event, 'dwell'):
                    text_error += f'dwell:{event.dwell * 1e9}ns '
                if getattr(event, 'type', None) == 'trap':
                    text_error += (
                        f'rise_time:{event.rise_time * 1e6}us flat_time:{event.flat_time * 1e6}us '
                        f'fall_time:{event.fall_time * 1e6}us '
                    )
                text_error += ']'

        return is_ok, text_error, total_dur

    @staticmethod
    def _div_check(a: float, b: float) -> bool:
        c = a / b
        return abs(c - round(c)) < 1e-9

    def duration(self) -> Tuple[int, int, np.ndarray]:
        """
        Returns the total duration of this sequence, and the total count of blocks and events.

        Returns
        -------
        duration : int
            Duration of this sequence in seconds (s).
        num_blocks : int
            Number of blocks in this sequence.
        event_count : np.ndarray
            Number of events in this sequence.
        """
        num_blocks = len(self.block_events)
        if num_blocks == 0:
            return 0, 0, np.zeros(0)

        event_count = np.zeros(len(next(iter(self.block_events.values()))))
        duration = 0
        for block_counter in self.block_events:
            event_count += self.block_events[block_counter] > 0
            duration += self.block_durations[block_counter]

        return duration, num_blocks, event_count

    def evaluate_labels(
        self,
        init: Union[dict, None] = None,
        evolution: str = 'none',
        time_range: Union[List[float], None] = None,
        blockRange: Union[List[int], Tuple[int, int], None] = None,
    ) -> dict:
        """
        Evaluate label values of the entire sequence.

        When no evolution is given, returns the label values at the end of the
        sequence. Returns a dictionary with keys named after the labels used
        in the sequence. Only the keys corresponding to the labels actually
        used are created.
        E.g. labels['LIN'] == 4

        When evolution is given, labels are tracked through the sequence. See
        below for options for different types of evolutions. The resulting
        dictionary will contain arrays of the label values.
        E.g. labels['LIN'] == np.array([0,1,2,3,4])

        Initial values for the labels can be given with the 'init' parameter.
        Useful if evaluating labels block-by-block.

        Parameters
        ----------
        init : dict, optional
            Dictionary containing initial label values. The default is None.
        evolution : str, optional
            Flag to specify tracking of label evolutions.
            Must be one of: 'none', 'adc', 'label', 'blocks' (default = 'none')
            'blocks': Return label values for all blocks.
            'adc':    Return label values only for blocks containing ADC events.
            'label':  Return label values only for blocks where labels are
                      manipulated.

        Returns
        -------
        labels : dict
            Dictionary containing label values.
            If evolution == 'none', the dictionary values only contains the
            final label value.
            Otherwise, the dictionary values are arrays of label evolutions.
            Only the labels that are used in the sequence are created in the
            dictionary.

        """
        labels = dict(init) if init is not None else {}
        label_evolution = []

        blocks, _ = self._resolve_blocks(blockRange=blockRange, time_range=time_range)
        for block_counter in blocks:
            block = self.get_block(block_counter)

            labels_in_block = getattr(block, 'label', None)
            if labels_in_block is not None:
                # Current block has labels
                for lab in labels_in_block:
                    if lab.type == 'labelinc':
                        # Increment label
                        if lab.label not in labels:
                            labels[lab.label] = 0

                        labels[lab.label] += lab.value
                    else:
                        # Set label
                        labels[lab.label] = lab.value

                if evolution == 'label':
                    label_evolution.append(dict(labels))

            if evolution == 'blocks' or (evolution == 'adc' and getattr(block, 'adc', None) is not None):
                label_evolution.append(dict(labels))

        # Convert evolutions into label dictionary
        if evolution != 'none' and len(label_evolution) > 1:
            for lab in labels:
                labels[lab] = np.array([e.get(lab, 0) for e in label_evolution])

        return labels

    def find_block_by_time(self, t: float) -> int:
        """
        Find the index of the block containing time `t`.

        Parameters
        ----------
        t : float
            Time (in seconds) to locate within the sequence.

        Returns
        -------
        int or None
            Index of the block that contains the given time, or None if out of range.
        """
        cumsum_durations = np.cumsum(list(self.block_durations.values()))
        block_index = np.searchsorted(cumsum_durations, t, side='right').item()

        if block_index >= len(self.block_durations):
            return None

        block_id = list(self.block_durations.keys())[block_index]
        if self.block_durations[block_id] <= 0:
            raise ValueError('Block duration cannot be negative')

        return block_id

    def flip_grad_axis(self, axis: str) -> None:
        """
        Invert all gradients along the corresponding axis/channel. The function acts on all gradient objects already
        added to the sequence object.

        Parameters
        ----------
        axis : str
            Gradients to invert or scale. Must be one of 'x', 'y' or 'z'.
        """
        self.mod_grad_axis(axis, modifier=-1)

    def get_raw_block_content_IDs(self, block_index: int) -> SimpleNamespace:
        """
        Returns PyPulseq block content IDs at `block_index` position in `self.block_events`.

        No block events are created, only the IDs of the objects are returned.

        See Also
        --------
        - `pypulseq.Sequence.sequence.Sequence.get_block()`.

        Parameters
        ----------
        block_index : int
            Index of block to be retrieved from `Sequence`.

        Returns
        -------
        SimpleNamespace
            PyPulseq block content IDs at 'block_index' position in `self.block_events`.
        """
        return block.get_raw_block_content_IDs(self, block_index)

    def get_block(self, block_index: int, add_ids: bool = False) -> SimpleNamespace:
        """
        Return a block of the sequence  specified by the index. The block is created from the sequence data with all
        events and shapes decompressed.

        See Also
        --------
        - `pypulseq.Sequence.sequence.Sequence.set_block()`.
        - `pypulseq.Sequence.sequence.Sequence.add_block()`.

        Parameters
        ----------
        block_index : int
            Index of block to be retrieved from `Sequence`.

        Returns
        -------
        SimpleNamespace
            Event identified by `block_index`.
        """
        return block.get_block(self, block_index, add_ids=add_ids)

    def get_definition(self, key: str) -> str:
        """
        Return value of the definition specified by the key. These definitions can be added manually or read from the
        header of a sequence file defined in the sequence header. An empty array is returned if the key is not defined.

        See also `pypulseq.Sequence.sequence.Sequence.set_definition()`.

        Parameters
        ----------
        key : str
            Key of definition to retrieve.

        Returns
        -------
        str
            Definition identified by `key` if found, else returns ''.
        """
        if key in self.definitions:
            return self.definitions[key]
        else:
            return ''

    def get_extension_type_ID(self, extension_string: str) -> int:
        """
        Get numeric extension ID for `extension_string`. Will automatically create a new ID if unknown.

        Parameters
        ----------
        extension_string : str
            Given string extension ID.

        Returns
        -------
        extension_id : int
            Numeric ID for given string extension ID.

        """
        if extension_string not in self.extension_string_idx:
            if len(self.extension_numeric_idx) == 0:
                extension_id = 1
            else:
                extension_id = 1 + max(self.extension_numeric_idx)

            self.extension_numeric_idx.append(extension_id)
            self.extension_string_idx.append(extension_string)
            assert len(self.extension_numeric_idx) == len(self.extension_string_idx)
        else:
            num = self.extension_string_idx.index(extension_string)
            extension_id = self.extension_numeric_idx[num]

        return extension_id

    def get_extension_type_string(self, extension_id: int) -> str:
        """
        Get string extension ID for `extension_id`.

        Parameters
        ----------
        extension_id : int
            Given numeric extension ID.

        Returns
        -------
        extension_str : str
            String ID for the given numeric extension ID.

        Raises
        ------
        ValueError
            If given numeric extension ID is unknown.
        """
        if extension_id in self.extension_numeric_idx:
            num = self.extension_numeric_idx.index(extension_id)
        else:
            raise ValueError(f'Extension for the given ID - {extension_id} - is unknown.')

        extension_str = self.extension_string_idx[num]
        return extension_str

    def get_gradients(
        self,
        trajectory_delay: Union[float, List[float], np.ndarray] = 0,
        gradient_offset: Union[float, List[float], np.ndarray] = 0,
        time_range: Union[List[float], None] = None,
        blockRange: Union[List[int], Tuple[int, int], None] = None,
    ) -> List[PPoly]:
        """
        Get all gradient waveforms of the sequence in a piecewise-polynomial
        format (scipy PPoly). Gradient values can be accessed easily at one or
        more timepoints using `gw_pp[channel](t)` (where t is a float, list of
        floats, or numpy array). Note that PPoly objects return nan for
        timepoints outside the waveform.

        Parameters
        ----------
        trajectory_delay : float or list or numpy.ndarray, default=0
            Compensation factor in seconds (s) to align ADC and gradients in the reconstruction.
            If trajectory_delay is a single value, this value will be used for all gradient channels.
            If trajectory_delay is a list or array, it is expected to have the same length as the number of gradient
            channels and the first element is applied to the first gradient channel, the second to the second, and so on.
        gradient_offset : float or list or numpy.ndarray, default=0
            Simulates background gradients (specified in Hz/m)
            If gradient_offset is a single value, this value will be used for all gradient channels.
            If gradient_offset is a list or array, it is expected to have the same length as the number of gradient
            channels and the first element is applied to the first gradient channel, the second to the second, and so on.

        Returns
        -------
        gw_pp : List[PPoly]
            List of gradient waveforms for each of the gradient channels,
            expressed as scipy PPoly objects.
        """
        if np.any(np.abs(trajectory_delay) > 100e-6):
            warn(f'Trajectory delay of {trajectory_delay * 1e6} us is suspiciously high')

        if blockRange is not None and time_range is not None:
            raise ValueError('Specify either blockRange or time_range, not both')

        gw_data = self.waveforms(time_range=time_range, blockRange=blockRange)
        ng = len(gw_data)
        blocks, _ = self._resolve_blocks(blockRange=blockRange, time_range=time_range)
        total_duration = float(sum(self.block_durations[block_counter] for block_counter in blocks))

        # Gradient delay handling
        if isinstance(trajectory_delay, (int, float)):
            gradient_delays = [trajectory_delay] * ng
        else:
            assert len(trajectory_delay) == ng  # Need to have same number of gradient channels
            gradient_delays = trajectory_delay

        # Gradient offset handling
        if isinstance(gradient_offset, (int, float)):
            gradient_offset = [gradient_offset] * ng
        else:
            assert len(gradient_offset) == ng  # Need to have same number of gradient channels

        gw_pp = [None] * ng
        teps = 1e-12
        for j in range(ng):
            wave_cnt = gw_data[j].shape[1]
            if wave_cnt == 0:
                if np.abs(gradient_offset[j]) <= eps:
                    continue
                gw = np.array([[0.0, total_duration], [0.0, 0.0]], dtype=float)
            else:
                gw = np.array(gw_data[j], dtype=float, copy=True)

            if abs(gradient_delays[j]) > eps:
                gw[0, :] = gw[0, :] - gradient_delays[j]
            if not np.all(np.isfinite(gw)):
                warn('Not all elements of the generated waveform are finite.')

            starts_after_zero = gw[0, 0] > teps
            ends_before_total = gw[0, -1] < total_duration - 1e-9
            if starts_after_zero and ends_before_total:
                gw = np.hstack(
                    (
                        np.array([[-teps, gw[0, 0] - teps], [0.0, 0.0]]),
                        gw,
                        np.array([[gw[0, -1] + teps, total_duration + teps], [0.0, 0.0]]),
                    )
                )
            elif starts_after_zero:
                gw = np.hstack((np.array([[-teps, gw[0, 0] - teps], [0.0, 0.0]]), gw))
            elif ends_before_total:
                gw = np.hstack((gw, np.array([[gw[0, -1] + teps, total_duration + teps], [0.0, 0.0]])))

            if np.abs(gradient_offset[j]) > eps:
                gw[1, :] = gw[1, :] + gradient_offset[j]

            time_diff = np.diff(gw[0, :])
            unique_time_mask = np.concatenate(([True], time_diff > 1e-9))
            if not np.all(unique_time_mask):
                if np.any(time_diff <= 0.0):
                    warn('Warning: not all elements of the generated time vector are unique and sorted in accending order!')
                gw = gw[:, unique_time_mask]

            gw[1, :][gw[1, :] == -0.0] = 0.0
            gw_pp[j] = PPoly(np.vstack((np.diff(gw[1, :]) / np.diff(gw[0, :]), gw[1, :-1])), gw[0, :], extrapolate=True)

        return gw_pp

    def install(self, target: Union[str, None] = None, clear_cache: bool = False, **kwargs: Any) -> bool:
        """Install a sequence to a target scanner.

        The sequence will be installed to a scanner specified by `target`. If `target` is not specified,
        all known scanners will be attempted to be detected. Targets supported by PyPulseq:
            siemens: All siemens targets below
            siemens_nx: Siemens Numaris X
            siemens_n4: Siemens Numaris 4
            siemens_n4_2: Siemens Numeris 4 on IP 192.168.2.2
            siemens_n4_3: Siemens Numeris 4 on IP 192.168.2.3
        Once a scanner is successfully detected, this result will be cached so
        `install` will operate more quickly on successive calls. The cache can
        be cleared by specifying `clear_cache=True`.
        Parameters
        ----------
        target : str, optional
            Target scanner. The default is None.
        clear_cache : bool, optional
            Clear the scanner detection cache. The default is False.
        **kwargs : Any
            Keyword arguments to be passed to the target's `install` function.
        Raises
        ------
        RuntimeError
            If the scanner could not be detected, or if the installation failed.
        """
        name, definition = detect_scanner(target, clear_cache=clear_cache)

        if definition is None:
            raise RuntimeError('Scanner could not be detected')

        if not definition.install(self, **kwargs):
            raise RuntimeError('Sequence install failed')

        print(f'Sequence installed correctly on target `{name}`')
        return True

    def mod_grad_axis(self, axis: str, modifier: float) -> None:
        """
        Invert or scale all gradients along the corresponding axis/channel.

        The function acts on all gradient objects already added to the sequence object.
        This modifies the amplitude of gradients while preserving timing parameters
        (rise_time, flat_time, fall_time, delay). The gradient area scales proportionally
        with the amplitude.

        Parameters
        ----------
        axis : str
            Gradients to invert or scale. Must be one of 'x', 'y' or 'z'.
        modifier : float
            Scaling value.

        Raises
        ------
        ValueError
            If invalid `axis` is passed. Must be one of 'x', 'y','z'.
        RuntimeError
            If same gradient event is used on multiple axes.

        Examples
        --------
        Disable phase encoding gradients (y-axis):

        >>> seq.mod_grad_axis('y', 0.0)

        Invert readout gradients (x-axis):

        >>> seq.mod_grad_axis('x', -1.0)

        Reduce slice selection gradients by half (z-axis):

        >>> seq.mod_grad_axis('z', 0.5)

        Double all gradients on x-axis:

        >>> seq.mod_grad_axis('x', 2.0)

        Notes
        -----
        - Only amplitude-related parameters are modified (amplitude, area, first, last)
        - Timing parameters remain unchanged (rise_time, flat_time, fall_time, delay)
        - For arbitrary gradients, the entire waveform is scaled
        - For trapezoid gradients, only the amplitude is scaled
        - Setting modifier to 0.0 effectively disables gradients on that axis
        """
        if axis not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid axis. Must be one of 'x', 'y','z'. Passed: {axis}")

        channel_num = ['x', 'y', 'z'].index(axis)
        other_channels = [0, 1, 2]
        other_channels.remove(channel_num)

        # Go through all event table entries and list gradient objects in the library
        if len(self.block_events) == 0:
            # Empty sequence - nothing to modify
            return

        all_grad_events = np.array(list(self.block_events.values()))
        all_grad_events = all_grad_events[:, 2:5]

        selected_events = np.unique(all_grad_events[:, channel_num])
        selected_events = selected_events[selected_events != 0]
        other_events = np.unique(all_grad_events[:, other_channels])
        if len(np.intersect1d(selected_events, other_events)) > 0:
            raise RuntimeError('mod_grad_axis does not yet support the same gradient event used on multiple axes.')

        for i in range(len(selected_events)):
            event_id = selected_events[i]
            old_data = self.grad_library.data[event_id]
            grad_type = self.grad_library.type[event_id]

            # Convert tuple to list for modification
            grad_data = list(old_data)
            grad_data[0] *= modifier

            if grad_type == 'g' and len(grad_data) == 6:
                # Need to update first and last fields for arbitrary gradients
                # Data structure: (amplitude, first, last, shape_ID1, shape_ID2, delay) # changed in v1.5.x
                grad_data[1] *= modifier  # first # changed in v1.5.x
                grad_data[2] *= modifier  # last # changed in v1.5.x

            # Use EventLibrary.update() to properly maintain keymap integrity
            new_data = tuple(grad_data)
            self.grad_library.update(event_id, old_data, new_data, grad_type)

        # Clear block cache to ensure get_block() uses the modified gradient data
        if self.use_block_cache:
            self.block_cache.clear()

    def paper_plot(
        self,
        block_range: Tuple[float] = (1, np.inf),
        line_width: float = 1.2,
        axes_color: Tuple[float] = (0.5, 0.5, 0.5),
        rf_color: str = 'black',
        gx_color: str = 'blue',
        gy_color: str = 'red',
        gz_color: Tuple[float] = (0, 0.5, 0.3),
        rf_plot: str = 'abs',
    ):
        """
        Plot sequence using paper-style formatting (minimalist, high-contrast layout).

        Parameters
        ----------
        block_range : iterable, default=(1, np.inf)
            1-based inclusive block range to plot.
        line_width : float, default=1.2
            Line width used in plots.
        axes_color : color, default=(0.5, 0.5, 0.5)
            Color of horizontal zero axes (e.g., gray).
        rf_color : color, default='black'
            Color for RF and ADC events.
        gx_color : color, default='blue'
            Color for gradient X waveform.
        gy_color : color, default='red'
            Color for gradient Y waveform.
        gz_color : color, default=(0, 0.5, 0.3)
            Color for gradient Z waveform.
        rf_plot : {'abs', 'real', 'imag'}, default='abs'
            Determines how to plot RF waveforms (magnitude, real or imaginary part).

        """
        return ext_paper_plot(self, block_range, line_width, axes_color, rf_color, gx_color, gy_color, gz_color, rf_plot)

    def plot(
        self,
        label: str = str(),
        show_blocks: bool = False,
        time_range=(0, np.inf),
        block_range=None,
        time_disp: str = 's',
        grad_disp: str = 'kHz/m',
        show_guides: bool = True,
        stacked: bool = False,
        hide: bool = False,
    ) -> SeqPlot:
        """
        Plot `Sequence`.

        Parameters
        ----------
        label : str, default=str()
            Plot label values for ADC events: in this example for LIN and REP labels; other valid labels are accepted as
            a comma-separated list.
        show_blocks : bool, default=False
            Boolean flag to indicate if grid and tick labels at the block boundaries are to be plotted.
        time_range : iterable, default=(0, np.inf)
            Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
        block_range : iterable or None, default=None
            Optional block range [first, last] (1-based, inclusive), aligned with MATLAB Pulseq plot behavior.
        time_disp : str, default='s'
            Time display type, must be one of 's', 'ms' or 'us'.
        grad_disp : str, default='kHz/m'
            Gradient display unit, must be one of 'kHz/m' or 'mT/m'.
        show_guides : bool, default=True
            If True, enable dynamic vertical hairline guides that follow the cursor. Requires `mplcursors`.
        stacked : bool, default=False
            If True, stack all six Pulseq axes vertically, matching MATLAB Pulseq SeqPlot stacked mode.
        hide : bool, default=False
            If True, prepare the plot without showing it.

        Returns
        -------
        SeqPlot
            SeqPlot handle.
        """
        return SeqPlot(
            self,
            label=label,
            show_blocks=show_blocks,
            time_range=time_range,
            block_range=block_range,
            time_disp=time_disp,
            grad_disp=grad_disp,
            show_guides=show_guides,
            stacked=stacked,
            hide=hide,
        )

    def read(self, file_path: str, detect_rf_use: bool = False, remove_duplicates: bool = True) -> None:
        """
        Read `.seq` file from `file_path`.

        Parameters
        ----------
        detect_rf_use
        file_path : str
            Path to `.seq` file to be read.
        remove_duplicates : bool, default=True
            Remove duplicate events from the sequence after reading.
        """
        if self.use_block_cache:
            self.block_cache.clear()

        read(self, path=file_path, detect_rf_use=detect_rf_use, remove_duplicates=remove_duplicates)

        # Initialize next free block ID
        self.next_free_block_ID = (max(self.block_events) + 1) if self.block_events else 1

    def register_adc_event(self, event: EventLibrary) -> int:
        return block.register_adc_event(self, event)

    def register_grad_event(self, event: SimpleNamespace) -> Union[int, Tuple[int, int]]:
        return block.register_grad_event(self, event)

    def register_label_event(self, event: SimpleNamespace) -> int:
        return block.register_label_event(self, event)


    def register_rotation_event(self, event: SimpleNamespace) -> int:
        """
        Register a rotation event in the rotation library.

        Parameters
        ----------
        event : SimpleNamespace
            Rotation event object.

        Returns
        -------
        int
            ID of the registered event.
        """
        if hasattr(event, 'rot_quaternion'):
            q = np.asarray(event.rot_quaternion, dtype=float).flatten()
        else:
            raise ValueError('invalid rotation quaternion detected during register_rotation_event()')

        if q.size != 4:
            raise ValueError('invalid rotation quaternion detected during register_rotation_event()')

        if abs(1.0 - float(np.sum(q * q))) >= 1e-6:
            raise ValueError('invalid rotation quaternion detected during register_rotation_event()')

        # Pulseq 1.5.1 writes: id RotQuat0 RotQuatX RotQuatY RotQuatZ
        return self.rotation_library.find_or_insert(q)[0]

    def register_rf_shim_event(self, event: SimpleNamespace) -> int:
        """
        Register an RF shim event in the RF shim library.

        Parameters
        ----------
        event : SimpleNamespace
            RF shim event object.

        Returns
        -------
        int
            ID of the registered event.
        """
        shim_vector = getattr(event, 'shim_vector', None)
        if shim_vector is None:
            raise ValueError('RF shim event must contain shim_vector.')

        # Interleave magnitude and phase into a flat numeric vector for the file extension payload.
        shim_vector = np.asarray(shim_vector, dtype=complex).reshape(-1, order='F')
        data = np.empty(2 * shim_vector.size, dtype=float)
        data[0::2] = np.abs(shim_vector)
        data[1::2] = np.angle(shim_vector)

        return self.rf_shim_library.find_or_insert(data)[0]

    def register_rf_event(self, event: SimpleNamespace) -> Tuple[int, List[int]]:
        return block.register_rf_event(self, event)

    def register_soft_delay_event(self, event: SimpleNamespace) -> int:
        return block.register_soft_delay_event(self, event)

    def remove_duplicates(self, in_place: bool = False) -> 'Sequence':
        """
        Removes duplicate events from the shape and event libraries contained
        in this sequence.

        Parameters
        ----------
        in_place : bool, optional
            If true, removes the duplicates from the current sequence.
            Otherwise, a copy is created. The default is False.

        Returns
        -------
        seq_copy : Sequence
            If `in_place`, returns self. Otherwise returns a copy of the
            sequence.
        """
        if in_place:
            seq_copy = self
        else:
            # Avoid copying block_cache for performance
            tmp = self.block_cache
            self.block_cache = {}
            seq_copy = deepcopy(self)
            self.block_cache = tmp

        # Find duplicate in shape library
        seq_copy.shape_library, mapping = seq_copy.shape_library.remove_duplicates(9)

        # Remap shape IDs of arbitrary gradient events
        for grad_id in seq_copy.grad_library.data:
            if seq_copy.grad_library.type[grad_id] == 'g':
                data = seq_copy.grad_library.data[grad_id]
                new_data = (
                    *data[0:3],
                    mapping.get(int(data[3]), int(data[3])),
                    mapping.get(int(data[4]), int(data[4])),
                    data[5],
                )
                if not np.array_equal(data, new_data):
                    seq_copy.grad_library.update(grad_id, None, np.array(new_data))

        # Remap shape IDs of RF events
        for rf_id in seq_copy.rf_library.data:
            data = seq_copy.rf_library.data[rf_id]
            new_data = (
                data[0],
                mapping.get(int(data[1]), int(data[1])),
                mapping.get(int(data[2]), int(data[2])),
                mapping.get(int(data[3]), int(data[3])),
                *data[4:],
            )
            if not np.array_equal(data, new_data):
                seq_copy.rf_library.update(rf_id, None, np.array(new_data, dtype=object))

        # Filter duplicates in gradient library
        seq_copy.grad_library, mapping = seq_copy.grad_library.remove_duplicates((6, -6, -6, -6, -6, -6))

        # Remap gradient event IDs
        for block_id in seq_copy.block_events:
            seq_copy.block_events[block_id][2] = mapping[seq_copy.block_events[block_id][2]]
            seq_copy.block_events[block_id][3] = mapping[seq_copy.block_events[block_id][3]]
            seq_copy.block_events[block_id][4] = mapping[seq_copy.block_events[block_id][4]]

        # Filter duplicates in RF library
        seq_copy.rf_library, mapping = seq_copy.rf_library.remove_duplicates((6, 0, 0, 0, 6, 6, 6, 6, 6, 6))

        # Remap RF event IDs
        for block_id in seq_copy.block_events:
            seq_copy.block_events[block_id][1] = mapping[seq_copy.block_events[block_id][1]]

        # Filter duplicates in ADC library
        seq_copy.adc_library, mapping = seq_copy.adc_library.remove_duplicates((0, -9, -6, 6, 6, 6, 6, 6, 6))

        # Remap ADC event IDs
        for block_id in seq_copy.block_events:
            seq_copy.block_events[block_id][5] = mapping[seq_copy.block_events[block_id][5]]

        return seq_copy

    def rf_from_lib_data(self, lib_data: list, use: str = '') -> SimpleNamespace:
        """
        Construct RF object from `lib_data`.

        Parameters
        ----------
        lib_data : list
            RF envelope.
        use : str, default=''
            RF event use.

        Returns
        -------
        rf : SimpleNamespace
            RF object constructed from `lib_data`.
        """
        rf = SimpleNamespace()
        rf.type = 'rf'

        amplitude, mag_shape, phase_shape = lib_data[0], lib_data[1], lib_data[2]
        shape_data = self.shape_library.data[mag_shape]
        compressed = SimpleNamespace()
        compressed.num_samples = shape_data[0]
        compressed.data = shape_data[1:]
        mag = decompress_shape(compressed)
        shape_data = self.shape_library.data[phase_shape]
        compressed.num_samples = shape_data[0]
        compressed.data = shape_data[1:]
        phase = decompress_shape(compressed)
        rf.signal = amplitude * mag * np.exp(1j * 2 * math.pi * phase)
        time_shape = lib_data[3]
        if time_shape > 0:
            shape_data = self.shape_library.data[time_shape]
            compressed.num_samples = shape_data[0]
            compressed.data = shape_data[1:]
            rf.t = decompress_shape(compressed) * self.rf_raster_time
            rf.shape_dur = math.ceil((rf.t[-1] - eps) / self.rf_raster_time) * self.rf_raster_time
        else:  # Generate default time raster on the fly
            rf.t = (np.arange(1, len(rf.signal) + 1) - 0.5) * self.rf_raster_time
            rf.shape_dur = len(rf.signal) * self.rf_raster_time

        rf.center = lib_data[4]  # v150: new field
        rf.delay = lib_data[5]  # v150: changed from lib_data[4] to lib_data[5]
        rf.freq_ppm = lib_data[6]  # v150: new field
        rf.phase_ppm = lib_data[7]  # v150: new field
        rf.freq_offset = lib_data[8]  # v150: changed from lib_data[5] to lib_data[8]
        rf.phase_offset = lib_data[9]  # v150: changed from lib_data[6] to lib_data[9]

        rf.dead_time = self.system.rf_dead_time
        rf.ringdown_time = self.system.rf_ringdown_time

        use_cases = {
            'e': 'excitation',
            'r': 'refocusing',
            'i': 'inversion',
            's': 'saturation',
            'p': 'preparation',
            'o': 'other',
        }
        rf.use = use_cases.get(use, 'undefined')

        return rf

    def rf_times(
        self, time_range: Union[List[float], None] = None
    ) -> Tuple[List[float], np.ndarray, List[float], np.ndarray, np.ndarray]:
        """
        Return time points of excitations and refocusings.

        Returns
        -------
        t_excitation : List[float]
            Contains time moments of the excitation RF pulses
        fp_excitation : np.ndarray
            Contains frequency and phase offsets of the excitation RF pulses
        t_refocusing : List[float]
            Contains time moments of the refocusing RF pulses
        fp_refocusing : np.ndarray
            Contains frequency and phase offsets of the excitation RF pulses
        """
        # Collect RF timing data
        t_excitation = []
        fp_excitation = []
        t_refocusing = []
        fp_refocusing = []

        curr_dur = 0
        if time_range is None:
            blocks = self.block_events
        else:
            if len(time_range) != 2:
                raise ValueError('Time range must be list of two elements')
            if time_range[0] > time_range[1]:
                raise ValueError('End time of time_range must be after begin time')

            # Calculate end times of each block
            bd = np.array(list(self.block_durations.values()))
            t = np.cumsum(bd)
            # Search block end times for start of time range
            begin_block = np.searchsorted(t, time_range[0])
            # Search block begin times for end of time range
            end_block = np.searchsorted(t - bd, time_range[1], side='right')
            blocks = list(self.block_durations.keys())[begin_block:end_block]
            curr_dur = t[begin_block] - bd[begin_block]

        for block_counter in blocks:
            block = self.get_block(block_counter)

            if block.rf is not None:
                rf = block.rf

                tc = calc_rf_center(rf)[0]
                t = rf.delay + tc

                full_freq_offset = rf.freq_offset + rf.freq_ppm * 1e-6 * self.system.gamma * self.system.B0
                full_phase_offset = rf.phase_offset + rf.phase_ppm * 1e-6 * self.system.gamma * self.system.B0
                full_phase_offset = full_phase_offset + 2 * math.pi * full_freq_offset * tc

                if not hasattr(rf, 'use') or block.rf.use in [
                    'excitation',
                    'undefined',
                ]:
                    t_excitation.append(curr_dur + t)
                    fp_excitation.append([full_freq_offset, full_phase_offset])
                elif block.rf.use == 'refocusing':
                    t_refocusing.append(curr_dur + t)
                    fp_refocusing.append([full_freq_offset, full_phase_offset])

            curr_dur += self.block_durations[block_counter]

        if len(t_excitation) != 0:
            fp_excitation = np.array(fp_excitation).T
        else:
            fp_excitation = np.empty((2, 0))

        if len(t_refocusing) != 0:
            fp_refocusing = np.array(fp_refocusing).T
        else:
            fp_refocusing = np.empty((2, 0))

        return np.asarray(t_excitation, dtype=float), fp_excitation, np.asarray(t_refocusing, dtype=float), fp_refocusing

    def set_block(self, block_index: int, *args: SimpleNamespace) -> None:
        """
        Replace block at index with new block provided as block structure, add sequence block, or create a new block
        from events and store at position specified by index. The block or events are provided in uncompressed form and
        will be stored in the compressed, non-redundant internal libraries.

        See Also
        --------
        - `pypulseq.Sequence.sequence.Sequence.get_block()`
        - `pypulseq.Sequence.sequence.Sequence.add_block()`

        Parameters
        ----------
        block_index : int
            Index at which block is replaced.
        args : SimpleNamespace
            Block or events to be replaced/added or created at `block_index`.
        """
        if trace_enabled():
            self.block_trace[block_index] = SimpleNamespace(block=trace())

        block.set_block(self, block_index, *args)

        if block_index >= self.next_free_block_ID:
            self.next_free_block_ID = block_index + 1

    def set_definition(self, key: str, value: Union[float, int, list, np.ndarray, str, tuple]) -> None:
        """
        Modify a custom definition of the sequence. Set the user definition 'key' to value 'value'. If the definition
        does not exist it will be created.

        See also `pypulseq.Sequence.sequence.Sequence.get_definition()`.

        Parameters
        ----------
        key : str
            Definition key.
        value : int, list, np.ndarray, str or tuple
            Definition value.
        """
        if key == 'FOV' and np.max(value) > 1:
            text = 'Definition FOV uses values exceeding 1 m. '
            text += 'New Pulseq interpreters expect values in units of meters.'
            warn(text)

        self.definitions[key] = value

    def set_extension_string_ID(self, extension_str: str, extension_id: int) -> None:
        """
        Set numeric ID for the given string extension ID.

        Parameters
        ----------
        extension_str : str
            Given string extension ID.
        extension_id : int
            Given numeric extension ID.

        Raises
        ------
        ValueError
            If given numeric or string extension ID is not unique.
        """
        if extension_str in self.extension_string_idx or extension_id in self.extension_numeric_idx:
            raise ValueError('Numeric or string ID is not unique')

        self.extension_numeric_idx.append(extension_id)
        self.extension_string_idx.append(extension_str)
        assert len(self.extension_numeric_idx) == len(self.extension_string_idx)

    def apply_soft_delay(self, **kwargs):
        """
        Apply soft delay values to modify block durations in the sequence.

        This method updates the durations of blocks containing soft delay events
        based on the provided values. The new block duration is calculated using
        the formula: duration = (input_value / factor) + offset.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments where keys are soft delay hint strings and values
            are the desired delay values in seconds. Not all soft delays in the
            sequence need to be specified.

        Raises
        ------
        ValueError
            If a specified soft delay hint does not exist in the sequence.
        ValueError
            If the calculated block duration would be negative.
        ValueError
            If soft delay hint and numeric ID mapping is inconsistent.

        Examples
        --------
        Apply single soft delay:

        >>> seq.apply_soft_delay(TE=40e-3)  # Set TE to 40ms

        Apply multiple soft delays:

        >>> seq.apply_soft_delay(TE=50e-3, TR=2.0)  # Set TE to 50ms, TR to 2s

        See Also
        --------
        pypulseq.make_soft_delay : Create soft delay events

        Notes
        -----
        - Only soft delays present in the sequence can be modified
        - Block durations are automatically rounded to the block duration raster
        - A warning is issued if substantial rounding occurs
        """

        # Go through all the blocks and update durations, at the same time checking the consistency of the soft delays
        sd_str2numID = {}
        sd_numID2hint = {}
        sd_warns = {}
        for block_counters in range(1, len(self.block_durations) + 1):
            block = self.get_block(block_counters)
            if hasattr(block, 'soft_delay') and block.soft_delay is not None:
                # Check the numeric ID consistency
                if block.soft_delay.hint not in sd_str2numID:
                    sd_str2numID[block.soft_delay.hint] = block.soft_delay.numID
                else:
                    if sd_str2numID[block.soft_delay.hint] != block.soft_delay.numID:
                        raise ValueError(
                            f"Soft delay in block {block_counters} with numeric ID {block.soft_delay.numID} and string hint '{block.soft_delay.hint}' is inconsistent with the previous occurrences of the same string hint"
                        )

                if block.soft_delay.numID not in sd_numID2hint:
                    sd_numID2hint[block.soft_delay.numID] = block.soft_delay.hint
                else:
                    if sd_numID2hint[block.soft_delay.numID] != block.soft_delay.hint:
                        raise ValueError(
                            f"Soft delay in block {block_counters} with numeric ID {block.soft_delay.numID} and string hint '{block.soft_delay.hint}' is inconsistent with the previous occurrences of the same numeric ID"
                        )

                if block.soft_delay.hint in kwargs:
                    # Calculate the new block duration
                    new_dur_ru = (
                        kwargs[block.soft_delay.hint] / block.soft_delay.factor + block.soft_delay.offset
                    ) / self.system.block_duration_raster
                    new_dur = round(new_dur_ru) * self.system.block_duration_raster
                    # Check if rounding error is significant (threshold: 0.5 microseconds)
                    rounding_threshold = 0.5e-6
                    rounding_error = abs(new_dur - new_dur_ru * self.system.block_duration_raster)
                    if rounding_error > rounding_threshold and block.soft_delay.numID not in sd_warns:
                        warn(
                            f"Soft delay '{block.soft_delay.hint}' in block {block_counters}: "
                            f'Duration rounded by {rounding_error * 1e6:.1f} μs to align with raster time '
                            f'({self.system.block_duration_raster * 1e6:.1f} μs). '
                            f'This warning is shown only once per soft delay ID.'
                        )
                        sd_warns[block.soft_delay.numID] = True

                    if new_dur < 0:
                        raise ValueError(
                            f"Soft delay '{block.soft_delay.hint}' in block {block_counters}: "
                            f'Calculated duration is negative ({new_dur * 1e6:.1f} μs). '
                            f'Check the offset ({block.soft_delay.offset * 1e6:.1f} μs) and factor '
                            f'({block.soft_delay.factor}) parameters.'
                        )

                    self.block_durations[block_counters] = new_dur

        # Now check if there are some input soft delays which haven't been found in the sequence
        all_input_hints = kwargs.keys()
        for hint in all_input_hints:
            if hint not in sd_str2numID:
                available_hints = list(sd_str2numID.keys())
                raise ValueError(
                    f"Soft delay '{hint}' not found in sequence. "
                    f'Available soft delays: {available_hints if available_hints else "none"}'
                )

    def calc_rf_power(
        self,
        blockRange: Union[List[int], Tuple[int, int], None] = None,
        windowDuration: float = np.nan,
    ) -> Tuple[float, float, float, float]:
        if blockRange is None:
            blockRange = [1, np.inf]
        if not np.isfinite(blockRange[1]):
            blockRange = [blockRange[0], len(self.block_events)]

        dur = 0.0
        total_energy = 0.0
        peak_pwr = 0.0
        rf_ms = 0.0

        window_on = np.isfinite(windowDuration)
        if window_on:
            bookkeeping = np.zeros((2, int(blockRange[1] - blockRange[0] + 1)))
            current_window_dur = 0.0
            current_window_start = int(blockRange[0])
            total_energy_max = 0.0
            rf_ms_max = 0.0

        for i_block in range(int(blockRange[0]), int(blockRange[1]) + 1):
            block = self.get_block(i_block)
            dur += self.block_durations[i_block]

            if getattr(block, 'rf', None) is not None:
                energy, peak, rms = ext_calc_rf_power(block.rf)
                total_energy += energy
                rf_ms += rms**2 * block.rf.shape_dur
                peak_pwr = max(peak_pwr, peak)

                if window_on:
                    bookkeeping[:, i_block - int(blockRange[0])] = [energy, rms**2 * block.rf.shape_dur]
                    total_energy_max = max(total_energy_max, total_energy)
                    rf_ms_max = max(rf_ms_max, rf_ms)

            if window_on:
                current_window_dur += self.block_durations[i_block]
                while current_window_dur > windowDuration:
                    total_energy -= bookkeeping[0, current_window_start - int(blockRange[0])]
                    rf_ms -= bookkeeping[1, current_window_start - int(blockRange[0])]
                    current_window_dur -= self.block_durations[current_window_start]
                    current_window_start += 1

        if window_on:
            total_energy = total_energy_max
            mean_pwr = total_energy / windowDuration
            rf_rms = np.sqrt(rf_ms_max / windowDuration)
        else:
            mean_pwr = total_energy / dur if dur > 0 else 0.0
            rf_rms = np.sqrt(rf_ms / dur) if dur > 0 else 0.0

        return mean_pwr, peak_pwr, rf_rms, total_energy

    def get_default_soft_delay_values(self):
        error_report = []
        soft_delay_state = []

        for i_block in self.block_events:
            block = self.get_block(i_block)
            if hasattr(block, 'soft_delay') and block.soft_delay is not None:
                if block.soft_delay.factor == 0:
                    error_report.append(
                        f'   Block:{i_block} soft delay {block.soft_delay.hint}/{block.soft_delay.numID} has factor parameter of 0 which is invalid\n'
                    )

                default_delay = (self.block_durations[i_block] - block.soft_delay.offset) * block.soft_delay.factor
                soft_num = int(block.soft_delay.numID)
                if soft_num >= 0:
                    while len(soft_delay_state) < soft_num + 1:
                        soft_delay_state.append(None)
                    if soft_delay_state[soft_num] is None:
                        soft_delay_state[soft_num] = {
                            'def': default_delay,
                            'hint': block.soft_delay.hint,
                            'blk': i_block,
                            'min': 0.0,
                            'max': np.inf,
                        }
                    else:
                        prev = soft_delay_state[soft_num]
                        if abs(default_delay - prev['def']) > 1e-7:
                            error_report.append(
                                f"   Block:{i_block} soft delay {block.soft_delay.hint}/{soft_num}: default duration derived from this block ({default_delay * 1e6}us) is inconsistent with the previous default ({prev['def'] * 1e6}us) that was derived from block {prev['blk']}\n"
                            )
                        if block.soft_delay.hint != prev['hint']:
                            error_report.append(
                                f"   Block:{i_block} soft delay {block.soft_delay.hint}/{soft_num}: soft delays with the same numeric ID are expected to share the same text hint but previous hint recorded in block {prev['blk']} is {prev['hint']}\n"
                            )

                    limit_delay = (-block.soft_delay.offset) * block.soft_delay.factor
                    if block.soft_delay.factor > 0:
                        soft_delay_state[soft_num]['min'] = max(soft_delay_state[soft_num]['min'], limit_delay)
                    else:
                        soft_delay_state[soft_num]['max'] = min(soft_delay_state[soft_num]['max'], limit_delay)
                else:
                    error_report.append(
                        f'   Block:{i_block} contains a soft delay {block.soft_delay.hint} with an invalid numeric ID{soft_num}\n'
                    )

        easy_struct = {}
        for index, state in enumerate(soft_delay_state):
            if state is None:
                warn(
                    f'SoftDelay numeric ID {index} is unused, we expect contiguous numbering of soft delays',
                    stacklevel=2,
                )
                continue
            if state['hint'] in easy_struct:
                raise ValueError(
                    f"SoftDelay with numeric ID {index} uses the same hint '{state['hint']}' as some previous SoftDelay"
                )
            easy_struct[state['hint']] = state['def']

        return easy_struct, error_report, soft_delay_state

    def sound(
        self,
        blockRange: Union[List[int], Tuple[int, int], None] = None,
        channelWeights: Union[List[float], Tuple[float, float, float], np.ndarray] = (1, 1, 1),
        sampleRate: int = 44100,
        onlyProduceSoundData: bool = False,
    ) -> np.ndarray:
        if blockRange is None:
            blockRange = [1, np.inf]
        if not np.isfinite(blockRange[1]):
            blockRange = [blockRange[0], len(self.block_events)]

        wave_data, _, _, _, _, _ = self.waveforms_and_times(append_RF=False, blockRange=blockRange)
        total_duration = float(sum(self.block_durations.values()))

        sample_rate = int(sampleRate)
        dwell_time = 1.0 / sample_rate
        sound_length = int(np.floor(total_duration / dwell_time) + 1)
        sound_data = np.zeros((2, sound_length), dtype=float)
        t_axis = np.arange(sound_length, dtype=float) * dwell_time
        weights = np.asarray(channelWeights, dtype=float).reshape(3)

        if wave_data[0].size:
            sound_data[0, :] = np.interp(t_axis, wave_data[0][0, :], wave_data[0][1, :] * weights[0], left=0.0, right=0.0)
        if wave_data[1].size:
            sound_data[1, :] = np.interp(t_axis, wave_data[1][0, :], wave_data[1][1, :] * weights[1], left=0.0, right=0.0)
        if wave_data[2].size:
            tmp = np.interp(t_axis, wave_data[2][0, :], 0.5 * wave_data[2][1, :] * weights[2], left=0.0, right=0.0)
            sound_data[0, :] += tmp
            sound_data[1, :] += tmp

        gw_len = int(round(sample_rate / 6000.0) * 2 + 1)
        x = np.linspace(-1.0, 1.0, gw_len)
        gw = np.exp(-0.5 * (x / 0.25) ** 2)
        gw /= np.sum(gw)
        sound_data[0, :] = np.convolve(sound_data[0, :], gw, mode='same')
        sound_data[1, :] = np.convolve(sound_data[1, :], gw, mode='same')

        sound_max = float(np.max(np.abs(sound_data))) if sound_data.size else 0.0
        if sound_max > 0:
            sound_data = 0.95 * sound_data / sound_max

        if not onlyProduceSoundData:
            warn('sound() playback is not implemented in PyPulseq; returning sound data only.', stacklevel=2)

        return sound_data

    def test_report(self) -> str:
        """
        Analyze the sequence and return a text report.
        """
        return ext_test_report(self)

    def waveforms(
        self,
        append_RF: bool = False,
        time_range: Union[List[float], None] = None,
        blockRange: Union[List[int], Tuple[int, int], None] = None,
    ) -> Tuple[np.ndarray]:
        """
        Decompress the entire gradient waveform. Returns gradient waveforms as a tuple of `np.ndarray` of
        `gradient_axes` (typically 3) dimensions. Each `np.ndarray` contains timepoints and the corresponding
        gradient amplitude values.

        Parameters
        ----------
        append_RF : bool, default=False
            Boolean flag to indicate if RF wave shapes are to be appended after the gradients.

        Returns
        -------
        wave_data : np.ndarray
        """
        wave_data, _, _, _, _, _ = self._waveforms_and_times_impl(
            append_RF=append_RF, time_range=time_range, blockRange=blockRange
        )
        return wave_data

    def waveforms_and_times(
        self,
        append_RF: bool = False,
        time_range: Union[List[float], None] = None,
        blockRange: Union[List[int], Tuple[int, int], None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompress the entire gradient waveform. Returns gradient waveforms as a tuple of `np.ndarray` of
        `gradient_axes` (typically 3) dimensions. Each `np.ndarray` contains timepoints and the corresponding
        gradient amplitude values. Additional return values are time points of excitations, refocusings and ADC
        sampling points.

        Parameters
        ----------
        append_RF : bool, default=False
            Boolean flag to indicate if RF wave shapes are to be appended after the gradients.

        Returns
        -------
        wave_data : np.ndarray
        tfp_excitation : np.ndarray
            Contains time moments, frequency and phase offsets of the excitation RF pulses (similar for `
            tfp_refocusing`).
        tfp_refocusing : np.ndarray
        t_adc: np.ndarray
            Contains times of all ADC sample points.
        fp_adc : np.ndarray
            Contains frequency and phase offsets of each ADC sample.
        pm_adc : np.ndarray
            Contains phase modulation of each ADC sample.
        """
        wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc, pm_adc = self._waveforms_and_times_impl(
            append_RF=append_RF, time_range=time_range, blockRange=blockRange
        )
        return wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc, pm_adc

    def _resolve_blocks(
        self,
        blockRange: Union[List[int], Tuple[int, int], None] = None,
        time_range: Union[List[float], None] = None,
    ) -> Tuple[List[int], float]:
        if blockRange is not None and time_range is not None:
            raise ValueError('Specify either blockRange or time_range, not both')

        if blockRange is not None:
            if len(blockRange) != 2:
                raise ValueError("parameter 'blockRange' must contain exactly two numbers")
            block_start = max(int(blockRange[0]), 1)
            block_stop = len(self.block_events) if not np.isfinite(blockRange[1]) else int(blockRange[1])
            blocks = list(self.block_durations.keys())[block_start - 1 : block_stop]
            curr_dur = 0.0
            return blocks, curr_dur

        curr_dur = 0.0
        if time_range is None:
            return list(self.block_events.keys()), curr_dur

        if len(time_range) != 2:
            raise ValueError('Time range must be list of two elements')
        if time_range[0] > time_range[1]:
            raise ValueError('End time of time_range must be after begin time')

        bd = np.array(list(self.block_durations.values()))
        t = np.cumsum(bd)
        begin_block = np.searchsorted(t, time_range[0])
        end_block = np.searchsorted(t - bd, time_range[1], side='right')
        blocks = list(self.block_durations.keys())[begin_block:end_block]
        curr_dur = t[begin_block] - bd[begin_block]
        return blocks, curr_dur

    @staticmethod
    def _get_external_field(container: Any, name: str, default=None):
        if isinstance(container, dict):
            return container.get(name, default)
        return getattr(container, name, default)

    def _waveforms_and_times_impl(
        self,
        append_RF: bool = False,
        time_range: Union[List[float], None] = None,
        blockRange: Union[List[int], Tuple[int, int], None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if blockRange is not None and time_range is not None:
            raise ValueError('Specify either blockRange or time_range, not both')

        grad_channels = ['gx', 'gy', 'gz']
        shape_channels = len(grad_channels) + 1 if append_RF else len(grad_channels)
        shape_pieces = [[] for _ in range(shape_channels)]
        out_len = np.zeros(shape_channels, dtype=int)

        tfp_excitation = []
        tfp_refocusing = []
        t_adc = []
        fp_adc = []
        pm_adc = []

        blocks, curr_dur = self._resolve_blocks(blockRange=blockRange, time_range=time_range)

        for block_counter in blocks:
            block = self.get_block(block_counter)
            if getattr(block, 'rotation', None) is not None:
                rotated_events = rotate_3d(block.rotation.rot_quaternion, block, 'system', self.system)
                rotated_block = deepcopy(block)
                rotated_block.gx = None
                rotated_block.gy = None
                rotated_block.gz = None
                for event in rotated_events:
                    if hasattr(event, 'type') and hasattr(event, 'channel') and event.type in ['grad', 'trap']:
                        setattr(rotated_block, 'g' + event.channel, event)
                block = rotated_block

            for j, grad_name in enumerate(grad_channels):
                grad = getattr(block, grad_name)
                if grad is None:
                    continue
                if grad.type == 'grad':
                    tt_rast = grad.tt / self.grad_raster_time
                    if np.all(np.abs(tt_rast - (np.arange(1, len(tt_rast) + 1) - 0.5)) < 1e-6):
                        tt_chg, waveform_chg = restore_additional_shape_samples(
                            grad.tt,
                            grad.waveform,
                            grad.first,
                            grad.last,
                            self.grad_raster_time,
                            i_block=block_counter,
                        )
                        piece = np.vstack((curr_dur + grad.delay + tt_chg, waveform_chg))
                    else:
                        if abs(tt_rast[0] - 0.5) < 1e-6:
                            piece = np.vstack(
                                (
                                    curr_dur + grad.delay + np.concatenate(([0.0], grad.tt, [grad.shape_dur])),
                                    np.concatenate(([grad.first], grad.waveform, [grad.last])),
                                )
                            )
                        else:
                            piece = np.vstack((curr_dur + grad.delay + grad.tt, grad.waveform))
                else:
                    if abs(grad.flat_time) > eps:
                        piece = np.vstack(
                            (
                                cumsum(curr_dur + grad.delay, grad.rise_time, grad.flat_time, grad.fall_time),
                                grad.amplitude * np.array([0, 1, 1, 0]),
                            )
                        )
                    elif abs(grad.rise_time) > eps and abs(grad.fall_time) > eps:
                        piece = np.vstack(
                            (
                                cumsum(curr_dur + grad.delay, grad.rise_time, grad.fall_time),
                                grad.amplitude * np.array([0, 1, 0]),
                            )
                        )
                    else:
                        if abs(grad.amplitude) > eps:
                            warn(f'"empty" gradient with non-zero magnitude detected in block {block_counter}')
                        piece = None

                if piece is not None:
                    out_len[j] += piece.shape[1]
                    shape_pieces[j].append(piece)

            if block.rf is not None:
                rf = block.rf
                tc = calc_rf_center(rf)[0]
                t = rf.delay + tc
                full_freq_offset = rf.freq_offset + rf.freq_ppm * 1e-6 * self.system.gamma * self.system.B0
                full_phase_offset = rf.phase_offset + rf.phase_ppm * 1e-6 * self.system.gamma * self.system.B0
                tfp_col = [curr_dur + t, full_freq_offset, full_phase_offset + 2 * math.pi * full_freq_offset * tc]
                if not hasattr(rf, 'use') or rf.use in ['excitation', 'undefined']:
                    tfp_excitation.append(tfp_col)
                elif rf.use == 'refocusing':
                    tfp_refocusing.append(tfp_col)

                if append_RF:
                    pre = np.zeros((2, 0))
                    post = np.zeros((2, 0))
                    if abs(rf.signal[0]) > 0:
                        pre = np.array([[curr_dur + rf.delay + rf.t[0] - eps], [0.0]])
                    if abs(rf.signal[-1]) > 0:
                        post = np.array([[curr_dur + rf.delay + rf.t[-1] + eps], [0.0]])
                    rf_piece = np.vstack(
                        (
                            curr_dur + rf.delay + rf.t,
                            rf.signal * np.exp(1j * (full_phase_offset + 2 * math.pi * full_freq_offset * rf.t)),
                        )
                    )
                    rf_piece = np.hstack((pre, rf_piece, post))
                    out_len[-1] += rf_piece.shape[1]
                    shape_pieces[-1].append(rf_piece)

            if block.adc is not None:
                adc = block.adc
                ta = adc.dwell * (np.arange(adc.num_samples) + 0.5)
                adc_times = ta + adc.delay + curr_dur
                t_adc.append(adc_times)
                full_freq_offset = adc.freq_offset + adc.freq_ppm * 1e-6 * self.system.gamma * self.system.B0
                full_phase_offset = adc.phase_offset + adc.phase_ppm * 1e-6 * self.system.gamma * self.system.B0
                phase_modulation = getattr(adc, 'phase_modulation', None)
                if phase_modulation is None or len(phase_modulation) == 0:
                    phase_modulation = np.zeros(adc.num_samples)
                else:
                    phase_modulation = np.asarray(phase_modulation)
                pm_adc.append(phase_modulation)
                fp_adc.append(
                    np.vstack(
                        (
                            full_freq_offset * np.ones(adc.num_samples),
                            full_phase_offset + phase_modulation + full_freq_offset * ta,
                        )
                    )
                )

            curr_dur += self.block_durations[block_counter]

        wave_data = [self._assemble_wave_channel(pieces, axis_index=j + 1) for j, pieces in enumerate(shape_pieces)]

        tfp_excitation_arr = np.array(tfp_excitation, dtype=float).T if tfp_excitation else np.zeros((3, 0))
        tfp_refocusing_arr = np.array(tfp_refocusing, dtype=float).T if tfp_refocusing else np.zeros((3, 0))
        t_adc_arr = np.concatenate(t_adc) if t_adc else np.zeros(0)
        fp_adc_arr = np.hstack(fp_adc) if fp_adc else np.zeros((2, 0))
        pm_adc_arr = np.concatenate(pm_adc) if pm_adc else np.zeros(0)

        return wave_data, tfp_excitation_arr, tfp_refocusing_arr, t_adc_arr, fp_adc_arr, pm_adc_arr

    def _assemble_wave_channel(self, pieces: List[np.ndarray], axis_index: int) -> np.ndarray:
        if not pieces:
            return np.zeros((2, 0))

        assembled = pieces[0]
        for cur in pieces[1:]:
            if assembled[0, -1] + self.grad_raster_time < cur[0, 0]:
                if assembled[1, -1] != 0:
                    if abs(assembled[1, -1]) > 1e-6:
                        warn(
                            f'waveforms_and_times(): forcing ramp-down from a non-zero gradient sample on axis {axis_index} '
                            f"at t={round(1e6 * assembled[0, -1])} us \n"
                            "check your sequence, some calculations are possibly wrong. "
                            "If using mr.makeArbitraryGrad() consider using explicit values for 'first' and 'last' "
                            "and setting them correctly."
                        )
                        assembled = np.hstack(
                            (assembled, np.array([[assembled[0, -1] + self.grad_raster_time / 2], [0.0]]))
                        )
                    else:
                        assembled[1, -1] = 0.0
                if cur[1, 0] != 0:
                    if abs(cur[1, 0]) > 1e-6:
                        warn(
                            f'waveforms_and_times(): forcing ramp-up to a non-zero gradient sample on axis {axis_index} '
                            f"at t={round(1e6 * cur[0, 0])} us \n"
                            "check your sequence, some calculations are probably wrong. "
                            "If using mr.makeArbitraryGrad() consider using explicit values for 'first' and 'last' "
                            "and setting them correctly."
                        )
                        cur = np.hstack((np.array([[cur[0, 0] - self.grad_raster_time / 2], [0.0]]), cur))
                    else:
                        cur[1, 0] = 0.0

            if assembled[0, -1] < cur[0, 0]:
                assembled = np.hstack((assembled, cur))
            else:
                if cur[0, 0] < assembled[0, -1] - 1e-9:
                    warn('Warning: looks like rounding errors for some elements exceed the acceptable tolerance!')
                mask = cur[0, :] > assembled[0, -1]
                if np.any(mask):
                    first_idx = int(np.argmax(mask))
                    assembled = np.hstack((assembled, cur[:, first_idx:]))

        if np.any(np.diff(assembled[0, :]) <= 0):
            warn('Warning: not all elements of the generated time vector are unique and sorted in accending order!')

        return assembled

    def write(
        self,
        name: str,
        create_signature: bool = True,
        remove_duplicates: bool = True,
        check_timing: bool = True,
        v141_compat: bool = False,
    ) -> Union[str, None]:
        """
        Write the sequence data to the given filename using the open file format for MR sequences.

        See also `pypulseq.Sequence.read_seq.read()`.

        Parameters
        ----------
        name : str
            Filename of `.seq` file to be written to disk.
        create_signature : bool, default=True
            Boolean flag to indicate if the file has to be signed.
        remove_duplicates : bool, default=True
            Remove duplicate events from the sequence before writing
        v141_compat: bool, default=False
            Write the sequence in v1.4.1 compatible file format.
        Returns
        -------
        signature or None : If create_signature is True, it returns the written .seq file's signature as a string,
        otherwise it returns None. Note that, if remove_duplicates is True, signature belongs to the
        deduplicated sequences signature, and not the Sequence that is stored in the Sequence object.
        """
        # Check if there are any timing errors in the sequence
        if check_timing:
            is_ok, error_report = self.check_timing()
            if not is_ok:
                warn(f'write(): {len(error_report)} timing errors found in the sequence', stacklevel=2)

        # Calculate sequence duration and stored it in the TotalDuration definition
        self.set_definition('TotalDuration', sum(self.block_durations.values()))

        # Check whether all gradients in the last block are ramped down properly
        last_block_id = next(reversed(self.block_events))
        last_block = self.get_block(last_block_id)
        for channel, event in zip(('x', 'y', 'z'), (last_block.gx, last_block.gy, last_block.gz), strict=False):
            if (
                event is not None
                and event.type == 'grad'
                and abs(event.last) > self.system.max_slew * self.system.grad_raster_time
            ):
                warn_msg = f'write(): Gradient on channel {channel} in last sequence block does not ramp down to 0'

                if trace_enabled():
                    trace = self.block_trace.get(last_block_id, None)

                    if hasattr(trace, 'block'):
                        warn_msg += '\nLast block defined here:\n' + format_trace(trace.block)
                    if hasattr(trace, 'g' + channel):
                        warn_msg += f'\n`g{channel}` defined here:\n' + format_trace(getattr(trace, 'g' + channel))

                warn(warn_msg, stacklevel=2)

        if v141_compat:
            signature = write_seq_v141(self, name, create_signature, remove_duplicates)
        else:
            signature = write_seq(self, name, create_signature, remove_duplicates)

        # Return the sequence md5 signature if requested
        if signature is not None:
            self.signature_type = 'md5'
            self.signature_file = 'text'
            self.signature_value = signature
            return signature
        else:
            return None

    def write_file(self, filename: str) -> None:
        self.write(filename, create_signature=False)

    @staticmethod
    def get_binary_codes():
        def signed_int64(value):
            value &= (1 << 64) - 1
            if value >= (1 << 63):
                value -= 1 << 64
            return value

        prefix = 0xFFFFFFFF << 32
        return {
            'fileHeader': int.from_bytes(b'\x01pulseq\x02', byteorder='little', signed=True),
            'section': {
                'definitions': signed_int64(prefix | 1),
                'blocks': signed_int64(prefix | 2),
                'rf': signed_int64(prefix | 3),
                'gradients': signed_int64(prefix | 4),
                'trapezoids': signed_int64(prefix | 5),
                'adc': signed_int64(prefix | 6),
                'delays': signed_int64(prefix | 7),
                'shapes': signed_int64(prefix | 8),
                'extensions': signed_int64(prefix | 9),
                'triggers': signed_int64(prefix | 10),
                'labelset': signed_int64(prefix | 11),
                'labelinc': signed_int64(prefix | 12),
                'softdelays': signed_int64(prefix | 13),
                'rfshims': signed_int64(prefix | 14),
                'rotations': signed_int64(prefix | 15),
                'signature': signed_int64(prefix | 0x00FFFFFF),
            },
        }

Sequence.calc_moments_b_tensor = calc_moments_b_tensor
Sequence.auto_label = auto_label
Sequence.read_binary = read_binary
Sequence.write_binary = write_binary
