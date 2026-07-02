from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pypulseq.block_to_events import block_to_events
from pypulseq.calc_rf_bandwidth import calc_rf_bandwidth
from pypulseq.make_label import make_label


def _adc_blocks(seq, block_range):
    first, last = int(block_range[0]), int(block_range[1])
    out = []
    for block_id in range(first, last + 1):
        block_event = seq.block_events.get(block_id)
        if block_event is None or len(block_event) < 6 or int(block_event[5]) == 0:
            continue
        out.append((block_id, seq.get_block(block_id)))
    return out


def _as_label_dict(labels):
    if labels is None:
        return {}
    if isinstance(labels, SimpleNamespace):
        return vars(labels)
    return dict(labels)


def _num_samples(adc):
    return getattr(adc, 'num_samples', 0)


def auto_label(
    self,
    block_range=None,
    use_labels=None,
    use_aux=None,
    skip_apply: bool = False,
    plot: bool = False,
    mirror_fourier: bool = False,
    reflect=None,
    reorder=None,
    no_plots: bool = False,
):
    if block_range is None:
        block_range = [1, np.inf]
    if not np.isfinite(block_range[1]):
        block_range = [block_range[0], len(self.block_events)]
    block_range = [int(block_range[0]), int(block_range[1])]

    reflect = [] if reflect is None else [int(x) - 1 for x in np.asarray(reflect).reshape(-1)]
    reorder = [] if reorder is None else [int(x) - 1 for x in np.asarray(reorder).reshape(-1)]
    if use_labels is not None and (reflect or reorder or mirror_fourier):
        raise ValueError(
            "'reflect', 'reorder' or 'mirror_fourier' only affect detection and cannot be used together with 'use_labels'"
        )
    if len(reflect) != len(set(reflect)):
        raise ValueError("All indices in 'reflect' must be unique")
    if any(axis < 0 or axis > 2 for axis in reflect):
        raise ValueError("'reflect' indices must be in the range 1..3")
    if reorder:
        if len(reorder) != len(set(reorder)):
            raise ValueError("All indices in 'reorder' must be unique")
        if any(axis < 0 or axis > 2 for axis in reorder):
            raise ValueError("'reorder' indices must be in the range 1..3")
        if len(reorder) != 3 and set(reorder) != {0, 1}:
            raise ValueError("If 'reorder' contains two indices they must be [1, 2] or [2, 1]")

    adc_blocks = _adc_blocks(self, block_range)
    n_adcs = len(adc_blocks)
    aux = {}

    labels = _as_label_dict(use_labels)
    if not labels:
        ktraj_adc, t_adc, _, _, t_excitation, _, slicepos, t_slicepos, gw_pp, _ = self.calculate_kspacePP(
            blockRange=block_range
        )
        labels = {}

        adc_lengths = [_num_samples(block.adc) for _, block in adc_blocks]
        adc_starts = np.cumsum([0, *adc_lengths[:-1]]).astype(int) if adc_lengths else np.array([], dtype=int)
        t_adc_starts = np.asarray(t_adc)[adc_starts] if len(adc_starts) else np.array([])
        block_start_times = np.cumsum([0.0, *[self.block_durations[i] for i in range(block_range[0], block_range[1] + 1)]])

        first_non_noise_adc = 0
        first_non_noise_sample = 0
        if len(t_excitation) != 0 and len(t_adc_starts) != 0:
            after_excitation = np.flatnonzero(t_adc_starts > np.asarray(t_excitation)[0])
            if after_excitation.size:
                first_non_noise_adc = int(after_excitation[0])
                after_excitation_samples = np.flatnonzero(np.asarray(t_adc) > np.asarray(t_excitation)[0])
                if after_excitation_samples.size:
                    first_non_noise_sample = int(after_excitation_samples[0])
                if first_non_noise_adc > 0:
                    noise = np.zeros(n_adcs, dtype=int)
                    noise[:first_non_noise_adc] = 1
                    labels['NOISE'] = noise

        ktraj_adc = np.asarray(ktraj_adc, dtype=float)
        if mirror_fourier:
            ktraj_adc = -ktraj_adc
        if reflect:
            ktraj_adc[reflect, :] = -ktraj_adc[reflect, :]
        if reorder:
            n_reorder = len(reorder)
            ktraj_adc[:n_reorder, :] = ktraj_adc[reorder, :]

        if slicepos is not None and np.asarray(slicepos).size and len(t_adc_starts) != 0:
            slicepos = np.asarray(slicepos, dtype=float)
            t_slicepos = np.asarray(t_slicepos, dtype=float)
            slice_grads = np.zeros_like(slicepos)
            for axis in range(min(3, len(gw_pp))):
                if gw_pp[axis] is not None:
                    slice_grads[axis, :] = gw_pp[axis](t_slicepos)
            if reflect:
                slicepos[reflect, :] = -slicepos[reflect, :]
                slice_grads[reflect, :] = -slice_grads[reflect, :]
            if reorder:
                n_reorder = len(reorder)
                slicepos[:n_reorder, :] = slicepos[reorder, :]
                slice_grads[:n_reorder, :] = slice_grads[reorder, :]
            dominant_axis = np.argmax(np.abs(slice_grads), axis=0)
            signs = np.sign(slice_grads[dominant_axis, np.arange(slice_grads.shape[1])])
            signs[signs == 0] = 1
            norms = np.linalg.norm(slice_grads, axis=0)
            normals = np.divide(slice_grads, norms, out=np.zeros_like(slice_grads), where=norms > 0) * signs
            offsets = np.sum(slicepos * normals, axis=0)
            offsets[~np.isfinite(offsets)] = 0
            unique_offsets = []
            slc = np.zeros(n_adcs, dtype=int)
            for adc_idx in range(first_non_noise_adc, n_adcs):
                prev = np.flatnonzero(t_slicepos < t_adc_starts[adc_idx])
                if prev.size == 0:
                    continue
                offset = offsets[prev[-1]]
                match = next((i for i, value in enumerate(unique_offsets) if np.isclose(value, offset)), None)
                if match is None:
                    unique_offsets.append(offset)
                    match = len(unique_offsets) - 1
                slc[adc_idx] = match
            if unique_offsets:
                labels['SLC'] = slc
                aux['SlicePositions'] = np.asarray(unique_offsets)
                grad_norm = float(np.linalg.norm(slice_grads[:, 0])) if slice_grads.size else 0.0
                if grad_norm > 0:
                    slice_block_rel = int(np.searchsorted(block_start_times, t_slicepos[0], side='right'))
                    slice_block = self.get_block(block_range[0] + slice_block_rel - 1)
                    if getattr(slice_block, 'rf', None) is not None:
                        aux['SliceThickness'] = calc_rf_bandwidth(slice_block.rf) / grad_norm
                        if len(unique_offsets) > 1:
                            aux['SliceGap'] = unique_offsets[1] - unique_offsets[0] - aux['SliceThickness']

        if len(adc_lengths) and ktraj_adc.size:
            centers = []
            rev = np.zeros(n_adcs, dtype=int)
            echo_positions = np.zeros(n_adcs, dtype=int)
            echo_times = np.zeros(n_adcs, dtype=float)
            grad_readout = np.zeros((3, n_adcs), dtype=float)

            center_sample = first_non_noise_sample
            if ktraj_adc[:, first_non_noise_sample:].size:
                center_sample += int(np.argmin(np.linalg.norm(ktraj_adc[:, first_non_noise_sample:], axis=0)))
            center_point = ktraj_adc[:, center_sample] if center_sample < ktraj_adc.shape[1] else np.zeros(3)

            for adc_idx, start in enumerate(adc_starts):
                stop = start + adc_lengths[adc_idx]
                segment = ktraj_adc[:, start:stop]
                if segment.size == 0:
                    centers.append(np.zeros(ktraj_adc.shape[0]))
                    continue
                echo_pos = int(np.argmin(np.linalg.norm(segment - center_point[:, None], axis=0)))
                echo_positions[adc_idx] = echo_pos
                k_echo = segment[:, echo_pos]
                centers.append(k_echo)
                echo_index = start + echo_pos
                echo_time = float(t_adc[echo_index])
                if np.linalg.norm(center_point) > np.finfo(float).eps:
                    indices_to_check = []
                    if echo_pos > 0:
                        indices_to_check.append(echo_index - 1)
                    if echo_index < stop - 1:
                        indices_to_check.append(echo_index + 1)
                    for index_to_check in indices_to_check:
                        v_i_to_0 = -k_echo
                        v_i_to_t = ktraj_adc[:, index_to_check] - k_echo
                        denom = np.linalg.norm(v_i_to_t) ** 2
                        if denom == 0:
                            continue
                        p_vit = float(np.matmul(v_i_to_0, v_i_to_t) / denom)
                        if p_vit > 0:
                            echo_time = echo_time * (1 - p_vit) + float(t_adc[index_to_check]) * p_vit
                            break
                echo_times[adc_idx] = echo_time
                for axis in range(min(3, len(gw_pp))):
                    if gw_pp[axis] is not None:
                        grad_readout[axis, adc_idx] = gw_pp[axis](echo_time)
                if mirror_fourier:
                    grad_readout[:, adc_idx] = -grad_readout[:, adc_idx]
                if reflect:
                    grad_readout[reflect, adc_idx] = -grad_readout[reflect, adc_idx]
                if reorder:
                    n_reorder = len(reorder)
                    grad_readout[:n_reorder, adc_idx] = grad_readout[reorder, adc_idx]
                if stop - start >= 2:
                    delta = segment[:, -1] - segment[:, 0]
                    main = int(np.argmax(np.abs(delta)))
                    rev[adc_idx] = int(delta[main] < 0)
            centers = np.asarray(centers).T
            if centers.size:
                extent = np.max(np.abs(centers - centers[:, :1]), axis=1)
                active = np.flatnonzero(extent > max(np.max(extent), 1.0) / 4e6)
                if active.size >= 1:
                    lin_values = np.round(centers[active[0]] - np.min(centers[active[0]])).astype(int)
                    labels['LIN'] = lin_values
                if active.size >= 2:
                    par_values = np.round(centers[active[1]] - np.min(centers[active[1]])).astype(int)
                    labels['PAR'] = par_values
                if np.any(rev):
                    labels['REV'] = rev

                central_adc = int(np.searchsorted(adc_starts, center_sample, side='right') - 1)
                central_adc = max(0, min(central_adc, n_adcs - 1))
                aux['kSpaceCenterSample'] = int(echo_positions[central_adc])
                if 'LIN' in labels:
                    aux['kSpaceCenterLine'] = int(np.asarray(labels['LIN'])[central_adc])
                if 'PAR' in labels:
                    aux['kSpaceCenterPartition'] = int(np.asarray(labels['PAR'])[central_adc])

                if 0 <= central_adc < len(adc_starts):
                    c1 = int(adc_starts[central_adc])
                    c2 = int(c1 + adc_lengths[central_adc])
                    projection_axis = int(np.argmax(np.abs(grad_readout[:, central_adc])))
                    sign = np.sign(grad_readout[projection_axis, central_adc]) or 1
                    central_projection = sign * ktraj_adc[projection_axis, c1:c2]
                    if central_projection.size > 2:
                        dk = np.median(np.diff(central_projection))
                        if dk != 0 and np.any(np.abs(np.diff(central_projection, 2) / dk * central_projection.size) > 0.1):
                            block_rel = int(np.searchsorted(block_start_times, t_adc_starts[central_adc], side='right'))
                            block = self.get_block(block_range[0] + block_rel - 1)
                            if reorder and reorder[0] != 0:
                                pass
                            elif getattr(block, 'gx', None) is not None and block.gx.type == 'trap' and getattr(block, 'adc', None) is not None:
                                aux['TrapezoidGriddingParameters'] = np.array(
                                    [
                                        block.gx.rise_time,
                                        block.gx.flat_time,
                                        block.gx.fall_time,
                                        block.adc.delay - block.gx.delay,
                                        block.adc.num_samples * block.adc.dwell,
                                    ]
                                )
                                aux['TargetGriddedSamples'] = int(block.adc.num_samples)

                is_navigator = np.zeros(n_adcs, dtype=bool)
                if n_adcs >= 16:
                    center_point = ktraj_adc[:, center_sample] if center_sample < ktraj_adc.shape[1] else np.zeros(3)
                    nav_candidates = np.linalg.norm(centers - center_point[:, None], axis=0) < 1e-4
                    ordered = np.diff(centers - center_point[:, None], axis=1)
                    active_order = ordered[np.max(np.abs(ordered), axis=1) > 1e-4, :]
                    active_order_flat = active_order.reshape(-1)
                    if (
                        active_order_flat.size >= 16
                        and np.all(nav_candidates[:3])
                        and abs(active_order_flat[0] - active_order_flat[1]) < 1e-4
                        and np.max(np.abs(np.diff(active_order_flat[3:16]))) < 1e-4
                    ):
                        aux['epiWithThreeEchoNavigator'] = True
                        is_navigator = (nav_candidates.astype(int) + np.roll(nav_candidates, 1) + np.roll(nav_candidates, -1)) > 1.5
                        labels['NAV'] = is_navigator.astype(int)

                repeat = np.zeros(n_adcs, dtype=int)
                valid_adc_indices = [idx for idx in range(first_non_noise_adc, n_adcs) if not is_navigator[idx]]
                if valid_adc_indices:
                    active_for_repeat = active if active.size else np.arange(centers.shape[0])
                    repeat_counts = {}
                    quant_scale = max(np.max(np.abs(centers[active_for_repeat, valid_adc_indices])), 1.0) / 4e6
                    for adc_idx in valid_adc_indices:
                        spatial_key = tuple(np.round(centers[active_for_repeat, adc_idx] / quant_scale).astype(int))
                        if 'SLC' in labels:
                            spatial_key = (*spatial_key, int(np.asarray(labels['SLC'])[adc_idx]))
                        repeat[adc_idx] = repeat_counts.get(spatial_key, 0)
                        repeat_counts[spatial_key] = repeat[adc_idx] + 1

                    n_rep = int(np.max(repeat)) + 1
                    if n_rep > 1:
                        rep_te = np.zeros(n_rep, dtype=float)
                        skip_eco = False
                        for rep_idx in range(n_rep):
                            adc_mask = [idx for idx in valid_adc_indices if repeat[idx] == rep_idx]
                            if not adc_mask:
                                skip_eco = True
                                break
                            rep_te[rep_idx] = float(np.median(echo_times[adc_mask]))
                        if not skip_eco:
                            te_order = np.argsort(rep_te)
                            te_sorted = rep_te[te_order]
                            cluster_ids = np.cumsum(np.r_[0, np.diff(te_sorted) > 10e-6])
                            unique_te = np.array(
                                [np.mean(te_sorted[cluster_ids == cluster_id]) for cluster_id in range(cluster_ids[-1] + 1)]
                            )
                            rep_to_echo = np.zeros(n_rep, dtype=int)
                            for sorted_idx, rep_idx in enumerate(te_order):
                                rep_to_echo[rep_idx] = int(cluster_ids[sorted_idx])
                            aux['TE'] = unique_te

                            echo = np.zeros(n_adcs, dtype=int)
                            new_repeat = np.zeros(n_adcs, dtype=int)
                            echo_rep_counts = {}
                            for rep_idx in range(n_rep):
                                echo_idx = int(rep_to_echo[rep_idx])
                                echo_rep_counts.setdefault(echo_idx, 0)
                                adc_mask = [idx for idx in valid_adc_indices if repeat[idx] == rep_idx]
                                for adc_idx in adc_mask:
                                    echo[adc_idx] = echo_idx
                                    new_repeat[adc_idx] = echo_rep_counts[echo_idx]
                                echo_rep_counts[echo_idx] += 1
                            repeat = new_repeat
                            if np.max(echo) > 0:
                                labels['ECO'] = echo
                    if np.max(repeat) > 0:
                        labels['REP'] = repeat

    if use_aux is not None:
        aux = dict(use_aux) if not isinstance(use_aux, SimpleNamespace) else vars(use_aux).copy()

    if not skip_apply:
        for adc_idx, (block_id, block) in enumerate(adc_blocks):
            label_events = []
            for name, values in labels.items():
                values = np.asarray(values).reshape(-1)
                if adc_idx < len(values):
                    label_events.append(make_label(name, 'SET', float(values[adc_idx])))
            if label_events:
                events = list(block_to_events(block))
                events.extend(label_events)
                self.set_block(block_id, *events)

        for field in [
            'kSpaceCenterLine',
            'kSpaceCenterPartition',
            'kSpaceCenterSample',
            'kSpacePhaseEncodingLines',
            'PhaseResolution',
            'ReadoutOversamplingFactor',
            'SliceGap',
            'SlicePositions',
            'SliceThickness',
            'TargetGriddedSamples',
            'TrapezoidGriddingParameters',
            'AccelerationFactorPE',
            'AccelerationFactor3D',
            'FirstFourierLine',
            'FirstRefLine',
            'FirstFourier3D',
            'FirstRef3D',
        ]:
            if field in aux:
                previous = self.get_definition(field)
                if not (isinstance(previous, str) and previous == ''):
                    import warnings

                    warnings.warn(f'Overwriting existing sequence definition {field} = {previous}', stacklevel=2)
                self.set_definition(field, aux[field])

    if no_plots:
        return labels, aux

    if plot:
        try:
            import matplotlib.pyplot as plt

            for name, values in labels.items():
                plt.plot(np.asarray(values).reshape(-1), label=name)
            plt.legend()
            plt.show()
        except ImportError:
            pass

    return labels, aux
