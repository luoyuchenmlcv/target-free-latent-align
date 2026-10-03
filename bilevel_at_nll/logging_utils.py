from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import wandb
except Exception:
    wandb = None


_PLOT_STATE = {
    "inner_eval_rounds": [],
    "inner_eval_pre": [],
    "inner_eval_post": [],
    "mean_shift_norm_rounds": [],
    "mean_shift_norm_inner": [],
    "mean_shift_norm_outer": [],
    "mean_shift_dir_sim_rounds": [],
    "mean_shift_dir_sim_inner": [],
    "mean_shift_dir_sim_outer": [],
    "outer_loss_rounds": [],
    "outer_loss_harmful_start": [],
    "outer_loss_harmful_end": [],
    "intra_prompt_consistency_rounds": [],
    "intra_prompt_consistency_inner": [],
}


def init_wandb(args):
    if wandb is None or getattr(args, 'wandb_mode', 'disabled') == 'disabled':
        return None
    tags = [x.strip() for x in str(getattr(args, 'wandb_tags', '')).split(',') if x.strip()]
    run = wandb.init(
        project=getattr(args, 'wandb_project', 'bilevel-at'),
        entity=getattr(args, 'wandb_entity', None),
        name=getattr(args, 'wandb_name', None),
        mode=getattr(args, 'wandb_mode', 'online'),
        tags=tags,
        config=vars(args),
    )
    try:
        wandb.define_metric('*', step_metric='trainer/round')
    except Exception:
        pass
    return run


def _log(data: Dict[str, Any], step: Optional[int] = None):
    if wandb is None or wandb.run is None:
        return
    wandb.log(data, step=step)


def _line_series(xs, ys, keys, *, title: str, xname: str = 'round'):
    if wandb is None or wandb.run is None:
        return None
    try:
        return wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title=title, xname=xname)
    except Exception:
        return None


def _update_inner_eval_plot(step: int | None = None):
    chart = _line_series(
        xs=_PLOT_STATE['inner_eval_rounds'],
        ys=[_PLOT_STATE['inner_eval_pre'], _PLOT_STATE['inner_eval_post']],
        keys=['pre_outer_step', 'post_outer_step'],
        title='inner_eval mean_strongreject: pre vs post outer_step',
    )
    if chart is not None:
        _log({'charts/inner_eval_mean_strongreject_pre_vs_post_outer': chart}, step=step)


def _update_mean_shift_norm_plot(step: int | None = None):
    chart = _line_series(
        xs=_PLOT_STATE['mean_shift_norm_rounds'],
        ys=[_PLOT_STATE['mean_shift_norm_inner'], _PLOT_STATE['mean_shift_norm_outer']],
        keys=['inner_step', 'outer_step'],
        title='mean_shift_norm: inner_step vs outer_step',
    )
    if chart is not None:
        _log({'charts/mean_shift_norm_inner_step_vs_outer_step': chart}, step=step)


def _update_mean_shift_dir_sim_plot(step: int | None = None):
    chart = _line_series(
        xs=_PLOT_STATE['mean_shift_dir_sim_rounds'],
        ys=[_PLOT_STATE['mean_shift_dir_sim_inner'], _PLOT_STATE['mean_shift_dir_sim_outer']],
        keys=['inner_step', 'outer_step'],
        title='mean_shift_dir_sim: inner_step vs outer_step',
    )
    if chart is not None:
        _log({'charts/mean_shift_dir_sim_inner_step_vs_outer_step': chart}, step=step)




def _update_intra_prompt_consistency_plot(step: int | None = None):
    chart = _line_series(
        xs=_PLOT_STATE['intra_prompt_consistency_rounds'],
        ys=[_PLOT_STATE['intra_prompt_consistency_inner']],
        keys=['inner_step'],
        title='intra_prompt_consistency: inner_step',
    )
    if chart is not None:
        _log({'charts/intra_prompt_consistency_inner_step': chart}, step=step)


def log_inner_step_history(round_idx: int, step_history: list[dict[str, Any]], *, step: Optional[int] = None):
    if not step_history:
        return
    payload = {'trainer/round': int(round_idx)}
    last = step_history[-1]
    payload['inner_step_history/num_steps'] = int(len(step_history))
    for k in ('objective', 'nuclear', 'mean_shift_avg_norm', 'mean_shift_dir_sim', 'v_self_sim', 'intra_prompt_consistency'):
        if k in last:
            payload[f'inner_step_history/final_{k}'] = float(last[k])
    _log(payload, step=step)

def log_inner_step(round_idx: int, inner_result: Dict[str, Any], *, step: Optional[int] = None):
    mean_shift_norm = inner_result.get('mean_shift_avg_norm', 0.0)
    mean_shift_dir_sim = inner_result.get('mean_shift_dir_sim', 0.0)
    payload = {
        'trainer/round': int(round_idx),
        'inner_step/mean_shift_avg_norm': mean_shift_norm,
        'inner_step/mean_shift_dir_sim': mean_shift_dir_sim,
        'inner_step/v_self_sim': inner_result.get('v_self_sim', 0.0),
        'inner_step/intra_prompt_consistency': inner_result.get('intra_prompt_consistency', 0.0),
        'inner_step/inner_method': str(inner_result.get('inner_method', 'nuclear')),
    }
    _log(payload, step=step)

    _PLOT_STATE['mean_shift_norm_rounds'].append(int(round_idx))
    _PLOT_STATE['mean_shift_norm_inner'].append(float(mean_shift_norm))
    _PLOT_STATE['mean_shift_norm_outer'].append(float('nan'))
    _update_mean_shift_norm_plot(step=step)

    _PLOT_STATE['mean_shift_dir_sim_rounds'].append(int(round_idx))
    _PLOT_STATE['mean_shift_dir_sim_inner'].append(float(mean_shift_dir_sim))
    _PLOT_STATE['mean_shift_dir_sim_outer'].append(float('nan'))
    _update_mean_shift_dir_sim_plot(step=step)

    _PLOT_STATE['intra_prompt_consistency_rounds'].append(int(round_idx))
    _PLOT_STATE['intra_prompt_consistency_inner'].append(float(inner_result.get('intra_prompt_consistency', 0.0)))
    _update_intra_prompt_consistency_plot(step=step)


def log_eval_batch_steer(round_idx: int, eval_result: Dict[str, Any], *, prefix: str, step: Optional[int] = None):
    mean_sr = float(eval_result.get('mean_strongreject', 0.0))
    num_rows = int(len(eval_result.get('rows', [])))
    _log({
        'trainer/round': int(round_idx),
        f'{prefix}/mean_strongreject': mean_sr,
        f'{prefix}/num_rows': num_rows,
    }, step=step)


def log_inner_eval(round_idx: int, *, pre_eval: Dict[str, Any], post_eval: Dict[str, Any], step: Optional[int] = None):
    pre_sr = float(pre_eval.get('mean_strongreject', 0.0))
    post_sr = float(post_eval.get('mean_strongreject', 0.0))
    _log({
        'trainer/round': int(round_idx),
        'inner_eval/pre_outer_step_mean_strongreject': pre_sr,
        'inner_eval/post_outer_step_mean_strongreject': post_sr,
        'inner_eval/delta_mean_strongreject': post_sr - pre_sr,
    }, step=step)

    _PLOT_STATE['inner_eval_rounds'].append(int(round_idx))
    _PLOT_STATE['inner_eval_pre'].append(pre_sr)
    _PLOT_STATE['inner_eval_post'].append(post_sr)
    _update_inner_eval_plot(step=step)


def log_outer_step(round_idx: int, stats: Dict[str, Any], *, step: Optional[int] = None):
    mean_shift_norm = stats.get('mean_shift_avg_norm', stats.get('mean_shift_norm', 0.0))
    mean_shift_dir_sim = stats.get('mean_shift_dir_sim', 0.0)
    harmful_start = stats.get('loss_harmful_start', stats.get('loss_harmful', 0.0))
    harmful_end = stats.get('loss_harmful_end', stats.get('loss_harmful', 0.0))
    payload = {
        'trainer/round': int(round_idx),
        'outer_step/mean_shift_avg_norm': mean_shift_norm,
        'outer_step/mean_shift_dir_sim': mean_shift_dir_sim,
        'outer_step/loss_harmful': stats.get('loss_harmful', 0.0),
        'outer_step/loss_harmful_start': harmful_start,
        'outer_step/loss_harmful_end': harmful_end,
        'outer_step/loss_benign': stats.get('loss_benign', 0.0),
        'outer_step/loss_benign_start': stats.get('loss_benign_start', stats.get('loss_benign', 0.0)),
        'outer_step/loss_benign_end': stats.get('loss_benign_end', stats.get('loss_benign', 0.0)),
        'outer_step/harmful_unweighted': stats.get('harmful_unweighted', 0.0),
        'outer_step/harmful_unweighted_start': stats.get('harmful_unweighted_start', stats.get('harmful_unweighted', 0.0)),
        'outer_step/harmful_unweighted_end': stats.get('harmful_unweighted_end', stats.get('harmful_unweighted', 0.0)),
        'outer_step/benign_unweighted': stats.get('benign_unweighted', 0.0),
        'outer_step/benign_unweighted_start': stats.get('benign_unweighted_start', stats.get('benign_unweighted', 0.0)),
        'outer_step/benign_unweighted_end': stats.get('benign_unweighted_end', stats.get('benign_unweighted', 0.0)),
        'outer_step/grad_norm': stats.get('grad_norm', 0.0),
        'outer_step/num_steers_harmful_including_zero': stats.get('num_steers_harmful_including_zero', stats.get('num_steers_including_zero', 0)),
        'outer_step/num_steers_benign_including_zero': stats.get('num_steers_benign_including_zero', stats.get('num_steers_including_zero', 0)),
    }
    _log(payload, step=step)

    if int(round_idx) in _PLOT_STATE['outer_loss_rounds']:
        idx = _PLOT_STATE['outer_loss_rounds'].index(int(round_idx))
        _PLOT_STATE['outer_loss_harmful_start'][idx] = float(harmful_start)
        _PLOT_STATE['outer_loss_harmful_end'][idx] = float(harmful_end)
    else:
        _PLOT_STATE['outer_loss_rounds'].append(int(round_idx))
        _PLOT_STATE['outer_loss_harmful_start'].append(float(harmful_start))
        _PLOT_STATE['outer_loss_harmful_end'].append(float(harmful_end))
    _update_outer_loss_harmful_plot(step=step)

    if int(round_idx) in _PLOT_STATE['mean_shift_norm_rounds']:
        idx = _PLOT_STATE['mean_shift_norm_rounds'].index(int(round_idx))
        _PLOT_STATE['mean_shift_norm_outer'][idx] = float(mean_shift_norm)
    else:
        _PLOT_STATE['mean_shift_norm_rounds'].append(int(round_idx))
        _PLOT_STATE['mean_shift_norm_inner'].append(float('nan'))
        _PLOT_STATE['mean_shift_norm_outer'].append(float(mean_shift_norm))
    _update_mean_shift_norm_plot(step=step)

    if int(round_idx) in _PLOT_STATE['mean_shift_dir_sim_rounds']:
        idx = _PLOT_STATE['mean_shift_dir_sim_rounds'].index(int(round_idx))
        _PLOT_STATE['mean_shift_dir_sim_outer'][idx] = float(mean_shift_dir_sim)
    else:
        _PLOT_STATE['mean_shift_dir_sim_rounds'].append(int(round_idx))
        _PLOT_STATE['mean_shift_dir_sim_inner'].append(float('nan'))
        _PLOT_STATE['mean_shift_dir_sim_outer'].append(float(mean_shift_dir_sim))
    _update_mean_shift_dir_sim_plot(step=step)



def _update_outer_loss_harmful_plot(step: int | None = None):
    chart = _line_series(
        xs=_PLOT_STATE['outer_loss_rounds'],
        ys=[_PLOT_STATE['outer_loss_harmful_start'], _PLOT_STATE['outer_loss_harmful_end']],
        keys=['pre_outer_optimization', 'post_outer_optimization'],
        title='outer_step loss_harmful: pre vs post optimization',
    )
    if chart is not None:
        _log({'charts/outer_step_loss_harmful_pre_vs_post': chart}, step=step)

def log_validation(step: int, round_idx: int, val: Dict[str, Any]):
    _log({
        'trainer/round': int(round_idx),
        'validation/mean_strongreject': val.get('mean_strongreject', 0.0),
    }, step=step)


def finalize_wandb(final_val: Dict[str, Any] | None = None, bank_size: int | None = None, peft_target_modules: Any = None):
    if wandb is None or wandb.run is None:
        return
    final_val = final_val or {}
    if 'mean_strongreject' in final_val:
        wandb.summary['final_validation_mean_sr'] = final_val.get('mean_strongreject', 0.0)
    elif 'final_validation_mean_sr' in final_val:
        wandb.summary['final_validation_mean_sr'] = final_val.get('final_validation_mean_sr', 0.0)
    if bank_size is not None:
        wandb.summary['final_bank_size'] = bank_size
    if peft_target_modules is not None:
        wandb.summary['peft_target_modules'] = peft_target_modules
    wandb.finish()
