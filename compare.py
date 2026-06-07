import pandas as pd
from pathlib import Path

polygons = ['polygon1', 'polygon3', 'polygon4', 'polygon5']
run_tag = 't128m_o50pct_r1m'

print(f"{'polygon':<10}  {'orig':>5}  {'rot':>5}  {'change':>7}  {'theta':>9}  {'orig_vf':>8}  {'rot_vf':>8}")
print('-' * 70)
for poly in polygons:
    orig_p = Path(f'outputs/{poly}/{run_tag}/manifest.csv')
    rot_p  = Path(f'outputs/{poly}/{run_tag}_rot/manifest.csv')
    df_o = pd.read_csv(orig_p)
    df_r = pd.read_csv(rot_p)
    n_o, n_r = len(df_o), len(df_r)
    change = f'{(n_r - n_o) / n_o * 100:+.0f}%'
    theta = f"{df_r['theta_deg'].iloc[0]:.1f} deg" if 'theta_deg' in df_r.columns else 'N/A'
    v_o = df_o.valid_frac.mean()
    v_r = df_r.valid_frac.mean()
    print(f'{poly:<10}  {n_o:>5}  {n_r:>5}  {change:>7}  {theta:>9}  {v_o:>8.3f}  {v_r:>8.3f}')

print()
print('Labeled pixel totals (all polygons combined):')
for label, suffix in [('original', ''), ('rotated', '_rot')]:
    px = {}
    for poly in polygons:
        p = Path(f'outputs/{poly}/{run_tag}{suffix}/manifest.csv')
        df = pd.read_csv(p)
        for col in [c for c in df.columns if c.endswith('_px') and c != 'background_px']:
            px[col] = px.get(col, 0) + int(df[col].sum())
    total = sum(px.values())
    parts = '  '.join(f'{k}={v:,}' for k, v in px.items())
    print(f'  {label:<10} total labeled px={total:,}  |  {parts}')
