# -*- coding: utf-8 -*-
"""
Script 94 -- Two robustness tables quoted with the evaluation re-run of 26-27 November 2026.
Run from the repository root after Script 15 (saves ablation_market_predictions.parquet) and Script 69 (seeds 42-46).
Outputs: Model_Output/table_stability_skill.csv, Model_Output/table_observed_only_scoring.csv
"""
import pandas as pd


def seed_spread():
    """Spread of the skill of a cell (M0, M6) over the five seeds of Script 69 (new validation rule)."""
    ab = pd.read_csv('Model_Output/ablation_raw_results.csv')
    nv = ab[ab.variant == 'B1_Naive'].groupby(['crop', 'horizon_weeks']).MAE.mean()
    rows = []
    for seed in range(42, 47):
        p = pd.read_parquet(f'Model_Output/_stability_runs/new_{seed}.parquet', columns=['variant', 'crop', 'fold', 'horizon_weeks', 'y_true', 'y_pred'])
        p['variant'] = p.variant.astype(str); p['crop'] = p.crop.astype(str)
        p['ae'] = (p.y_true - p.y_pred).abs()
        m = p.groupby(['variant', 'crop', 'horizon_weeks', 'fold']).ae.mean().groupby(level=[0, 1, 2]).mean()
        rows += [(v, c, h, seed, (1 - e / nv.loc[(c, h)]) * 100) for (v, c, h), e in m.items()]
    s = pd.DataFrame(rows, columns=['variant', 'crop', 'horizon_weeks', 'seed', 'skill'])
    g = s.groupby(['variant', 'crop', 'horizon_weeks']).skill.agg(['mean', 'std', 'min', 'max'])
    g['range'] = g['max'] - g['min']
    g.reset_index().round(3).to_csv('Model_Output/table_stability_skill.csv', index=False)


def observed_only():
    """Skill on all scored rows against rows whose origin-week and target-week prices were both observed (models unchanged)."""
    p=pd.read_parquet('Model_Output/ablation_market_predictions.parquet',columns=['variant','crop','fold','horizon_weeks','market_id','week_start','y_true','y_pred'])
    p=p[p.variant.isin(['M0','M6'])].copy(); p['variant']=p.variant.astype(str); p['crop']=p.crop.astype(str); p['market_id']=p.market_id.astype(int); p['week_start']=pd.to_datetime(p.week_start)
    d=pd.read_csv('data/agmarknet_weekly/top_weekly_panel.csv',usecols=['crop','market_id','week_start','modal_price_weighted','imputed'],parse_dates=['week_start']).dropna(subset=['market_id']); d['market_id']=d.market_id.astype(int)
    d['obs']=(d.imputed==0)&d.modal_price_weighted.notna()
    p=p.merge(d[['crop','market_id','week_start','modal_price_weighted','obs']],on=['crop','market_id','week_start'],how='left').rename(columns={'modal_price_weighted':'orig','obs':'obs_o'})
    p['tw']=p.week_start+pd.to_timedelta(p.horizon_weeks*7,unit='D')
    p=p.merge(d[['crop','market_id','week_start','obs']].rename(columns={'week_start':'tw','obs':'obs_t'}),on=['crop','market_id','tw'],how='left')
    p=p.dropna(subset=['orig']); p['ae']=(p.y_true-p.y_pred).abs(); p['nae']=(p.y_true-p.orig).abs()
    def skill(x):
        g=x.groupby(['variant','crop','horizon_weeks','fold'])[['ae','nae']].mean().groupby(level=[0,1,2]).mean()
        return (1-g.ae/g.nae)*100
    a=skill(p); r=p[p.obs_o.eq(True)&p.obs_t.eq(True)]; b=skill(r); diff=b-a
    n=p.groupby(['variant','crop','horizon_weeks']).size(); no=r.groupby(['variant','crop','horizon_weeks']).size()
    pd.DataFrame({'skill_all':a,'skill_observed_only':b,'difference':diff,'rows_removed_pct':(1-no/n)*100}).reset_index().round(3).to_csv('Model_Output/table_observed_only_scoring.csv',index=False)


if __name__ == '__main__':
    seed_spread()
    observed_only()
    print('Script 94 complete')
