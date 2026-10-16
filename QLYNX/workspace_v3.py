"""QLYNX V3 analyst workspace.

Presentation layer only. Uses the original operational model scores, policy
thresholds, graph aggregates and optional observed edge list. Never invents
relationships or treats model scores as probabilities of misconduct.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk_ui import RISK_COLORS, RISK_NAMES, RISK_PALE, PRIORITY_COLORS, add_triage
from graph_view import (context_for_account, aggregate_graph, parse_edges,
                        linked_graph, render_graph)

TASKS = ("fraud", "aml", "protection")


def H(value):
    return html.escape(str(value), quote=True)


def num(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def fmt_score(v):
    v = num(v, float('nan'))
    if not math.isfinite(v):
        return "—"
    return "<0.001" if 0 <= v < .001 else f"{v:.3f}"


def priority_badge(priority):
    fg, bg = PRIORITY_COLORS.get(str(priority), PRIORITY_COLORS["Routine"])
    return f'<span class="ql3-priority" style="--prio:{fg};--prio-bg:{bg}">{H(priority)}</span>'


def status_badge(alert):
    return '<span class="ql3-alert">Alert</span>' if alert else '<span class="ql3-quiet">Below threshold</span>'


def score_bar(task, score, threshold, *, width="120px"):
    value = min(100., max(0., num(score) * 100))
    mark = min(100., max(0., num(threshold) * 100))
    return (f'<div class="ql3-score-mini"><span class="ql3-score-label" '
            f'style="color:{RISK_COLORS[task] if score >= threshold else "#475569"}">{fmt_score(score)}</span>'
            f'<div class="ql3-track" style="width:{width}">'
            f'<div class="ql3-fill" style="width:{value:.4f}%;background:{RISK_COLORS[task]}"></div>'
            f'<span class="ql3-tick" style="left:{mark:.4f}%"></span>'
            '</div></div>')


def panel_heading(heading, subtitle=""):
    st.markdown(f'<div class="ql3-panel-heading">{H(heading)}</div>'
                + (f'<div class="ql3-muted">{H(subtitle)}</div>' if subtitle else ''),
                unsafe_allow_html=True)


def alert_flags(row, manifest):
    return {t: num(row.get(f"{t}_score", 0)) >= float(manifest["operational_models"][t]["threshold"]) for t in TASKS}


def risk_score_card(task, score, threshold):
    flagged = num(score) >= threshold
    p = 100 * min(max(num(score), 0), 1)
    marker = 100 * threshold
    caption = "Exceeds alert policy" if flagged else "Below alert threshold"
    return (f'<div class="ql3-risk-box" style="--risk:{RISK_COLORS[task]};--risk-bg:{RISK_PALE[task]}">'
            f'<div class="ql3-risk-top"><span>{H(RISK_NAMES[task])}</span>{status_badge(flagged)}</div>'
            f'<div class="ql3-risk-big">{num(score):.4f}</div>'
            f'<div class="ql3-track ql3-wide"><div class="ql3-fill" style="width:{p:.4f}%"></div>'
            f'<span class="ql3-tick" style="left:{marker:.4f}%"></span></div>'
            f'<div class="ql3-risk-foot">{caption} · Threshold {threshold:.3f}</div></div>')


def _queue_page_reset():
    st.session_state.ql3_page_index = 0


def _move_page(delta):
    st.session_state.ql3_page_index = max(0, st.session_state.get('ql3_page_index', 0) + delta)


def queue_view(*, data, manifest, sources, source_name, set_source,
               on_account, audit_path: Path, on_intake):
    st.markdown('<div class="ql3-subtitle">Accounts ranked by investigation priority. '
                'Scores come from the trained models; every case requires human review.</div>',
                unsafe_allow_html=True)
    base = add_triage(data, manifest)
    with st.container(border=True):
        a, b, c, d = st.columns([1.65, 1.25, 1.25, 1.0], gap="small")
        with a:
            chosen_source = st.selectbox('Data source', sources, index=sources.index(source_name),
                                         key='ql3_source', on_change=_queue_page_reset)
        if chosen_source != source_name:
            set_source(chosen_source)
            st.rerun()
        with b:
            query = st.text_input('Account ID', placeholder='e.g. A0715807', key='ql3_search',
                                  on_change=_queue_page_reset)
        with c:
            cat = st.selectbox('Alerted category', ['Any category'] + [RISK_NAMES[t] for t in TASKS],
                               key='ql3_category', on_change=_queue_page_reset)
        with d:
            sort = st.selectbox('Sort by', ['Priority', 'Fraud', 'AML', 'Protection', 'Account ID'],
                                key='ql3_sort', on_change=_queue_page_reset)
        minimum = st.slider('Minimum category score', min_value=0., max_value=1., value=0.,
                            step=.01, key='ql3_score_limit', on_change=_queue_page_reset)
        counts = base.priority.value_counts()
        st.markdown('<div class="ql3-filter-title">Queue priority</div>', unsafe_allow_html=True)
        checks = st.columns(5)
        chosen_priorities = []
        for col, label in zip(checks, PRIORITY_COLORS):
            with col:
                if st.checkbox(f'{label} · {int(counts.get(label, 0)):,}', value=True,
                               key=f'ql3_priority_{label}', on_change=_queue_page_reset):
                    chosen_priorities.append(label)
    view = base.loc[base.priority.isin(chosen_priorities)].copy()
    if query.strip():
        view = view.loc[view.account_id.astype(str).str.contains(query.strip(), case=False, regex=False)]
    if cat != 'Any category':
        category = next(t for t in TASKS if RISK_NAMES[t] == cat)
        view = view.loc[view[f'{category}_alert']]
    view = view.loc[view[[f'{t}_score' for t in TASKS]].max(axis=1) >= minimum]
    if sort == 'Priority':
        view = view.sort_values(['_priority_sort', '_risk_peak'], ascending=[False, False], kind='stable')
    elif sort == 'Account ID':
        view = view.sort_values('account_id', kind='stable')
    else:
        task = {'Fraud':'fraud','AML':'aml','Protection':'protection'}[sort]
        view = view.sort_values(f'{task}_score', ascending=False, kind='stable')
    with st.container(border=True):
        panel_heading('Cases', f'{len(view):,} matching cases · {source_name} · '
                             'Click an account below to open its evidence dossier.')
        if view.empty:
            st.info('No cases match these filters.')
            return
        page_size = 20
        total_pages = max(1, (len(view) + page_size - 1) // page_size)
        current = min(st.session_state.get('ql3_page_index', 0), total_pages - 1)
        st.session_state.ql3_page_index = current
        start = current * page_size
        display = view.iloc[start:start + page_size]
        headers = ['#', 'Account', 'Priority', 'Fraud', 'AML', 'Protection', 'Reviewed']
        trs = []
        tresholds = {t:float(manifest['operational_models'][t]['threshold']) for t in TASKS}
        decided = set()
        if audit_path.exists():
            try:
                for line in audit_path.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        decided.add(str(json.loads(line).get('account_id', '')))
            except (OSError, ValueError):
                pass
        for offset, (_, row) in enumerate(display.iterrows()):
            ident = H(row.account_id)
            values = [f'<td class="ql3-muted">{start + offset + 1}</td>',
                      f'<td class="ql3-id">{ident}</td>',
                      f'<td>{priority_badge(row.priority)}</td>']
            for task in TASKS:
                values.append(f'<td>{score_bar(task,num(row[f"{task}_score"]),tresholds[task])}</td>')
            values.append('<td><span class="ql3-reviewed">Decided</span></td>' if str(row.account_id) in decided
                          else '<td><span class="ql3-muted">Open</span></td>')
            trs.append('<tr>' + ''.join(values) + '</tr>')
        st.markdown('<div class="ql3-table-scroll"><table class="ql3-cases">'
                    '<thead><tr>' + ''.join(f'<th>{H(h)}</th>' for h in headers) + '</tr></thead>'
                    '<tbody>' + ''.join(trs) + '</tbody></table></div>', unsafe_allow_html=True)
        left, center, right = st.columns([2, 1.2, 1])
        with left:
            st.caption(f'{len(view):,} results · page {current + 1} of {total_pages}')
        with center:
            st.button('← Previous', disabled=current == 0, on_click=_move_page, args=(-1,),
                      key='ql3_prev', use_container_width=True)
        with right:
            st.button('Next →', disabled=current >= total_pages - 1,
                      on_click=_move_page, args=(1,), key='ql3_next', use_container_width=True)
        eligible = display.account_id.astype(str).tolist()
        aa, bb = st.columns([3, 1])
        with aa:
            account = st.selectbox('Select case to inspect', eligible,
                                   key=f'ql3_row_choice_{source_name}_{current}_{cat}_{sort}')
        with bb:
            st.write('')
            st.write('')
            if st.button('Open Account 360° →', type='primary', use_container_width=True):
                on_account(account)
                st.rerun()
    st.caption('Bars show uncalibrated classifier scores; vertical ticks show each model’s saved operational threshold. '
               'Review priority is a transparent triage heuristic, not an additional trained risk classifier.')


def _feature_context(row, features):
    if features is None or 'account_id' not in features.columns:
        return row.to_dict()
    matches = features.loc[features.account_id.astype(str) == str(row.account_id)]
    context = row.to_dict()
    if len(matches):
        context.update(matches.iloc[0].to_dict())
    return context


def _contrib_for_account(evidence, account_id):
    if evidence is None or evidence.empty or 'account_id' not in evidence:
        return pd.DataFrame()
    filtered = evidence.loc[evidence.account_id.astype(str) == str(account_id)].copy()
    if 'observed_feature_value' in filtered:
        filtered = filtered.rename(columns={'observed_feature_value':'observed_value'})
    return filtered


def _model_evidence_plot(evidence, task):
    if evidence.empty or 'risk_category' not in evidence:
        return False
    subset = evidence.loc[evidence.risk_category.astype(str).str.lower() == task].copy()
    if subset.empty:
        return False
    subset['contribution_log_odds'] = pd.to_numeric(subset['contribution_log_odds'], errors='coerce')
    subset = subset.dropna(subset=['contribution_log_odds'])
    subset = subset.sort_values('contribution_log_odds', key=lambda x:x.abs(), ascending=False).head(5)
    if subset.empty:
        return False
    values = subset.contribution_log_odds.to_numpy(dtype=float)
    labels = [str(f).replace('_',' ').capitalize() for f in subset.feature]
    fig = go.Figure(go.Bar(y=labels[::-1], x=values[::-1], orientation='h',
                           marker_color=['#d9485b' if x > 0 else '#8b9db4' for x in values[::-1]],
                           customdata=[str(x) for x in subset.get('observed_value',pd.Series(['']*len(subset))).iloc[::-1]],
                           hovertemplate='%{y}<br>Contribution: %{x:+.3f}<br>Observed: %{customdata}<extra></extra>'))
    fig.update_layout(height=260, margin=dict(l=0,r=10,t=12,b=0),
                      xaxis=dict(title='Raw-margin contribution', zeroline=True, showgrid=True),
                      yaxis=dict(title=None, tickfont=dict(size=10)),
                      paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    return True


def _timeline(context):
    periods = [('Days 1–30','first30'),('Days 31–60','middle30'),('Days 61–90','last30')]
    if not any(f'wager_{suffix}' in context or f'deposit_{suffix}' in context for _,suffix in periods):
        st.info('Historical windows are not included in this account export.')
        return
    names = [label for label,_ in periods]
    fig = go.Figure()
    for name, color in [('deposit','#8799b5'),('wager','#263955')]:
        fig.add_trace(go.Bar(name=f'{name.title()}s (CAD)', x=names,
                             y=[num(context.get(f'{name}_{suffix}')) for _,suffix in periods],
                             marker_color=color))
    fig.update_layout(height=300, barmode='group', yaxis_title='CAD',
                      legend=dict(orientation='h',y=-.18),
                      margin=dict(l=12,r=12,t=15,b=45),
                      plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    tx = [context.get(f'tx_{s}', '—') for _,s in periods]
    st.caption('Transactions per window: ' + ' · '.join(f'{n}: {v}' for n,v in zip(names,tx)))


GROUPS = {
    'Profile': {'Home region':'home_region', 'Income band':'income_band', 'Account age (days)':'account_age_days'},
    'Transactions': {'Transactions':'tx_count', 'Approved':'approved_count','Declined':'declined_count',
                     'Deposit count':'deposit_count','Withdrawal count':'withdrawal_count',
                     'Wager count':'wager_count','Payout count':'payout_count',
                     'Deposits (CAD)':'deposit_total','Withdrawals (CAD)':'withdrawal_total',
                     'Wagers (CAD)':'wager_total','Payouts (CAD)':'payout_total',
                     'Maximum transaction':'max_tx_amount','Mean transaction':'avg_tx_amount'},
    'Recent behaviour': {'Deposits last 30 days':'deposit_count_last30',
                         'Mean wager, first 30':'avg_wager_first30','Mean wager, last 30':'avg_wager_last30',
                         'Sessions last 30':'sessions_last30','Devices last 30':'devices_last30'},
    'Sessions & devices': {'Session count':'session_count','Distinct devices':'distinct_devices',
                           'Distinct networks':'distinct_networks', 'Regions':'distinct_regions',
                           'Away-region sessions':'away_region_sessions','Peak sessions/day':'peak_sessions_day',
                           'Peak devices/day':'peak_devices_day','Multi-device days':'multi_device_days',
                           'Region switches/hour':'region_switches_1h'},
    'Payments': {'Payment events':'payment_event_count', 'Tokens':'distinct_payment_tokens',
                 'Payment deposits':'payment_deposits','Payment withdrawals':'payment_withdrawals'},
    'Shared resources': {'Shared device count':'shared_device_count',
                         'Max device account degree':'max_device_account_degree'},
    'Velocity · 6-hour peaks': {'Peak transactions':'peak_transactions_6h',
                                'Peak deposits':'peak_deposits_6h','Peak withdrawals':'peak_withdrawals_6h',
                                'Peak wagers':'peak_wagers_6h','Peak deposit CAD':'peak_deposit_cad_6h',
                                'Peak withdrawal CAD':'peak_withdrawal_cad_6h','Peak wager CAD':'peak_wager_cad_6h',
                                'Deposit/withdraw windows':'deposit_withdrawal_6h_windows',
                                'Loss proxy peaks':'peak_loss_proxy_6h'},
}


def _format_value(value):
    if value is None or pd.isna(value):
        return '—'
    if isinstance(value, (int,float, np.number)):
        return f'{float(value):,.2f}' if abs(float(value)) >= 1000 or abs(float(value)) % 1 != 0 else f'{value:,.0f}'
    return str(value)


def _group_grid(context):
    # Native containers retain responsiveness, including when data are absent.
    items = [(heading,[(label,key,context[key]) for label,key in keys.items() if key in context and pd.notna(context[key])])
             for heading,keys in GROUPS.items()]
    items = [(h,values) for h,values in items if values]
    for begin in range(0,len(items),3):
        columns = st.columns(3, gap='small')
        for col,(heading,values) in zip(columns,items[begin:begin+3]):
            with col:
                with st.container(border=True):
                    panel_heading(heading)
                    st.markdown('<div class="ql3-fact-list">' + ''.join(
                        f'<div class="ql3-fact"><span>{H(label)}</span><strong>{H(_format_value(value))}</strong></div>'
                        for label,_,value in values)+'</div>',unsafe_allow_html=True)


def account360(*, row, data, manifest, source_name, features, evidence, on_queue, on_copilot, on_audit,
               run_neural, explain_model=None):
    if row is None:
        st.info('Select an account from the Investigation Queue.')
        return
    scored = add_triage(pd.DataFrame([row.to_dict()]), manifest).iloc[0]
    ident = str(row.account_id)
    context = _feature_context(row, features)
    contrib = _contrib_for_account(evidence, ident)
    flags = alert_flags(scored, manifest)
    alert_types = [t for t in TASKS if flags[t]]
    display_head, back = st.columns([4.3,2])
    with display_head:
        st.markdown(f'<div class="ql3-acc-title">{H(ident)} {priority_badge(scored.priority)}</div>',
                    unsafe_allow_html=True)
        st.caption(f'Synthetic research account · {source_name} · Operational model scores')
    with back:
        ids = data.account_id.astype(str).tolist()
        chosen = st.selectbox('Account', ids, index=ids.index(ident) if ident in ids else 0,
                              key=f'ql3_account_switch_{ident}', label_visibility='collapsed')
        if chosen != ident:
            st.session_state.active_account = chosen
            st.rerun()
        if st.button('← Queue',use_container_width=True):
            on_queue()
            st.rerun()
    if len(alert_types) >= 2:
        suggestion='Refer for cross-category investigation'
        reason='Multiple operational categories exceed their separate thresholds. Review independent evidence and benign explanations.'
    elif alert_types:
        name = RISK_NAMES[alert_types[0]]
        suggestion=f'Refer for {name.lower()} review'
        reason='The selected operational model exceeds its saved alert threshold. Check observed evidence and alternative explanations.'
    else:
        suggestion='No automatic escalation'
        reason='No operational threshold is exceeded. Further review may still be appropriate when supported by other evidence.'
    score = 100 * (int(scored.alert_count) + num(scored.get('_risk_peak')))
    st.markdown('<div class="ql3-nextstep"><div><span class="ql3-eyebrow">SUGGESTED NEXT STEP</span>'
                f'<strong>{H(suggestion)}</strong><small>{H(reason)}</small></div>'
                f'<div class="ql3-priority-score"><span>Review score*</span><strong>{score:.1f}</strong></div></div>',
                unsafe_allow_html=True)
    st.caption('*Transparent display heuristic (100 × operational alert count + 100 × maximum model score); '
               'not a calibrated combined probability.')
    st.markdown('<div class="ql3-risk-three">'+''.join(
        risk_score_card(t,num(scored.get(f'{t}_score')),float(manifest['operational_models'][t]['threshold']))
        for t in TASKS)+'</div>', unsafe_allow_html=True)
    with st.container(border=True):
        panel_heading('Hybrid neural corroboration', 'Three-seed quantum/classical V5 ensembles use the same frozen prepared account features. '
                      'Experimental outputs do not create operational alerts.')
        saved = st.session_state.get('ql3_neural_cache',{})
        if ident not in saved:
            if st.button('Run six V5 neural checkpoints for this account', key=f'ql3_neural_{ident}'):
                try:
                    with st.spinner('Executing 6 saved V5 model checkpoints…'):
                        neural = run_neural([ident])
                    if neural is not None and not neural.empty:
                        saved = dict(saved)
                        saved[ident] = neural.iloc[0].to_dict()
                        st.session_state['ql3_neural_cache'] = saved
                        st.rerun()
                except Exception as exc:
                    st.error(f'Known-account V5 inference unavailable: {exc}')
            st.caption('Click Run to calculate fresh neural checkpoint predictions; none are fabricated or substituted.')
        else:
            entries = []
            output = saved[ident]
            for t in TASKS:
                q = output.get(f'quantum_v5_{t}_score')
                c = output.get(f'classical_v5_{t}_score')
                entries.append({'Category':RISK_NAMES[t], 'Quantum V5 (3-seed mean)':q,
                                'Strong classical V5 (3-seed mean)':c,
                                'Difference Q−C':num(q)-num(c) if q is not None and c is not None else None})
            st.dataframe(pd.DataFrame(entries),hide_index=True,use_container_width=True)
            st.caption('Uncalibrated neural scores; quantitative comparison only, not model agreement with the operational alert policy.')
    details, case_note, audit = st.tabs(['Evidence', 'Case note', 'Decision & audit'])
    with details:
        if features is not None and explain_model is not None and 'account_id' in features:
            if st.button('Recompute top feature contributions using trained LightGBM weights',
                         key=f'ql3_explain_{ident}'):
                try:
                    with st.spinner('Calculating per-feature model contributions…'):
                        matched = features.loc[features.account_id.astype(str) == ident].head(1)
                        if matched.empty:
                            raise ValueError('This account is not in the feature input table.')
                        _, detail = explain_model(matched, explanations=True)
                        saved_notes = dict(st.session_state.get('ql3_explanations', {}))
                        saved_notes[ident] = detail
                        st.session_state.ql3_explanations = saved_notes
                        st.rerun()
                except Exception as exc:
                    st.error(f'Could not recalculate feature contributions: {exc}')
        if ident in st.session_state.get('ql3_explanations', {}):
            contrib = st.session_state.ql3_explanations[ident]
        with st.container(border=True):
            panel_heading('Why the operational models scored this account',
                          'Top LightGBM feature contributions (raw margin/log-odds); red increases the model score and grey decreases it.')
            panes = st.columns(3, gap='small')
            for col, task in zip(panes,TASKS):
                with col:
                    st.markdown(f'<div class="ql3-explain-label" style="color:{RISK_COLORS[task]}">'
                                f'{H(RISK_NAMES[task])}</div>',unsafe_allow_html=True)
                    if not _model_evidence_plot(contrib,task):
                        st.caption('No saved per-feature explanations for this source. Use Data Intake for fresh LightGBM contributions.')
            st.caption('These feature contributions explain LightGBM only; they are NOT explanations of the quantum VQC.')
        with st.container(border=True):
            panel_heading('90-day activity windows', 'Wagers and deposits in CAD for each 30-day window.')
            _timeline(context)
        if features is not None and 'account_id' in features:
            with st.container(border=True):
                panel_heading('Cohort context', 'This account compared with the median of the supplied feature-view accounts; '
                              'not inferred device neighbours or other account owners.')
                compare_fields = [('Transactions','tx_count'),('Deposit total (CAD)','deposit_total'),
                                  ('Wager total (CAD)','wager_total'),('Sessions','session_count'),
                                  ('Distinct devices','distinct_devices'),('Payment tokens','distinct_payment_tokens')]
                records=[]
                for title, key in compare_fields:
                    if key in context and key in features:
                        sample = pd.to_numeric(features[key],errors='coerce')
                        records.append({'Measure':title,'This account':num(context[key]),
                                        'Available cohort median':sample.median()})
                if records:
                    st.dataframe(pd.DataFrame(records),hide_index=True,use_container_width=True)
        _group_grid(context)
    with case_note:
        with st.container(border=True):
            panel_heading('Investigator case note', 'Generate a grounded note with your local LLM or a deterministic fallback.')
            st.caption(f'Case ID: {ident} · Risk categories exceeding thresholds: '
                       + (', '.join(RISK_NAMES[t] for t in alert_types) if alert_types else 'None'))
            if st.button('Open LLM Investigator Copilot →',type='primary'):
                on_copilot()
                st.rerun()
    with audit:
        with st.container(border=True):
            panel_heading('Human review & audit', 'No account action is taken automatically.')
            if st.button('Open Decisions & Audit →',type='primary'):
                on_audit()
                st.rerun()
    st.caption('Synthetic research demonstration. Model outputs are uncalibrated; quantum execution is simulated; all decisions require human review.')


def relationships(*, row, features):
    st.markdown('<div class="ql3-subtitle">Observed account↔resource graph exploration and explainable shared-resource context.</div>',
                unsafe_allow_html=True)
    if row is None:
        st.info('Select a case in Investigation Queue to inspect relationships.')
        return
    account_id = str(row.account_id)
    context = context_for_account(row,features)
    with st.container(border=True):
        panel_heading('Relationship explorer', f'Account {account_id} · only observed edges are shown as true relationships')
        with st.expander('Import observed account–resource edges (CSV)', expanded=False):
            st.caption('Columns: account_id, entity_type, entity_id. entity_type is device, network, or payment. '
                       'Use sanitized event-derived IDs; do not upload private ownership labels.')
            sample = 'account_id,entity_type,entity_id\nA0715807,device,DEVICE01\nA0715807,network,NET12\nA0108836,device,DEVICE01\n'
            st.download_button('Download CSV example',sample,file_name='relationship_edges_example.csv',mime='text/csv')
            uploaded = st.file_uploader('Observed edge export',type=['csv'],key='ql3_relation_upload')
            if uploaded is not None:
                try:
                    st.session_state.ql3_observed_edges = parse_edges(pd.read_csv(uploaded,dtype=str))
                    st.success(f'{len(st.session_state.ql3_observed_edges):,} validated observed relationships loaded.')
                except Exception as exc:
                    st.session_state.pop('ql3_observed_edges',None)
                    st.error(str(exc))
            if 'ql3_observed_edges' in st.session_state and st.button('Clear edge dataset'):
                st.session_state.pop('ql3_observed_edges',None)
                st.rerun()
        has_edges = 'ql3_observed_edges' in st.session_state
        modes = ['Aggregate evidence (from ZIP)'] + (['Observed resource graph (uploaded edges)'] if has_edges else [])
        mode = st.radio('Graph source',modes,horizontal=True)
        if mode.startswith('Aggregate'):
            graph,count = aggregate_graph(account_id,context)
            if count:
                st.plotly_chart(render_graph(graph,aggregate=True),use_container_width=True)
                st.info('These nodes represent counts and summary measures, NOT individual devices, shared tokens, '
                        'or verified account-to-account links. The source ZIP does not contain explicit relationship edges.')
            else:
                st.info('This account export has no numeric relationship features.')
        else:
            edges = st.session_state.ql3_observed_edges
            graph,limit = linked_graph(account_id,edges)
            if len(graph.nodes) <= 1:
                st.warning('No observed edges for this account in the uploaded export.')
            else:
                st.plotly_chart(render_graph(graph,aggregate=False),use_container_width=True)
                st.caption(f'Observed local graph: {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges. '
                           'Edges come from the uploaded CSV; no ownership identities are inferred.')
        st.markdown('<div class="ql3-legend"><span>● Account</span><span>● Device</span>'
                    '<span>● Network</span><span>● Payment token</span></div>',unsafe_allow_html=True)
    with st.container(border=True):
        panel_heading('Shared-resource evidence', 'Summary statistics from observable account features, not inferred ownership.')
        fields=[('Distinct devices','distinct_devices'),('Shared device count','shared_device_count'),
                ('Device account degree','max_device_account_degree'),('Distinct networks','distinct_networks'),
                ('Payment tokens','distinct_payment_tokens'),('Payment events','payment_event_count')]
        cards = []
        for title,key in fields:
            if key in context and pd.notna(context[key]):
                cards.append(f'<div class="ql3-stat"><small>{H(title)}</small><strong>{H(_format_value(context[key]))}</strong></div>')
        if cards:
            st.markdown('<div class="ql3-stat-grid">'+''.join(cards)+'</div>',unsafe_allow_html=True)
        else:
            st.caption('No shared-resource aggregates exported for this account.')
    st.caption('The optional 50K common-ownership experiment is a separate model from the 1.2M V5 fraud/AML/protection weights. '
               'This explorer does not label shared devices or networks as proven fraud.')


CSS = r'''
/* QLYNX V3 — screenshot-inspired investigation surfaces */
section[data-testid="stSidebar"] div[data-testid="stButton"] button [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] div[data-testid="stButton"] button p {width:100% !important;text-align:left !important;margin:0 !important;}
section[data-testid="stSidebar"] div[data-testid="stButton"] button {padding-left:14px !important;justify-content:flex-start !important;text-align:left !important}
.ql3-subtitle{color:#667895;font-size:.88rem;margin:.1rem 0 1.2rem}
.ql3-panel-heading{font-size:1rem;font-weight:750;letter-spacing:-.015em;color:#17243b;margin-bottom:2px}
.ql3-muted{color:#7b8ba3;font-size:.8rem}
.ql3-filter-title{font-size:.8rem;font-weight:660;margin:5px 0 5px}
.ql3-priority{display:inline-block;border-radius:7px;padding:4px 9px;font-weight:740;
  white-space:nowrap;color:var(--prio);background:var(--prio-bg);border:1px solid color-mix(in srgb,var(--prio) 18%,transparent);font-size:.76rem}
.ql3-table-scroll{overflow-x:auto;width:100%;margin:12px 0 16px}
.ql3-cases{width:100%;border-collapse:collapse;min-width:840px;font-size:.79rem;color:#15213c}
.ql3-cases thead th{font-weight:680;text-align:left;padding:10px 10px;color:#34445e;background:#f8fafd;border-bottom:1px solid #dce4ee;white-space:nowrap}
.ql3-cases tbody tr{border-bottom:1px solid #e8edf5}
.ql3-cases tbody tr:hover{background:#f7faff}
.ql3-cases td{padding:10px 10px;vertical-align:middle}
.ql3-cases .ql3-id{font-weight:760;color:#213658}
.ql3-reviewed{background:#edf4ff;border-radius:7px;color:#28579d;font-size:.72rem;padding:4px 7px}
.ql3-score-mini{display:flex;align-items:center;gap:8px}
.ql3-score-label{display:inline-block;min-width:43px;font-weight:760;font-variant-numeric:tabular-nums}
.ql3-track{position:relative;height:7px;border-radius:20px;background:#e8edf5;overflow:visible;flex-shrink:0}
.ql3-fill{height:100%;border-radius:20px}
.ql3-tick{position:absolute;top:-3px;height:13px;width:2px;border-radius:2px;background:#29364e}
.ql3-acc-title{color:#13213b;font-size:1.85rem;letter-spacing:-.035em;font-weight:800;display:flex;align-items:center;gap:12px}
.ql3-nextstep{border:1px solid #e1e8f2;border-radius:13px;padding:17px 22px;margin:12px 0 7px;display:flex;align-items:center;justify-content:space-between;gap:20px;background:#fff}
.ql3-nextstep strong,.ql3-nextstep small{display:block}
.ql3-nextstep strong{font-size:1.03rem;color:#18263e;margin:5px 0}
.ql3-nextstep small{color:#687b98;font-size:.8rem;max-width:690px;line-height:1.5}
.ql3-eyebrow{font-size:.67rem;font-weight:800;letter-spacing:.1em;color:#7990b0}
.ql3-priority-score{text-align:right;min-width:110px}.ql3-priority-score span{display:block;font-size:.73rem;color:#7a8aa3}
.ql3-priority-score strong{font-size:1.65rem;color:#1d2c4c}
.ql3-risk-three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:16px 0}
.ql3-risk-box{border:1px solid color-mix(in srgb,var(--risk) 25%,white);border-top:3px solid var(--risk);border-radius:12px;background:var(--risk-bg);padding:17px 18px}
.ql3-risk-top{font-size:.83rem;color:#3d506c;font-weight:710;display:flex;justify-content:space-between;align-items:center;gap:6px}
.ql3-alert{background:var(--risk);color:white;border-radius:6px;padding:3px 7px;font-size:.66rem;font-weight:760}
.ql3-quiet{background:#edf1f6;color:#5e6e82;border-radius:6px;padding:3px 6px;font-size:.63rem}
.ql3-risk-big{color:var(--risk);font-size:1.92rem;font-weight:790;letter-spacing:-.035em;margin:14px 0}
.ql3-wide{width:100%;background:color-mix(in srgb,var(--risk) 11%,white)}.ql3-wide .ql3-fill{background:var(--risk)}
.ql3-risk-foot{font-size:.77rem;color:#637890;margin-top:12px}
.ql3-explain-label{font-size:.85rem;font-weight:770;margin:9px 0 0}
.ql3-fact-list{margin:7px 0;min-height:90px}.ql3-fact{display:flex;align-items:baseline;justify-content:space-between;gap:9px;padding:5px 0;border-bottom:1px solid #f0f3f8;font-size:.75rem}
.ql3-fact span{color:#677b95;word-break:break-word}.ql3-fact strong{color:#182741;text-align:right;font-size:.78rem;font-weight:680}
.ql3-stat-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:15px 0}
.ql3-stat{border:1px solid #e3eaf4;border-radius:11px;padding:16px;background:#fafcff}
.ql3-stat small,.ql3-stat strong{display:block}.ql3-stat small{color:#72839b;font-size:.78rem}.ql3-stat strong{font-size:1.32rem;color:#253757;margin-top:8px}
.ql3-legend{display:flex;gap:20px;flex-wrap:wrap;font-size:.78rem;color:#60738e;margin:4px 0 10px}
@media(max-width:1000px){.ql3-risk-three{grid-template-columns:1fr}.ql3-stat-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.ql3-nextstep{flex-direction:column;align-items:flex-start}.ql3-priority-score{text-align:left}}
'''
