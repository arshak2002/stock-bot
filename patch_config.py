import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Update scoring
cfg["scoring"]["min_score_orb_window"] = 7
cfg["scoring"]["min_score_morning"] = 7
cfg["scoring"]["min_score_afternoon"] = 8
cfg["scoring"]["grade_a_plus"] = 9
cfg["scoring"]["grade_a"] = 7
cfg["scoring"]["vwap_proximity_pct"] = 1.5
cfg["scoring"]["cpr_proximity_pct"] = 0.75

# Update risk_management
cfg["risk_management"]["sl_atr_multiplier"] = 1.2
cfg["risk_management"]["target_atr_multiplier"] = 2.0
cfg["risk_management"]["true_edge_min_pct"] = 0.10
cfg["risk_management"]["true_edge_min_midcap"] = 0.15
cfg["risk_management"]["trailing_breakeven_pct"] = 0.5

# Update indicators
cfg["indicators"]["rsi_period"] = 14
cfg["indicators"]["ema_fast"] = 9
cfg["indicators"]["ema_slow"] = 21
cfg["indicators"]["supertrend_period"] = 10
cfg["indicators"]["supertrend_multiplier"] = 3.0
cfg["indicators"]["adx_period"] = 14

import sys
class Dumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, *args, **kwargs):
        return super().increase_indent(flow=flow, indentless=False)

with open("config.yaml", "w") as f:
    yaml.dump(cfg, f, sort_keys=False, Dumper=Dumper, default_flow_style=False)

print("Config patched.")
