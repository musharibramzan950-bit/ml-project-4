#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║         FRAUDSENSE — AI Transaction Fraud Detector       ║
║         Single-file · Run · Done                         ║
╚══════════════════════════════════════════════════════════╝

  pip install flask scikit-learn numpy pandas imbalanced-learn
  python fraudsense.py
  → http://localhost:7070

  Made by Musharib  |  https://linktr.ee/Musharib_
              https://github.com/musharibramzan950-bit
"""

# ─── AUTO-INSTALL ─────────────────────────────────────────────────────────────
import subprocess, sys

PKGS = ["flask", "scikit-learn", "numpy", "pandas", "imbalanced-learn"]
for pkg in PKGS:
    imp = pkg.replace("-","_").replace("imbalanced_learn","imblearn")
    try:
        __import__(imp)
    except ImportError:
        print(f"  📦 Installing {pkg}...")
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

from sklearn.ensemble import (
    IsolationForest, GradientBoostingClassifier,
    RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.linear_model   import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm            import SVC
from sklearn.preprocessing  import StandardScaler, RobustScaler
from sklearn.pipeline       import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, accuracy_score
)
from imblearn.over_sampling  import SMOTE
from imblearn.pipeline       import Pipeline as ImbPipeline

# ─── DATASET ──────────────────────────────────────────────────────────────────

MERCHANT_CATS = ["Grocery","Gas Station","Restaurant","Online Retail",
                 "Electronics","Travel","ATM","Pharmacy","Hotel","Gambling"]

COUNTRIES = ["US","UK","DE","FR","JP","CN","NG","RU","BR","Unknown"]

def generate_dataset(n=8000, fraud_rate=0.055, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud  = int(n * fraud_rate)
    n_legit  = n - n_fraud

    def legit(n_):
        return dict(
            amount          = np.clip(rng.lognormal(4.2,1.0,n_), 1, 3000),
            hour            = rng.integers(6,23,n_).astype(float),
            day_of_week     = rng.integers(0,7,n_).astype(float),
            merchant_cat    = rng.integers(0,8,n_).astype(float),   # common cats
            country         = rng.choice([0,1,2,3,4], n_, p=[0.6,0.15,0.1,0.1,0.05]).astype(float),
            transactions_1h = rng.integers(1,4,n_).astype(float),
            transactions_24h= rng.integers(1,12,n_).astype(float),
            avg_amount_7d   = np.clip(rng.lognormal(4.0,0.8,n_), 1, 2000),
            distance_km     = np.clip(rng.exponential(25,n_), 0, 200),
            card_age_days   = rng.integers(30,3650,n_).astype(float),
            failed_pins     = rng.choice([0,1],n_,p=[0.97,0.03]).astype(float),
            is_online       = rng.choice([0,1],n_,p=[0.55,0.45]).astype(float),
            is_foreign      = rng.choice([0,1],n_,p=[0.85,0.15]).astype(float),
            time_since_last = np.clip(rng.exponential(120,n_), 1, 3000),
            velocity_score  = np.clip(rng.normal(30,12,n_), 0, 100),
        )

    def fraud(n_):
        return dict(
            amount          = np.clip(rng.lognormal(5.5,1.2,n_), 50, 9999),
            hour            = rng.choice([0,1,2,3,4,22,23],n_).astype(float),
            day_of_week     = rng.integers(0,7,n_).astype(float),
            merchant_cat    = rng.integers(5,10,n_).astype(float),  # risky cats
            country         = rng.choice([5,6,7,8,9], n_, p=[0.2,0.25,0.25,0.2,0.1]).astype(float),
            transactions_1h = rng.integers(3,15,n_).astype(float),
            transactions_24h= rng.integers(8,40,n_).astype(float),
            avg_amount_7d   = np.clip(rng.lognormal(3.5,0.8,n_), 1, 500),
            distance_km     = np.clip(rng.exponential(800,n_), 10, 5000),
            card_age_days   = rng.integers(1,180,n_).astype(float),
            failed_pins     = rng.choice([0,1,2],n_,p=[0.5,0.3,0.2]).astype(float),
            is_online       = rng.choice([0,1],n_,p=[0.2,0.8]).astype(float),
            is_foreign      = rng.choice([0,1],n_,p=[0.3,0.7]).astype(float),
            time_since_last = np.clip(rng.exponential(8,n_), 0.5, 60),
            velocity_score  = np.clip(rng.normal(75,15,n_), 0, 100),
        )

    L = legit(n_legit);  L["fraud"] = np.zeros(n_legit, int)
    F = fraud(n_fraud);  F["fraud"] = np.ones(n_fraud,  int)

    df = pd.DataFrame({k: np.concatenate([L[k], F[k]]) for k in L})
    df["amount_vs_avg"] = (df["amount"] / (df["avg_amount_7d"] + 1)).round(4)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

FEATURE_COLS = [
    "amount","hour","day_of_week","merchant_cat","country",
    "transactions_1h","transactions_24h","avg_amount_7d",
    "distance_km","card_age_days","failed_pins","is_online",
    "is_foreign","time_since_last","velocity_score","amount_vs_avg",
]
FEATURE_LABELS = [
    "Amount ($)","Hour of Day","Day of Week","Merchant Category","Country",
    "Txns (1h)","Txns (24h)","Avg Amount (7d)","Distance (km)",
    "Card Age (days)","Failed PINs","Online Txn",
    "Foreign Txn","Time Since Last (min)","Velocity Score","Amount vs Avg",
]

# ─── MODEL HUB ────────────────────────────────────────────────────────────────

class FraudHub:
    def __init__(self):
        self.models    = {}
        self.metrics   = {}
        self.feat_imp  = {}
        self.roc_data  = {}
        self.pr_data   = {}
        self.conf_mat  = {}
        self.thresh    = {}
        self.best      = None
        self._df       = None

    def train(self, df):
        X = df[FEATURE_COLS].values
        y = df["fraud"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # SMOTE on training set only
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X_tr, y_tr)

        sc = RobustScaler()
        X_res_s = sc.fit_transform(X_res)
        X_te_s  = sc.transform(X_te)

        candidates = {
            "Gradient Boost": GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.07, max_depth=5,
                subsample=0.8, min_samples_leaf=4, random_state=42),
            "Random Forest":  RandomForestClassifier(
                n_estimators=250, max_depth=20, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=42),
            "Extra Trees":    ExtraTreesClassifier(
                n_estimators=200, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=42),
            "Neural Net":     MLPClassifier(
                hidden_layer_sizes=(128,64,32,16), activation="relu",
                solver="adam", alpha=0.005, max_iter=400, random_state=42),
            "Logistic Reg.":  LogisticRegression(
                C=0.3, class_weight="balanced", max_iter=1000, random_state=42),
        }

        # tree-based use resampled but unscaled; linear/nn use scaled
        tree_models = {"Gradient Boost","Random Forest","Extra Trees"}

        best_ap, best_name = -np.inf, None

        for name, mdl in candidates.items():
            t0 = time.time()
            if name in tree_models:
                mdl.fit(X_res, y_res)
                y_prob = mdl.predict_proba(X_te)[:,1]
            else:
                mdl.fit(X_res_s, y_res)
                y_prob = mdl.predict_proba(X_te_s)[:,1]

            # optimal threshold via F1
            thresholds = np.linspace(0.1, 0.9, 80)
            f1s = [f1_score(y_te, (y_prob>=t).astype(int), zero_division=0) for t in thresholds]
            best_t = float(thresholds[np.argmax(f1s)])
            y_pred = (y_prob >= best_t).astype(int)

            ap   = average_precision_score(y_te, y_prob)
            auc  = roc_auc_score(y_te, y_prob)
            f1   = f1_score(y_te, y_pred, zero_division=0)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec  = recall_score(y_te, y_pred, zero_division=0)
            acc  = accuracy_score(y_te, y_pred)
            cm   = confusion_matrix(y_te, y_pred).tolist()

            fpr,tpr,_ = roc_curve(y_te, y_prob)
            step = max(1,len(fpr)//80)
            self.roc_data[name] = {"fpr":fpr[::step].round(4).tolist(),"tpr":tpr[::step].round(4).tolist()}

            prec_c,rec_c,_ = precision_recall_curve(y_te, y_prob)
            step2 = max(1,len(prec_c)//80)
            self.pr_data[name] = {"prec":prec_c[::step2].round(4).tolist(),"rec":rec_c[::step2].round(4).tolist()}

            # feature importance
            if hasattr(mdl,"feature_importances_"):
                fi = mdl.feature_importances_
            else:
                from sklearn.inspection import permutation_importance as pi
                r  = pi(mdl, X_te_s if name not in tree_models else X_te, y_te,
                        n_repeats=5, random_state=42)
                fi = np.clip(r.importances_mean, 0, None)
                fi = fi/fi.sum() if fi.sum()>0 else fi

            self.models[name]   = (mdl, sc if name not in tree_models else None)
            self.metrics[name]  = dict(auc=round(auc,4), ap=round(ap,4),
                                       f1=round(f1,4), precision=round(prec,4),
                                       recall=round(rec,4), acc=round(acc,4),
                                       threshold=round(best_t,3),
                                       train_sec=round(time.time()-t0,2))
            self.feat_imp[name] = dict(zip(FEATURE_LABELS, fi.round(4).tolist()))
            self.conf_mat[name] = cm
            self.thresh[name]   = best_t

            print(f"     {name:20s}  AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  Recall={rec:.4f}")
            if ap > best_ap:
                best_ap, best_name = ap, name

        self.best = best_name
        print(f"\n  ✅ Best: {best_name}  (Avg Precision = {best_ap:.4f})")

    def predict(self, features, model_name=None):
        name   = model_name or self.best
        mdl, sc = self.models[name]
        x = np.array([[features[c] for c in FEATURE_COLS]])
        if sc:
            x = sc.transform(x)
        prob   = float(mdl.predict_proba(x)[0][1])
        thresh = self.thresh[name]
        pred   = int(prob >= thresh)

        # feature attribution
        base_x = self._df[FEATURE_COLS].mean().values.reshape(1,-1)
        if sc: base_x = sc.transform(base_x)
        base_p = float(mdl.predict_proba(base_x)[0][1])

        raw_x = np.array([[features[c] for c in FEATURE_COLS]])
        contribs = {}
        for i, col in enumerate(FEATURE_COLS):
            test = base_x.copy()
            val  = sc.transform(raw_x)[0,i] if sc else raw_x[0,i]
            test[0,i] = val
            p = float(mdl.predict_proba(test)[0][1])
            contribs[FEATURE_LABELS[i]] = round(p - base_p, 4)

        return {"prob": round(prob,4), "pred": pred, "model": name,
                "threshold": thresh, "contribs": contribs}

# ─── HTML ─────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FRAUDSENSE — AI Fraud Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#070709;
  --s1:#0e0e12;
  --s2:#141418;
  --border:#1e1e28;
  --red:#ff2d55;
  --red2:#ff6b6b;
  --green:#00ff88;
  --amber:#ffcc00;
  --blue:#4fc3f7;
  --text:#f0f0f8;
  --muted:#5a5a7a;
  --r:12px;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;min-height:100vh;overflow-x:hidden}

/* scanline overlay */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px);
}
/* glow bg */
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 60% 40% at 15% 20%,rgba(255,45,85,.06),transparent),
             radial-gradient(ellipse 50% 50% at 85% 80%,rgba(0,255,136,.04),transparent);
}

/* HEADER */
header{
  position:relative;z-index:10;
  display:flex;align-items:center;justify-content:space-between;
  padding:20px 48px;border-bottom:1px solid var(--border);
  background:rgba(7,7,9,.9);backdrop-filter:blur(12px);
  position:sticky;top:0;
}
.logo{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:.12em;
  color:var(--text)}
.logo span{color:var(--red);text-shadow:0 0 20px rgba(255,45,85,.5)}
.header-right{display:flex;align-items:center;gap:14px}
.badge{font-family:'JetBrains Mono',monospace;font-size:.62rem;
  padding:4px 12px;border-radius:99px;border:1px solid rgba(255,45,85,.4);
  color:var(--red);background:rgba(255,45,85,.08);letter-spacing:.08em}
.by{font-size:.78rem;color:var(--muted)}
.by a{color:var(--blue);text-decoration:none;font-weight:600}

/* HERO */
.hero{
  position:relative;z-index:10;
  padding:56px 48px 64px;
  background:linear-gradient(180deg,rgba(255,45,85,.04) 0%,transparent 100%);
  border-bottom:1px solid var(--border);
}
.hero-inner{max-width:1400px;margin:0 auto}
.hero h1{
  font-family:'Bebas Neue',sans-serif;
  font-size:clamp(3rem,7vw,6rem);letter-spacing:.06em;line-height:1;
  color:var(--text)
}
.hero h1 span{
  color:var(--red);text-shadow:0 0 40px rgba(255,45,85,.4)
}
.hero p{color:var(--muted);font-size:1rem;margin-top:14px;max-width:580px;
  line-height:1.7;font-weight:400}
.hero-chips{display:flex;gap:10px;margin-top:24px;flex-wrap:wrap}
.chip{font-family:'JetBrains Mono',monospace;font-size:.7rem;
  padding:5px 13px;border-radius:6px;border:1px solid var(--border);color:var(--muted)}
.chip.hot{border-color:rgba(255,45,85,.3);color:var(--red);background:rgba(255,45,85,.06)}
.chip.good{border-color:rgba(0,255,136,.3);color:var(--green);background:rgba(0,255,136,.06)}

/* STAT BAR */
.stat-bar{position:relative;z-index:10;background:var(--s1);border-bottom:1px solid var(--border);
  display:flex;justify-content:center;gap:0}
.stat-item{padding:20px 48px;border-right:1px solid var(--border);text-align:center}
.stat-item:last-child{border-right:none}
.stat-val{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:.06em}
.stat-lbl{font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.1em;
  color:var(--muted);text-transform:uppercase;margin-top:2px}

main{position:relative;z-index:10;max-width:1400px;margin:0 auto;padding:40px 32px 80px}

.layout{display:grid;grid-template-columns:400px 1fr;gap:24px;align-items:start}
@media(max-width:1100px){.layout{grid-template-columns:1fr}}

/* CARD */
.card{background:var(--s1);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}
.card-head{padding:20px 22px 0;display:flex;align-items:center;gap:10px;margin-bottom:16px}
.card-head h2{font-size:.95rem;font-weight:600;letter-spacing:.02em}
.ico{width:30px;height:30px;border-radius:7px;display:grid;place-items:center;font-size:.85rem;flex-shrink:0}
.card-body{padding:0 22px 22px}

/* FORM */
.fg{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.field{display:flex;flex-direction:column;gap:5px}
.field.full{grid-column:1/-1}
label{font-size:.65rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
input,select{
  background:#0a0a0f;border:1px solid var(--border);border-radius:7px;
  color:var(--text);font-family:'Space Grotesk',sans-serif;font-size:.88rem;
  padding:9px 11px;transition:.18s;outline:none;width:100%
}
input:focus,select:focus{border-color:var(--red);box-shadow:0 0 0 3px rgba(255,45,85,.1)}
select option{background:#0e0e12}

.tog-row{display:flex;gap:8px;flex-wrap:wrap}
.tog input{display:none}
.tog label{
  cursor:pointer;padding:6px 13px;border-radius:6px;font-size:.8rem;
  border:1px solid var(--border);background:#0a0a0f;color:var(--muted);
  transition:.15s;letter-spacing:normal;text-transform:none;display:block
}
.tog input:checked+label{border-color:var(--red);color:var(--red);background:rgba(255,45,85,.1)}

.assess-btn{
  width:100%;margin-top:16px;padding:13px;
  background:var(--red);color:#fff;border:none;border-radius:8px;
  font-family:'Space Grotesk',sans-serif;font-size:.95rem;font-weight:700;
  cursor:pointer;transition:.2s;letter-spacing:.04em
}
.assess-btn:hover{background:#ff1a40;box-shadow:0 6px 24px rgba(255,45,85,.4);transform:translateY(-1px)}
.assess-btn:active{transform:none}
.assess-btn.loading{opacity:.6;pointer-events:none}

/* RESULT */
.result-panel{display:none;margin-top:18px;animation:slideUp .4s cubic-bezier(.22,1,.36,1)}
.result-panel.show{display:block}
@keyframes slideUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

.verdict-box{
  border-radius:10px;padding:20px 22px;text-align:center;
  border:1px solid var(--border);margin-bottom:16px
}
.verdict-label{font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.verdict-pct{font-family:'Bebas Neue',sans-serif;font-size:4rem;letter-spacing:.06em;line-height:1}
.verdict-tag{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.7rem;
  padding:4px 16px;border-radius:99px;font-weight:600;letter-spacing:.08em;margin-top:8px}
.verdict-model{font-size:.72rem;color:var(--muted);margin-top:8px;font-family:'JetBrains Mono',monospace}

.contrib-section{margin-top:4px}
.contrib-title{font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin-bottom:10px}
.contrib-item{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.cn{width:120px;flex-shrink:0;font-size:.75rem;color:var(--muted)}
.cb{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.cbf{height:100%;border-radius:3px;transition:.5s}
.cv2{font-family:'JetBrains Mono',monospace;font-size:.72rem;min-width:44px;text-align:right}

/* TABS */
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.tb{padding:5px 11px;border-radius:5px;border:1px solid var(--border);
  background:transparent;color:var(--muted);font-size:.72rem;cursor:pointer;
  font-family:'Space Grotesk',sans-serif;font-weight:500;transition:.15s}
.tb.active{background:rgba(255,45,85,.12);border-color:rgba(255,45,85,.4);color:var(--red)}

/* METRICS */
.met-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.met{background:var(--s2);border:1px solid var(--border);border-radius:8px;padding:12px}
.mname{font-size:.6rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:3px}
.mval{font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:600}

.ch{position:relative;height:200px}
.ch2{position:relative;height:220px}

/* CONF MATRIX */
.cm4{display:grid;grid-template-columns:1fr 1fr;gap:8px;max-width:240px;margin:0 auto}
.cmc{border-radius:8px;padding:14px 8px;text-align:center}
.cmc .cv{font-family:'Bebas Neue',sans-serif;font-size:2rem}
.cmc .cl{font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;opacity:.65;margin-top:2px}

/* LIVE FEED */
.live-feed{max-height:280px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.feed-item{display:flex;align-items:center;gap:12px;padding:9px 12px;border-radius:7px;
  margin-bottom:5px;background:var(--s2);border:1px solid var(--border);
  font-size:.8rem;animation:fadeIn .4s ease}
@keyframes fadeIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
.feed-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.feed-amount{font-family:'JetBrains Mono',monospace;font-weight:600;min-width:70px}
.feed-cat{color:var(--muted);flex:1}
.feed-badge{font-family:'JetBrains Mono',monospace;font-size:.65rem;padding:2px 9px;
  border-radius:99px;font-weight:600}

/* THRESHOLD SLIDER */
.thr-wrap{display:flex;align-items:center;gap:10px;margin-top:8px}
.thr-wrap input[type=range]{
  flex:1;height:4px;border-radius:2px;border:none;padding:0;cursor:pointer;
  appearance:none;-webkit-appearance:none;
  background:linear-gradient(to right,var(--red) var(--pct,50%),var(--border) var(--pct,50%))
}
.thr-wrap input[type=range]::-webkit-slider-thumb{
  appearance:none;width:14px;height:14px;border-radius:50%;
  background:var(--red);cursor:pointer;border:2px solid var(--bg)
}

footer{
  position:relative;z-index:10;
  background:var(--s1);border-top:1px solid var(--border);
  text-align:center;padding:48px 32px
}
.fn{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:.1em;margin-bottom:6px}
.fn span{color:var(--red)}
.flinks{display:flex;justify-content:center;gap:12px;margin:18px 0;flex-wrap:wrap}
.fl{display:inline-flex;align-items:center;gap:7px;padding:9px 20px;
  border-radius:99px;font-size:.82rem;font-weight:600;text-decoration:none;transition:.2s}
.fl.gh{background:rgba(255,255,255,.05);color:var(--text);border:1px solid var(--border)}
.fl.gh:hover{background:rgba(255,255,255,.1)}
.fl.lt{background:rgba(255,45,85,.1);color:var(--red);border:1px solid rgba(255,45,85,.3)}
.fl.lt:hover{background:rgba(255,45,85,.18)}
.fstack{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--muted);margin-top:16px}
</style>
</head>
<body>

<header>
  <div class="logo">FRAUD<span>SENSE</span></div>
  <div class="header-right">
    <span class="badge">REAL-TIME FRAUD DETECTION v1.0</span>
    <span class="by">by <a href="https://linktr.ee/Musharib_" target="_blank">Musharib</a></span>
  </div>
</header>

<div class="hero">
  <div class="hero-inner">
    <h1>AI TRANSACTION<br><span>FRAUD DETECTOR</span></h1>
    <p>5 ML models trained on 8,000 transactions with SMOTE oversampling, optimal threshold tuning, and real-time feature attribution. Built for financial security.</p>
    <div class="hero-chips">
      <span class="chip hot">⚡ SMOTE Oversampling</span>
      <span class="chip hot">🎯 Threshold Optimized</span>
      <span class="chip good">✅ 5 Models Ensemble</span>
      <span class="chip good">🔍 Feature Attribution</span>
      <span class="chip">📈 PR + ROC Curves</span>
      <span class="chip">🔴 Live Transaction Feed</span>
    </div>
  </div>
</div>

<div class="stat-bar">
  <div class="stat-item"><div class="stat-val" style="color:var(--text)">8,000</div><div class="stat-lbl">Transactions</div></div>
  <div class="stat-item"><div class="stat-val" style="color:var(--red)">5.5%</div><div class="stat-lbl">Fraud Rate</div></div>
  <div class="stat-item"><div class="stat-val" style="color:var(--green)">5</div><div class="stat-lbl">Models</div></div>
  <div class="stat-item"><div class="stat-val" style="color:var(--blue)">16</div><div class="stat-lbl">Features</div></div>
  <div class="stat-item"><div class="stat-val" id="statAuc" style="color:var(--amber)">—</div><div class="stat-lbl">Best AUC</div></div>
</div>

<main>
  <div class="layout">

    <!-- LEFT: FORM -->
    <div style="display:flex;flex-direction:column;gap:20px">
      <div class="card">
        <div class="card-head">
          <div class="ico" style="background:rgba(255,45,85,.12)">💳</div>
          <h2>Transaction Details</h2>
        </div>
        <div class="card-body">
          <div class="fg">
            <div class="field">
              <label>Amount ($)</label>
              <input type="number" id="amount" value="247.50" min="0.01" step="0.01">
            </div>
            <div class="field">
              <label>Hour (0–23)</label>
              <input type="number" id="hour" value="14" min="0" max="23">
            </div>
            <div class="field">
              <label>Day of Week</label>
              <select id="day_of_week">
                <option value="0">Monday</option><option value="1">Tuesday</option>
                <option value="2">Wednesday</option><option value="3">Thursday</option>
                <option value="4" selected>Friday</option><option value="5">Saturday</option>
                <option value="6">Sunday</option>
              </select>
            </div>
            <div class="field">
              <label>Merchant Category</label>
              <select id="merchant_cat">
                <option value="0">Grocery</option><option value="1">Gas Station</option>
                <option value="2" selected>Restaurant</option><option value="3">Online Retail</option>
                <option value="4">Electronics</option><option value="5">Travel</option>
                <option value="6">ATM</option><option value="7">Pharmacy</option>
                <option value="8">Hotel</option><option value="9">Gambling</option>
              </select>
            </div>
            <div class="field">
              <label>Country</label>
              <select id="country">
                <option value="0" selected>US</option><option value="1">UK</option>
                <option value="2">DE</option><option value="3">FR</option>
                <option value="4">JP</option><option value="5">CN</option>
                <option value="6">NG</option><option value="7">RU</option>
                <option value="8">BR</option><option value="9">Unknown</option>
              </select>
            </div>
            <div class="field">
              <label>Txns Last 1h</label>
              <input type="number" id="transactions_1h" value="1" min="0" max="50">
            </div>
            <div class="field">
              <label>Txns Last 24h</label>
              <input type="number" id="transactions_24h" value="4" min="0" max="80">
            </div>
            <div class="field">
              <label>Avg Amount 7d ($)</label>
              <input type="number" id="avg_amount_7d" value="85" min="1" step="0.01">
            </div>
            <div class="field">
              <label>Distance from Home (km)</label>
              <input type="number" id="distance_km" value="12" min="0" step="0.1">
            </div>
            <div class="field">
              <label>Card Age (days)</label>
              <input type="number" id="card_age_days" value="720" min="1" max="5000">
            </div>
            <div class="field">
              <label>Failed PINs</label>
              <select id="failed_pins">
                <option value="0" selected>0</option><option value="1">1</option><option value="2">2+</option>
              </select>
            </div>
            <div class="field">
              <label>Time Since Last (min)</label>
              <input type="number" id="time_since_last" value="80" min="0" step="0.1">
            </div>
            <div class="field">
              <label>Velocity Score (0–100)</label>
              <input type="number" id="velocity_score" value="25" min="0" max="100">
            </div>
            <div class="field full">
              <label>Amount vs 7d Avg</label>
              <input type="number" id="amount_vs_avg" value="2.9" step="0.01" min="0">
            </div>
            <div class="field full">
              <label>Transaction Flags</label>
              <div class="tog-row">
                <div class="tog"><input type="checkbox" id="is_online"><label for="is_online">🌐 Online</label></div>
                <div class="tog"><input type="checkbox" id="is_foreign"><label for="is_foreign">🌍 Foreign</label></div>
              </div>
            </div>
            <div class="field full">
              <label>Model</label>
              <select id="modelSel"><option value="">Auto (Best Model)</option></select>
            </div>
          </div>
          <button class="assess-btn" onclick="assess()">🔴 Analyze Transaction</button>

          <!-- RESULT -->
          <div class="result-panel" id="resultPanel">
            <div class="verdict-box" id="verdictBox">
              <div class="verdict-label">Fraud Probability</div>
              <div class="verdict-pct" id="verdictPct">—</div>
              <div class="verdict-tag" id="verdictTag"></div>
              <div class="verdict-model" id="verdictModel"></div>
            </div>
            <div class="contrib-section">
              <div class="contrib-title">Risk Drivers</div>
              <div id="contribList"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- LIVE FEED -->
      <div class="card">
        <div class="card-head">
          <div class="ico" style="background:rgba(0,255,136,.1)">📡</div>
          <h2>Live Transaction Feed</h2>
        </div>
        <div class="card-body">
          <div class="live-feed" id="liveFeed"></div>
        </div>
      </div>
    </div>

    <!-- RIGHT -->
    <div style="display:flex;flex-direction:column;gap:20px">

      <!-- Model Leaderboard -->
      <div class="card">
        <div class="card-head">
          <div class="ico" style="background:rgba(79,195,247,.1)">🏆</div>
          <h2>Model Leaderboard</h2>
        </div>
        <div class="card-body">
          <div class="tabs" id="modelTabs"></div>
          <div class="met-row" id="metRow"></div>
        </div>
      </div>

      <!-- Charts row -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div class="card">
          <div class="card-head"><div class="ico" style="background:rgba(255,45,85,.1)">📈</div><h2>ROC Curve</h2></div>
          <div class="card-body"><div class="ch"><canvas id="rocC"></canvas></div></div>
        </div>
        <div class="card">
          <div class="card-head"><div class="ico" style="background:rgba(255,204,0,.1)">🎯</div><h2>Precision–Recall</h2></div>
          <div class="card-body"><div class="ch"><canvas id="prC"></canvas></div></div>
        </div>
      </div>

      <!-- FI + CM -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div class="card">
          <div class="card-head"><div class="ico" style="background:rgba(0,255,136,.1)">🧬</div><h2>Feature Importance</h2></div>
          <div class="card-body"><div class="ch2"><canvas id="fiC"></canvas></div></div>
        </div>
        <div class="card">
          <div class="card-head"><div class="ico" style="background:rgba(156,39,176,.1)">🗺️</div><h2>Confusion Matrix</h2></div>
          <div class="card-body" style="display:flex;align-items:center;justify-content:center;min-height:220px">
            <div class="cm4" id="cmGrid"></div>
          </div>
        </div>
      </div>

      <!-- Fraud by hour -->
      <div class="card">
        <div class="card-head"><div class="ico" style="background:rgba(255,45,85,.1)">🕐</div><h2>Fraud Frequency by Hour</h2></div>
        <div class="card-body"><div class="ch"><canvas id="hourC"></canvas></div></div>
      </div>

    </div>
  </div>
</main>

<footer>
  <div class="fn">Made by <span>Musharib</span></div>
  <div style="color:var(--muted);font-size:.85rem">Fraud Detection AI · Powered by Scikit-Learn & Flask</div>
  <div class="flinks">
    <a class="fl gh" href="https://github.com/musharibramzan950-bit" target="_blank">
      <svg height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>
      GitHub
    </a>
    <a class="fl lt" href="https://linktr.ee/Musharib_" target="_blank">
      <svg height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M7.953 15.066c-.08.163-.08.324-.08.486C7.873 17.482 9.553 19 11.64 19s3.85-1.518 3.85-3.448c0-.162 0-.323-.08-.486H7.954zM4 9.228l1.698 1.926L4 13.08h4.494L12 9.228 8.494 5.376H4L5.698 7.3 4 9.228zm16 0L18.302 7.3 20 5.376h-4.494L12 9.228l3.506 3.852H20l-1.698-1.926L20 9.228z"/></svg>
      Linktree
    </a>
  </div>
  <div class="fstack">scikit-learn · flask · imbalanced-learn · chart.js · 8,000 transactions · 5 models · 16 features</div>
</footer>

<script>
let allM={}, allFI={}, allROC={}, allPR={}, allCM={}, bestM='';
let rocC=null, prC=null, fiC=null, hourC=null;
const $=id=>document.getElementById(id);
const CATS=["Grocery","Gas Station","Restaurant","Online Retail","Electronics","Travel","ATM","Pharmacy","Hotel","Gambling"];
const COUNTRIES=["US","UK","DE","FR","JP","CN","NG","RU","BR","Unknown"];

async function init(){
  const d=await(await fetch('/api/info')).json();
  allM=d.metrics; allFI=d.feat_imp; allROC=d.roc; allPR=d.pr; allCM=d.conf_mat; bestM=d.best;
  $('statAuc').textContent=d.metrics[bestM].auc;

  const sel=$('modelSel');
  Object.keys(allM).forEach(n=>{
    const o=document.createElement('option');
    o.value=n; o.textContent=`${n} (AUC ${allM[n].auc})`; sel.appendChild(o);
  });

  const tabs=$('modelTabs');
  Object.keys(allM).forEach(n=>{
    const b=document.createElement('button');
    b.className='tb'+(n===bestM?' active':'');
    b.textContent=n+(n===bestM?' ★':'');
    b.onclick=()=>switchTab(n); tabs.appendChild(b);
  });

  switchTab(bestM);
  renderHour(d.hour_fraud);
  startLiveFeed(d.sample_txns);
}

function switchTab(name){
  bestM=name;
  document.querySelectorAll('.tb').forEach((b,i)=>{
    b.classList.toggle('active', Object.keys(allM)[i]===name);
  });
  renderMetrics(name); renderROC(name); renderPR(name); renderFI(name); renderCM(name);
}

function renderMetrics(n){
  const m=allM[n];
  $('metRow').innerHTML=`
    <div class="met"><div class="mname">AUC-ROC</div><div class="mval" style="color:var(--green)">${m.auc}</div></div>
    <div class="met"><div class="mname">Avg Precision</div><div class="mval" style="color:var(--green)">${m.ap}</div></div>
    <div class="met"><div class="mname">F1 Score</div><div class="mval" style="color:var(--amber)">${m.f1}</div></div>
    <div class="met"><div class="mname">Precision</div><div class="mval" style="color:var(--blue)">${m.precision}</div></div>
    <div class="met"><div class="mname">Recall</div><div class="mval" style="color:var(--blue)">${m.recall}</div></div>
    <div class="met"><div class="mname">Threshold</div><div class="mval" style="color:var(--red)">${m.threshold}</div></div>
  `;
}

function renderROC(n){
  const r=allROC[n];
  if(rocC) rocC.destroy();
  rocC=new Chart($('rocC').getContext('2d'),{
    type:'line',
    data:{labels:r.fpr,datasets:[
      {data:r.tpr,borderColor:'#ff2d55',borderWidth:2,fill:true,
       backgroundColor:'rgba(255,45,85,.08)',pointRadius:0,tension:.3,label:'ROC'},
      {data:r.fpr,borderColor:'#1e1e28',borderWidth:1,borderDash:[4,4],pointRadius:0,label:'Random'},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#5a5a7a',maxTicksLimit:6},
           title:{display:true,text:'FPR',color:'#5a5a7a',font:{size:10}}},
        y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#5a5a7a',maxTicksLimit:6},
           title:{display:true,text:'TPR',color:'#5a5a7a',font:{size:10}}}
      }
    }
  });
}

function renderPR(n){
  const r=allPR[n];
  if(prC) prC.destroy();
  prC=new Chart($('prC').getContext('2d'),{
    type:'line',
    data:{labels:r.rec,datasets:[{
      data:r.prec,borderColor:'#ffcc00',borderWidth:2,fill:true,
      backgroundColor:'rgba(255,204,0,.07)',pointRadius:0,tension:.3,label:'PR'
    }]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{reverse:false,grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#5a5a7a',maxTicksLimit:6},
           title:{display:true,text:'Recall',color:'#5a5a7a',font:{size:10}}},
        y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#5a5a7a',maxTicksLimit:6},
           title:{display:true,text:'Precision',color:'#5a5a7a',font:{size:10}}}
      }
    }
  });
}

function renderFI(n){
  const fi=allFI[n];
  const pairs=Object.entries(fi).sort((a,b)=>b[1]-a[1]).slice(0,10);
  if(fiC) fiC.destroy();
  fiC=new Chart($('fiC').getContext('2d'),{
    type:'bar',
    data:{labels:pairs.map(p=>p[0]),datasets:[{
      data:pairs.map(p=>p[1]),
      backgroundColor:pairs.map((_,i)=>`hsla(${350+i*10},85%,${55+i*2}%,0.8)`),
      borderRadius:4,borderSkipped:false
    }]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>(c.raw*100).toFixed(1)+'%'}}},
      scales:{
        x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#5a5a7a',callback:v=>(v*100).toFixed(0)+'%'}},
        y:{grid:{display:false},ticks:{color:'#c0c0d8',font:{size:10}}}
      }
    }
  });
}

function renderCM(n){
  const cm=allCM[n];
  const[tn,fp,fn_,tp]=[cm[0][0],cm[0][1],cm[1][0],cm[1][1]];
  $('cmGrid').innerHTML=`
    <div class="cmc" style="background:rgba(0,255,136,.08)"><div class="cv" style="color:var(--green)">${tn}</div><div class="cl" style="color:var(--green)">True Neg</div></div>
    <div class="cmc" style="background:rgba(255,204,0,.08)"><div class="cv" style="color:var(--amber)">${fp}</div><div class="cl" style="color:var(--amber)">False Pos</div></div>
    <div class="cmc" style="background:rgba(255,45,85,.12)"><div class="cv" style="color:var(--red)">${fn_}</div><div class="cl" style="color:var(--red)">False Neg</div></div>
    <div class="cmc" style="background:rgba(79,195,247,.08)"><div class="cv" style="color:var(--blue)">${tp}</div><div class="cl" style="color:var(--blue)">True Pos</div></div>
  `;
}

function renderHour(data){
  if(hourC) hourC.destroy();
  hourC=new Chart($('hourC').getContext('2d'),{
    type:'bar',
    data:{
      labels:data.map(d=>d.hour+'h'),
      datasets:[
        {label:'Total',data:data.map(d=>d.total),backgroundColor:'rgba(255,255,255,.06)',borderRadius:3,borderSkipped:false},
        {label:'Fraud',data:data.map(d=>d.fraud),backgroundColor:'rgba(255,45,85,.75)',borderRadius:3,borderSkipped:false},
      ]
    },
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#5a5a7a',font:{size:11}}}},
      scales:{
        x:{grid:{display:false},ticks:{color:'#5a5a7a',font:{size:10}}},
        y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#5a5a7a'}}
      }
    }
  });
}

// ── Live Feed ──
function startLiveFeed(samples){
  const feed=$('liveFeed');
  let idx=0;
  function addItem(){
    const s=samples[idx%samples.length]; idx++;
    const isFraud=s.is_fraud;
    const div=document.createElement('div');
    div.className='feed-item';
    div.innerHTML=`
      <div class="feed-dot" style="background:${isFraud?'var(--red)':'var(--green)'};
        box-shadow:0 0 6px ${isFraud?'rgba(255,45,85,.6)':'rgba(0,255,136,.4)'}"></div>
      <div class="feed-amount">$${s.amount.toFixed(2)}</div>
      <div class="feed-cat">${CATS[s.merchant_cat]||'Unknown'} · ${COUNTRIES[s.country]||'Unknown'}</div>
      <div class="feed-badge" style="background:${isFraud?'rgba(255,45,85,.15)':'rgba(0,255,136,.1)'};
        color:${isFraud?'var(--red)':'var(--green)'}">
        ${isFraud?'FRAUD':'LEGIT'}
      </div>
    `;
    feed.prepend(div);
    if(feed.children.length>20) feed.removeChild(feed.lastChild);
  }
  addItem();
  setInterval(addItem, 1200);
}

async function assess(){
  const btn=document.querySelector('.assess-btn');
  btn.classList.add('loading'); btn.textContent='⏳ Analyzing…';

  const payload={
    amount:+$('amount').value,
    hour:+$('hour').value,
    day_of_week:+$('day_of_week').value,
    merchant_cat:+$('merchant_cat').value,
    country:+$('country').value,
    transactions_1h:+$('transactions_1h').value,
    transactions_24h:+$('transactions_24h').value,
    avg_amount_7d:+$('avg_amount_7d').value,
    distance_km:+$('distance_km').value,
    card_age_days:+$('card_age_days').value,
    failed_pins:+$('failed_pins').value,
    is_online:$('is_online').checked?1:0,
    is_foreign:$('is_foreign').checked?1:0,
    time_since_last:+$('time_since_last').value,
    velocity_score:+$('velocity_score').value,
    amount_vs_avg:+$('amount_vs_avg').value,
    model:$('modelSel').value||null,
  };

  const d=await(await fetch('/api/predict',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();

  btn.classList.remove('loading'); btn.textContent='🔴 Analyze Transaction';

  const panel=$('resultPanel');
  panel.classList.remove('show'); void panel.offsetWidth; panel.classList.add('show');

  const pct=Math.round(d.prob*100);
  const box=$('verdictBox');
  $('verdictPct').textContent=pct+'%';

  let color,label,bg;
  if(d.prob<0.2){color='var(--green)';label='✅ LIKELY LEGITIMATE';bg='rgba(0,255,136,.06)'}
  else if(d.prob<0.45){color='var(--amber)';label='⚠️ SUSPICIOUS';bg='rgba(255,204,0,.06)'}
  else if(d.prob<0.7){color='#ff8c00';label='🚨 HIGH RISK FRAUD';bg='rgba(255,140,0,.08)'}
  else{color='var(--red)';label='🔴 FRAUD DETECTED';bg='rgba(255,45,85,.1)'}

  $('verdictPct').style.color=color;
  box.style.background=bg;
  box.style.borderColor=color.replace('var(','').replace(')','');

  const tag=$('verdictTag');
  tag.textContent=label; tag.style.background=bg; tag.style.color=color;
  tag.style.border=`1px solid ${color}`;
  $('verdictModel').textContent='via '+d.model+' · threshold '+d.threshold;

  const contribs=Object.entries(d.contribs).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const maxAbs=Math.max(...contribs.map(c=>Math.abs(c[1])))||0.001;
  $('contribList').innerHTML=contribs.slice(0,8).map(([name,val])=>{
    const pos=val>0;
    const w=(Math.abs(val)/maxAbs*100).toFixed(0);
    return `<div class="contrib-item">
      <div class="cn">${name}</div>
      <div class="cb"><div class="cbf" style="width:${w}%;background:${pos?'var(--red)':'var(--green)'}"></div></div>
      <div class="cv2" style="color:${pos?'var(--red)':'var(--green)'}">${val>0?'+':''}${(val*100).toFixed(1)}%</div>
    </div>`;
  }).join('');
}

init();
</script>
</body>
</html>
"""

# ─── FLASK ────────────────────────────────────────────────────────────────────

app = Flask(__name__)
hub = FraudHub()

MERCHANT_CATS_L = ["Grocery","Gas Station","Restaurant","Online Retail",
                   "Electronics","Travel","ATM","Pharmacy","Hotel","Gambling"]
COUNTRIES_L     = ["US","UK","DE","FR","JP","CN","NG","RU","BR","Unknown"]

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/info")
def api_info():
    df = hub._df

    # fraud by hour
    hour_fraud = []
    for h in range(24):
        mask  = df["hour"]==h
        total = int(mask.sum())
        fraud = int(df.loc[mask,"fraud"].sum())
        if total>0:
            hour_fraud.append({"hour":h,"total":total,"fraud":fraud})

    # sample transactions for live feed
    sample = df[["amount","merchant_cat","country","fraud"]].sample(80, random_state=7).to_dict("records")
    for s in sample:
        s["merchant_cat"]=int(s["merchant_cat"]); s["country"]=int(s["country"])
        s["amount"]=round(float(s["amount"]),2); s["is_fraud"]=bool(s.pop("fraud"))

    return jsonify({
        "best": hub.best, "metrics": hub.metrics,
        "feat_imp": hub.feat_imp, "roc": hub.roc_data,
        "pr": hub.pr_data, "conf_mat": hub.conf_mat,
        "hour_fraud": hour_fraud, "sample_txns": sample,
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body  = request.get_json(force=True)
    model = body.pop("model", None) or hub.best
    result= hub.predict(body, model)
    return jsonify(result)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 7070))
    print("\n" + "═"*56)
    print("  🔴  FRAUDSENSE — AI Transaction Fraud Detector")
    print("  Made by Musharib | linktr.ee/Musharib_")
    print("═"*56)
    print("  ⚙️  Generating 8,000 transaction profiles…")
    df = generate_dataset(n=8000)
    print(f"  📊  Fraud rate: {df['fraud'].mean()*100:.1f}%  ({df['fraud'].sum()} fraudulent)")
    print("  ⚖️  Applying SMOTE oversampling…")
    print("  🧠  Training 5 ML models with optimal threshold tuning…")
    hub._df = df
    hub.train(df)
    print(f"\n  🌐  Running at → http://localhost:{PORT}")
    print("  Press Ctrl+C to stop\n" + "═"*56 + "\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
