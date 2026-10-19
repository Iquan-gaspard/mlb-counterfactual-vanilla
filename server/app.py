import sys
import os
import io
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from werkzeug.middleware.proxy_fix import ProxyFix
from pitcher_arsenal_extractor import extract_pitcher_arsenal, get_pitch_physics
from dotenv import load_dotenv

load_dotenv()

# 🌟 雲端極致保護：強制單執行緒，防止低配 vCPU 搶占資源當機
torch.set_num_threads(1)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# 🛡️ 1. 動態 CORS 白名單 (支援本地測試與 Vercel 正式網域)
origins_env = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://127.0.0.1:3000"
)
ALLOWED_ORIGINS = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# 🛡️ 2. 真實 IP 限流器 (解決 BFF 代理導致的集體封鎖)
def get_real_client_ip():
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "127.0.0.1"

limiter = Limiter(
    key_func=get_real_client_ip,
    app=app,
    storage_uri="memory://",
    default_limits=["500 per day", "60 per minute"],
    default_limits_exempt_when=lambda: request.method == 'OPTIONS'
)

# 🛡️ 3. 內部服務通行金鑰驗證
INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET", "dev-secret-key-change-in-prod")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        client_key = request.headers.get("X-Internal-API-Key")
        if not client_key or client_key != INTERNAL_API_SECRET:
            return jsonify({"error": "Zero Trust Policy: Unauthorized service request"}), 401
        return f(*args, **kwargs)
    return decorated

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 神經網絡模型定義 (5 類別 / 13維序列 / 20維情境)
# ==========================================
class BaseballTransformerSingleTask(nn.Module):
    def __init__(self, seq_feature_dim=13, ctx_feature_dim=20, d_model=32, n_heads=4, num_layers=2):
        super(BaseballTransformerSingleTask, self).__init__()
        self.input_projection = nn.Linear(seq_feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 6, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_flatten = nn.Linear(d_model, 16)
        self.ctx_dense = nn.Linear(ctx_feature_dim, 16)
        self.fc = nn.Sequential(
            nn.Linear(16 + 16, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(), 
            nn.Linear(16, 5)
        )
    def forward(self, x_seq, x_ctx):
        x_seq = self.input_projection(x_seq) + self.pos_embedding
        enc_out = self.transformer_encoder(x_seq)
        seq_repr = torch.relu(self.seq_flatten(enc_out[:, -1, :]))
        ctx_repr = torch.relu(self.ctx_dense(x_ctx))
        return self.fc(torch.cat((seq_repr, ctx_repr), dim=1))

# 🌟 雲端極致提速：模型延遲常駐快取 (只載入一次，最多僅佔 ~25MB RAM，杜絕每次請求讀硬碟)
LOADED_MODELS = {}

def get_model(matchup_key):
    if matchup_key in LOADED_MODELS:
        return LOADED_MODELS[matchup_key]
    
    model_path = os.path.join(BASE_DIR, f'baseball_transformer_weights_MultiClass_{matchup_key}_best.pth')
    if os.path.exists(model_path):
        model = BaseballTransformerSingleTask().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        LOADED_MODELS[matchup_key] = model
        return model
    return None

STRIKE_ZONES = {
    '左上': (-0.55, 0.83), '中上': (0.0, 0.83), '右上': (0.55, 0.83),
    '左中': (-0.55, 0.50), '正中': (0.0, 0.50), '右中': (0.55, 0.50),
    '左下': (-0.55, 0.17), '中下': (0.0, 0.17), '右下': (0.55, 0.17)
}
BALL_ZONES = {
    '壞_左上': (-1.1, 1.17),  '壞_中上': (0.0, 1.17),  '壞_右上': (1.1, 1.17),
    '壞_左中': (-1.1, 0.50),                           '壞_右中': (1.1, 0.50),
    '壞_左下': (-1.1, -0.17), '壞_中下': (0.0, -0.17), '壞_右下': (1.1, -0.17),
    '壞_挖地瓜': (0.0, -0.34)
}
ALL_ZONES = {**STRIKE_ZONES, **BALL_ZONES}

def get_zone_name(px, pz):
    best_zone, min_dist = '正中', float('inf')
    for z_name, (cx, cz) in ALL_ZONES.items():
        dist = (px - cx)**2 + (pz - cz)**2
        if dist < min_dist:
            min_dist, best_zone = dist, z_name
    return best_zone

def get_pull_ohe_3way(pull_rate):
    if pd.isna(pull_rate): pull_rate = 0.40
    if pull_rate > 1.0: pull_rate /= 100.0
    if pull_rate > 0.45: return [1.0, 0.0, 0.0]
    elif pull_rate < 0.35: return [0.0, 0.0, 1.0]
    else: return [0.0, 1.0, 0.0]

def get_arm_angle_ohe(arm_angle):
    if pd.isna(arm_angle): arm_angle = 38.6
    if arm_angle > 50.0: return [1.0, 0.0, 0.0]
    elif arm_angle < 25.0: return [0.0, 0.0, 1.0]
    else: return [0.0, 1.0, 0.0]

# 🌟 修正：接收投手的真實 arm_angle，不跟打者討
def build_20d_context(adv, pitcher_arm_angle=38.6, b_count=0, s_count=0):
    ball_ohe = [1.0 if b_count == i else 0.0 for i in range(4)]
    strike_ohe = [1.0 if s_count == i else 0.0 for i in range(3)]
    pull_ohe = get_pull_ohe_3way(adv.get('pull_percent', 40.0))
    arm_ohe = get_arm_angle_ohe(pitcher_arm_angle)
    norm_age = max(0.0, min(1.0, (adv.get('player_age', 28.0) - 18.0) / 25.0))
    norm_z_swing = adv.get('z_swing_percent', 65.0) / 100.0
    norm_z_miss = adv.get('z_swing_miss_percent', 15.0) / 100.0
    norm_oz_swing = adv.get('oz_swing_percent', 30.0) / 100.0
    norm_oz_miss = adv.get('oz_swing_miss_percent', 45.0) / 100.0
    norm_meatball = adv.get('meatball_swing_percent', 75.0) / 100.0
    norm_attack_angle = max(0.0, min(1.0, adv.get('attack_angle', 8.0) / 20.0))
    advanced_stats = [norm_age, norm_z_swing, norm_z_miss, norm_oz_swing, norm_oz_miss, norm_meatball, norm_attack_angle]
    return ball_ohe + strike_ohe + pull_ohe + arm_ohe + advanced_stats

# ==========================================
# 迷你 CSV 備援載入邏輯 (保留以防 Supabase 故障)
# ==========================================
ALL_ARSENALS = {}
df_2026 = pd.DataFrame()
csv_path = os.path.join(BASE_DIR, 'mlb_mini_2026.csv.gz')

if os.path.exists(csv_path):
    try:
        use_cols = [
            'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'pitcher_name', 
            'p_throws', 'pitch_type', 'release_speed', 'plate_x', 'plate_z_norm', 
            'pfx_x', 'pfx_z', 'description', 'stand', 'batter_name', 'batter', 'launch_speed',
            'balls', 'strikes', 'game_date',
            'Pull%', 'player_age', 'z_swing_percent', 'z_swing_miss_percent', 
            'oz_swing_percent', 'oz_swing_miss_percent', 'meatball_swing_percent',
            'attack_angle', 'arm_angle'
        ]
        df_2026 = pd.read_csv(csv_path, compression='gzip', usecols=use_cols)
        df_2026['zone_name'] = df_2026.apply(lambda x: get_zone_name(float(x['plate_x']), float(x['plate_z_norm'])), axis=1)
        
        pitcher_info = df_2026[['pitcher', 'pitcher_name', 'p_throws']].drop_duplicates(subset=['pitcher'])
        for _, row in pitcher_info.iterrows():
            p_id = int(row['pitcher'])
            p_name = "Y. Yamamoto" if p_id == 808967 else str(row['pitcher_name'])
            try:
                arsenal = extract_pitcher_arsenal(df_2026, p_id, min_pitch_count=10)
                if len(arsenal['valid_pitches']) > 0:
                    ALL_ARSENALS[p_id] = {"name": p_name, "p_throws": str(row['p_throws']), "arsenal_data": arsenal, "valid_pitches": arsenal['valid_pitches']}
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ 載入備用 CSV 失敗：{e}")

# ==========================================
# API 路由區塊 (全數掛上金鑰防護)
# ==========================================
@app.route('/api/pitchers', methods=['GET'])
@require_api_key
def get_pitchers():
    response = {}
    for p_id, data in ALL_ARSENALS.items():
        arsenal_list = [{"value": pt, "label": f"{pt} ({data['arsenal_data']['overall_arsenal'][pt]['speed']*100:.1f} mph)"} for pt in data['valid_pitches']]
        response[str(p_id)] = {"name": data['name'], "p_throws": data['p_throws'], "arsenal": arsenal_list}
    return jsonify(response)

@app.route('/api/atbats/<pitcher_id>', methods=['GET'])
@require_api_key
def get_atbats(pitcher_id):
    if df_2026.empty: return jsonify([])
    p_df = df_2026[df_2026['pitcher'] == int(pitcher_id)]
    grouped = p_df.groupby(['game_pk', 'at_bat_number'])
    atbats_list = []
    for (g_pk, ab_num), group in grouped:
        if len(group) < 2: continue
        group = group.sort_values('pitch_number')
        first_pitch = group.iloc[0]
        stand = first_pitch.get('stand', 'R')
        adv_stats = {
            'pull_percent': float(first_pitch.get('Pull%', 0.40)),
            'player_age': float(first_pitch.get('player_age', 28.0)),
            'z_swing_percent': float(first_pitch.get('z_swing_percent', 65.0)),
            'z_swing_miss_percent': float(first_pitch.get('z_swing_miss_percent', 15.0)),
            'oz_swing_percent': float(first_pitch.get('oz_swing_percent', 30.0)),
            'oz_swing_miss_percent': float(first_pitch.get('oz_swing_miss_percent', 45.0)),
            'meatball_swing_percent': float(first_pitch.get('meatball_swing_percent', 75.0)),
            'attack_angle': float(first_pitch.get('attack_angle', 8.0)),
            'arm_angle': float(first_pitch.get('arm_angle', 38.6))
        }
        batter_name = first_pitch.get('batter_name', f"Batter {first_pitch.get('batter', 'Unknown')}")
        label = f"{first_pitch.get('game_date', '2026')} vs {batter_name} ({'左' if stand=='L' else '右'}打)"
        seq = []
        for _, p in group.iterrows():
            px, pz = float(p.get('plate_x', 0)), float(p.get('plate_z_norm', 0))
            is_strike = p['description'] in ['called_strike', 'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']
            ls_val = p.get('launch_speed')
            ls_str = f"{float(ls_val):.1f} mph" if pd.notna(ls_val) and str(ls_val).strip() != "" else ""
            x_ui = 50.0 + (px / 0.83) * 50.0
            y_ui = 100.0 - (pz * 100.0)
            seq.append({
                "num": int(p['pitch_number']), "type": str(p['pitch_type']), "name": str(p['pitch_type']), 
                "speed": f"{float(p['release_speed']):.1f} mph", "result": str(p['description']),
                "isStrike": is_strike, "count": f"{int(p['balls'])} - {int(p['strikes'])}",
                "x": max(-50, min(150, x_ui)), "y": max(-50, min(150, y_ui)),
                "zone_name": str(p.get('zone_name', '正中')),
                "raw_speed": float(p.get('release_speed', 95.0)), "raw_px": px, "raw_pz": pz,
                "raw_pfx_x": float(p.get('pfx_x', 0.0)), "raw_pfx_z": float(p.get('pfx_z', 0.0)),
                "launch_speed": ls_str 
            })
        atbats_list.append({"ab_id": f"{g_pk}_{ab_num}", "label": label, "stand": stand, "adv_stats": adv_stats, "sequence": seq})
    atbats_list.sort(key=lambda x: x['label'])
    return jsonify(atbats_list)

@app.route('/api/batters', methods=['GET'])
@require_api_key
def get_batters():
    if df_2026.empty: return jsonify([])
    unique_batters = df_2026.drop_duplicates(subset=['batter']).copy()
    batters_list = []
    for _, row in unique_batters.iterrows():
        stand = row.get('stand', 'R')
        batter_name = row.get('batter_name', f"Batter {row.get('batter')}")
        if pd.isna(batter_name) or batter_name == "Unknown": continue
        adv_stats = {
            'pull_percent': float(row.get('Pull%', 0.40)),
            'player_age': float(row.get('player_age', 28.0)),
            'z_swing_percent': float(row.get('z_swing_percent', 65.0)),
            'z_swing_miss_percent': float(row.get('z_swing_miss_percent', 15.0)),
            'oz_swing_percent': float(row.get('oz_swing_percent', 30.0)),
            'oz_swing_miss_percent': float(row.get('oz_swing_miss_percent', 45.0)),
            'meatball_swing_percent': float(row.get('meatball_swing_percent', 75.0)),
            'attack_angle': float(row.get('attack_angle', 8.0)),
            'arm_angle': float(row.get('arm_angle', 38.6))
        }
        batters_list.append({
            "batter_id": int(row['batter']), "name": batter_name, "stand": stand, 
            "label": f"{batter_name} ({'左' if stand=='L' else '右'}打)", "adv_stats": adv_stats
        })
    batters_list.sort(key=lambda x: x['name'])
    return jsonify(batters_list)

# 模式 1：反事實推演 (Counterfactual)
@app.route('/api/simulate', methods=['POST'])
@limiter.limit("30 per minute")
@require_api_key
def simulate_counterfactual():
    data = request.json or {}
    sim_mode = data.get('sim_mode', 'last')
    p_type = data.get('pitch_type')
    p_zone = data.get('pitch_zone')
    stand = data.get('stand', 'R')
    pitcher_id = int(data.get('pitcher_id', 808967))
    adv_stats = data.get('adv_stats', {})
    arm_angle = float(data.get('arm_angle', 38.6))
    b_count = int(data.get('b_count', 0))
    s_count = int(data.get('s_count', 0))
    
    p_throws = data.get('p_throws') or ALL_ARSENALS.get(pitcher_id, {}).get('p_throws', 'R')
    matchup_key = f"{p_throws}HP vs {stand}HB"
    
    model = get_model(matchup_key)
    if not model:
        return jsonify({"error": f"找不到對戰模型 {matchup_key}"}), 500

    # 優先使用 BFF 傳入的分區物理數值
    physics = data.get('physics')
    if not physics:
        p_arsenal = ALL_ARSENALS.get(pitcher_id, {}).get('arsenal_data')
        p_df = df_2026[df_2026['pitcher'] == pitcher_id] if not df_2026.empty else pd.DataFrame()
        physics = get_smart_physics(p_df, p_type, p_zone, p_arsenal) if p_arsenal else {'speed': 0.95, 'pfx_x': 0.0, 'pfx_z': 0.0}

    px_new, pz_new = ALL_ZONES.get(p_zone, (0.0, 0.0))

    n2 = data.get('n2_pitch', {})
    sn2, pxn2, pzn2, pfx_xn2, pfx_zn2 = float(n2.get('speed', 0.95)), float(n2.get('px', 0.0)), float(n2.get('pz', 0.0)), float(n2.get('pfx_x', 0.0)), float(n2.get('pfx_z', 0.0))
    
    n1 = data.get('n1_pitch', {})
    sn1, pxn1, pzn1, pfx_xn1, pfx_zn1 = float(n1.get('speed', 0.95)), float(n1.get('px', 0.0)), float(n1.get('pz', 0.0)), float(n1.get('pfx_x', 0.0)), float(n1.get('pfx_z', 0.0))
    n1_outcome = data.get('n1_outcome', [1.0, 0.0, 0.0])
    
    orig = data.get('orig_pitch', {})
    s_orig, px_orig, pz_orig, pfx_x_orig, pfx_z_orig = float(orig.get('speed', 0.95)), float(orig.get('px', 0.0)), float(orig.get('pz', 0.0)), float(orig.get('pfx_x', 0.0)), float(orig.get('pfx_z', 0.0))

    pitch_n1_orig = [sn1, pxn1, pzn1, pfx_xn1, pfx_zn1] + n1_outcome + [sn1 - sn2, pxn1 - pxn2, pzn1 - pzn2, pfx_xn1 - pfx_xn2, pfx_zn1 - pfx_zn2]
    pitch_n_orig = [s_orig, px_orig, pz_orig, pfx_x_orig, pfx_z_orig, 0.0, 0.0, 0.0, s_orig - sn1, px_orig - pxn1, pz_orig - pzn1, pfx_x_orig - pfx_xn1, pfx_z_orig - pfx_zn1]

    if sim_mode == 'last':
        pitch_n1_cf = pitch_n1_orig
        pitch_n_cf = [physics['speed'], px_new, pz_new, physics['pfx_x'], physics['pfx_z'], 0.0, 0.0, 0.0, physics['speed'] - sn1, px_new - pxn1, pz_new - pzn1, physics['pfx_x'] - pfx_xn1, physics['pfx_z'] - pfx_zn1]
    else:
        pitch_n1_cf = [physics['speed'], px_new, pz_new, physics['pfx_x'], physics['pfx_z']] + n1_outcome + [physics['speed'] - sn2, px_new - pxn2, pz_new - pzn2, physics['pfx_x'] - pfx_xn2, physics['pfx_z'] - pfx_zn2]
        pitch_n_cf = [s_orig, px_orig, pz_orig, pfx_x_orig, pfx_z_orig, 0.0, 0.0, 0.0, s_orig - physics['speed'], px_orig - px_new, pz_orig - pz_new, pfx_x_orig - physics['pfx_x'], pfx_z_orig - physics['pfx_z']]

    ctx = build_20d_context(adv_stats, pitcher_arm_angle=arm_angle, b_count=b_count, s_count=s_count)

    t_seq_cf = torch.tensor(np.array([[[0.0]*13]*4 + [pitch_n1_cf, pitch_n_cf]]), dtype=torch.float32).to(device)
    t_seq_orig = torch.tensor(np.array([[[0.0]*13]*4 + [pitch_n1_orig, pitch_n_orig]]), dtype=torch.float32).to(device)
    t_ctx = torch.tensor(np.array([ctx]), dtype=torch.float32).to(device)

    with torch.no_grad():
        probs_cf = F.softmax(model(t_seq_cf, t_ctx), dim=1).cpu().numpy().flatten().tolist()
        probs_orig = F.softmax(model(t_seq_orig, t_ctx), dim=1).cpu().numpy().flatten().tolist()

    return jsonify({
        "cf_probs": [round(p * 100, 1) for p in probs_cf],
        "orig_probs": [round(p * 100, 1) for p in probs_orig]
    })

# 模式 2 & 3：序列推演 (Sequence Simulation)
@app.route('/api/simulate_sequence', methods=['POST'])
@limiter.limit("30 per minute")
@require_api_key
def simulate_sequence():
    data = request.json or {}
    stand = data.get('stand', 'R')
    pitches = data.get('pitches', []) 
    adv_stats = data.get('adv_stats', {})
    arm_angle = float(data.get('arm_angle', 38.6))
    target_b = int(data.get('b_count', 0))
    target_s = int(data.get('s_count', 0))

    # 算力邊界檢查
    if not isinstance(pitches, list) or len(pitches) == 0 or len(pitches) > 7:
        return jsonify({"error": "Invalid pitches length (1-7)"}), 400

    p_throws = data.get('p_throws') or 'R'
    matchup_key = f"{p_throws}HP vs {stand}HB"

    model = get_model(matchup_key)
    if not model:
        return jsonify({"error": f"找不到對戰模型 {matchup_key}"}), 500

    results = []
    history_features = [] 
    current_b = 0
    current_s = 0

    for i, p_info in enumerate(pitches):
        p_type = p_info['type']
        p_zone = p_info['zone']
        is_target_pitch = (i == len(pitches) - 1)
        
        # 優先直接吃 BFF 傳入的真實分區數值
        speed = float(p_info.get('speed', 0.95))
        pfx_x = float(p_info.get('pfx_x', 0.0))
        pfx_z = float(p_info.get('pfx_z', 0.0))
        px, pz = ALL_ZONES.get(p_zone, (0.0, 0.0))
        
        if i == 0:
            d_s, d_px, d_pz, d_pfx, d_pfz = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            prev = history_features[-1]
            d_s = speed - prev['speed']
            d_px = px - prev['px']
            d_pz = pz - prev['pz']
            d_pfx = pfx_x - prev['pfx_x']
            d_pfz = pfx_z - prev['pfx_z']
            
        # 歷史球反應填寫
        res_str = str(p_info.get('result', '')).strip()
        if not is_target_pitch and res_str:
            outcome_vec = [
                1.0 if res_str == 'Take' else 0.0,
                1.0 if res_str == 'Whiff' else 0.0,
                1.0 if res_str == 'Foul' else 0.0
            ]
        else:
            outcome_vec = [0.0, 0.0, 0.0]

        pitch_vec = [
            speed, px, pz, pfx_x, pfx_z,
            *outcome_vec,
            d_s, d_px, d_pz, d_pfx, d_pfz
        ]
        history_features.append({'speed': speed, 'px': px, 'pz': pz, 'pfx_x': pfx_x, 'pfx_z': pfx_z, 'vec': pitch_vec})
        
        current_seq = [f['vec'] for f in history_features[-6:]]
        padded_seq = [[0.0]*13] * (6 - len(current_seq)) + current_seq
        
        pitch_b = target_b if is_target_pitch and ('b_count' in data or 's_count' in data) else current_b
        pitch_s = target_s if is_target_pitch and ('b_count' in data or 's_count' in data) else current_s
            
        ctx = build_20d_context(adv_stats, pitcher_arm_angle=arm_angle, b_count=pitch_b, s_count=pitch_s)
        
        t_seq = torch.tensor(np.array([padded_seq]), dtype=torch.float32).to(device)
        t_ctx = torch.tensor(np.array([ctx]), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            probs = F.softmax(model(t_seq, t_ctx), dim=1).cpu().numpy().flatten().tolist()
            
        results.append({
            "pitch_num": i + 1,
            "type": p_type,
            "zone": p_zone,
            "probs": {
                "take": round(probs[0]*100, 1),
                "whiff": round(probs[1]*100, 1),
                "foul": round(probs[2]*100, 1),
                "weak_contact": round(probs[3]*100, 1),
                "hard_contact": round(probs[4]*100, 1)
            }
        })

        if res_str == 'Take':
            if p_zone in STRIKE_ZONES: current_s = min(current_s + 1, 2)
            else: current_b = min(current_b + 1, 3)
        elif res_str in ['Whiff']:
            current_s = min(current_s + 1, 2)
        elif res_str in ['Foul']:
            if current_s < 2: current_s += 1
        else:
            if p_zone in STRIKE_ZONES: current_s = min(current_s + 1, 2)
            else: current_b = min(current_b + 1, 3)
        
    return jsonify({"results": results})

if __name__ == '__main__':
    is_dev = os.getenv("FLASK_ENV") == "development"
    app.run(host='0.0.0.0', port=5000, debug=is_dev)