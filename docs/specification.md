# サブシーズナル気象予測システム 仕様書

## 1. 概要

複数のAI気象予測モデル（GraphCast、FourCastNet、MetNet-3）を用いてサブシーズナル（1週間〜1ヶ月先）の気象予測を行い、アンサンブル推論とメタ学習により高精度な確率的予測を生成するシステム。予測結果はGoogleカレンダーに終日予定として連携する。

## 2. システム構成

```
[Google Colab]          [Supabase]         [Vercel]            [Google]
 ┌──────────────┐      ┌───────────┐      ┌───────────┐      ┌──────────────┐
 │ データ取得    │      │ PostgreSQL│      │ Next.js   │      │ Google       │
 │ ├ ERA5       │      │           │      │ ├ UI      │      │ Calendar API │
 │ └ GFS/ECMWF │─────▶│ 予測結果  │─────▶│ ├ API     │─────▶│              │
 │              │      │ テーブル   │      │ └ Auth    │      │              │
 │ アンサンブル  │      │           │      │ (Google   │      │              │
 │ 推論(×50)    │      │           │      │  OAuth)   │      │              │
 │              │      │           │      │           │      │              │
 │ メタ学習統合  │      │           │      │           │      │              │
 └──────────────┘      └───────────┘      └───────────┘      └──────────────┘
```

## 3. 使用AIモデル

| モデル | 開発元 | 特徴 | 主な用途 |
|---|---|---|---|
| GraphCast | Google DeepMind | GNNベース、ERA5学習済み、高精度中期予報 | 気温・気圧・風速の中期予測（〜10日） |
| FourCastNet | NVIDIA | Vision Transformerベース、高速推論 | 高解像度の短〜中期予測 |
| MetNet-3 | Google | 高解像度、降水予測に強い | 降水確率・天気概況の予測 |

## 4. 処理フロー

### Step 1: データ取得

#### 4.1.1 ERA5再解析データ
- **取得元**: Copernicus Climate Data Store (CDS) API
- **用途**: モデル学習・検証用の過去データ、初期値の基準
- **取得変数**:
  - `2m_temperature` (地上2m気温)
  - `mean_sea_level_pressure` (海面気圧)
  - `10m_u_component_of_wind` (10m東西風速)
  - `10m_v_component_of_wind` (10m南北風速)
  - `total_precipitation` (総降水量)
  - `relative_humidity` (相対湿度、複数気圧面)
  - `geopotential` (ジオポテンシャル高度、500hPa等)
- **解像度**: 0.25° × 0.25°
- **時間分解能**: 6時間ごと（00, 06, 12, 18 UTC）

#### 4.1.2 初期値データ（リアルタイム）
- **GFS (Global Forecast System)**
  - 取得元: NOAA NOMADS / AWS Open Data
  - 更新頻度: 6時間ごと
  - 解像度: 0.25°
- **ECMWF (IFS)**
  - 取得元: ECMWF Open Data（一般公開分）
  - 更新頻度: 12時間ごと
  - 解像度: 0.25°

#### 4.1.3 データ前処理
- NetCDF / GRIB2形式からモデル入力形式への変換
- 欠損値の補間（時間・空間）
- 正規化（各変数ごとのmean/stdによる標準化）
- 対象領域の切り出し（日本域: 北緯24°〜46°, 東経122°〜150° を既定とする）

### Step 2: アンサンブル推論

#### 4.2.1 初期値摂動
- ベースとなる初期値データ（GFS/ECMWF最新解析値）に対し、50パターンの摂動を生成
- **摂動手法**:
  - ガウスノイズ付加（標準偏差: 各変数の気候学的変動の1〜5%）
  - Bred Vector法の簡易実装（オプション）
- **摂動対象変数**: 気温、気圧、風速、湿度の全入力変数

#### 4.2.2 各モデルでの推論実行
- 各摂動パターン（50通り）× 各モデル（3種）= 最大150回の推論
- **推論ステップ**:
  - GraphCast: 6時間刻みでオートリグレッシブに最大30日先まで
  - FourCastNet: 6時間刻みでオートリグレッシブに最大14日先まで
  - MetNet-3: 1〜24時間先の短期予測を逐次延長
- **出力変数**: 気温、降水量、湿度、気圧、風速（各格子点）

#### 4.2.3 確率統計処理
各モデルの50メンバーアンサンブル出力に対し、以下を計算:

| 統計量 | 算出方法 | 用途 |
|---|---|---|
| アンサンブル平均 | 50メンバーの平均値 | 最尤予測値 |
| アンサンブル拡散 | 50メンバーの標準偏差 | 予測不確実性 |
| 確率分布 | ヒストグラム/カーネル密度推定 | 閾値超過確率 |
| パーセンタイル | 10th, 25th, 50th, 75th, 90th | 予測幅の表現 |

### Step 3: メタ学習統合

#### 4.3.1 特徴量設計
各モデルのアンサンブル出力から以下の特徴量を生成:

```
特徴量ベクトル = [
  GraphCast_mean, GraphCast_std, GraphCast_median,
  FourCastNet_mean, FourCastNet_std, FourCastNet_median,
  MetNet3_mean, MetNet3_std, MetNet3_median,
  モデル間の一致度（agreement_score）,
  リードタイム（予測先日数）,
  季節（月）,
  対象地点の緯度・経度,
  直近の気象パターン指標（NAO, ENSO等）
]
```

#### 4.3.2 メタ学習モデル
- **LightGBM（主モデル）**
  - 目的変数: 実際の観測値（ERA5またはアメダス）
  - 学習データ: 過去のモデル予測結果と実観測の対応
  - ハイパーパラメータ: Optunaによる自動チューニング
- **小規模ニューラルネットワーク（補助モデル）**
  - 2〜3層のMLP
  - モデル間の非線形な相互作用を捕捉
- **最終出力**: LightGBMとMLPの加重平均（重みもバリデーションで決定）

#### 4.3.3 学習・評価サイクル
- 学習期間: 過去2年分のバックテスト
- 評価指標: RMSE、CRPS（Continuous Ranked Probability Score）、信頼区間カバー率
- クロスバリデーション: 時系列を考慮した walk-forward validation

### Step 4: 天気概況への変換

#### 4.4.1 天気アイコン判定ロジック

```
if 降水確率 >= 60%:
    if 気温 <= 2℃:
        天気 = "雪"
    else:
        天気 = "雨"
elif 降水確率 >= 30%:
    if 日照時間推定 > 50%:
        天気 = "曇時々晴"
    else:
        天気 = "曇時々雨"
elif 雲量推定 >= 70%:
    天気 = "曇り"
elif 雲量推定 >= 30%:
    天気 = "晴時々曇"
else:
    天気 = "晴れ"
```

#### 4.4.2 出力フォーマット

```json
{
  "date": "2026-04-15",
  "location": "Tokyo",
  "weather": "晴れ",
  "temp_max": 22,
  "temp_min": 14,
  "precipitation_prob": 15,
  "confidence": 0.72,
  "model_agreement": 0.85,
  "details": {
    "humidity": 55,
    "wind_speed": 3.2,
    "pressure": 1015
  }
}
```

## 5. データベース設計（Supabase）

### 5.1 テーブル定義

#### `forecasts` テーブル
| カラム | 型 | 説明 |
|---|---|---|
| id | uuid | PK |
| target_date | date | 予測対象日 |
| location | text | 地点名 |
| latitude | float8 | 緯度 |
| longitude | float8 | 経度 |
| weather | text | 天気概況（晴れ/曇り/雨/雪） |
| temp_max | float4 | 最高気温（℃） |
| temp_min | float4 | 最低気温（℃） |
| precipitation_prob | int2 | 降水確率（%） |
| confidence | float4 | 予測信頼度（0〜1） |
| model_agreement | float4 | モデル一致度（0〜1） |
| humidity | float4 | 湿度（%） |
| wind_speed | float4 | 風速（m/s） |
| pressure | float4 | 気圧（hPa） |
| created_at | timestamptz | 予測実行日時 |
| run_id | uuid | 予測バッチID |

#### `forecast_runs` テーブル
| カラム | 型 | 説明 |
|---|---|---|
| id | uuid | PK（run_id） |
| executed_at | timestamptz | 実行日時 |
| initial_data_source | text | 初期値ソース（GFS/ECMWF） |
| ensemble_size | int4 | アンサンブルメンバー数 |
| models_used | text[] | 使用モデル一覧 |
| status | text | 実行ステータス |

#### `calendar_sync_log` テーブル
| カラム | 型 | 説明 |
|---|---|---|
| id | uuid | PK |
| user_id | uuid | ユーザーID |
| forecast_id | uuid | FK → forecasts.id |
| google_event_id | text | Googleカレンダーイベント ID |
| synced_at | timestamptz | 同期日時 |
| status | text | 同期ステータス |

## 6. Next.js フロントエンド

### 6.1 技術スタック
- **フレームワーク**: Next.js 14+ (App Router)
- **UI**: Tailwind CSS + shadcn/ui
- **認証**: NextAuth.js（Google OAuth 2.0）
- **DB接続**: Supabase Client SDK
- **Googleカレンダー**: Google Calendar API v3

### 6.2 画面構成

#### 6.2.1 ログインページ (`/`)
- Googleアカウントでのログインボタン
- OAuth scope: `calendar.events`（カレンダーイベントの読み書き）

#### 6.2.2 予測作成ページ (`/forecast`)
- **入力項目**:
  - 予測対象地点（プルダウン: 主要都市 or 緯度経度の手入力）
  - 期間指定モード:
    - 「期間」: 開始日〜終了日の日付レンジピッカー
    - 「ピンポイント」: 単一日付ピッカー
- **実行ボタン**: 「予測を取得」
- **結果表示**:
  - 日別の予測カード（天気アイコン、気温、降水確率、信頼度）
  - 信頼度のカラーインジケータ（緑:高 → 黄:中 → 赤:低）

#### 6.2.3 カレンダー連携ページ (`/calendar`)
- 予測結果一覧の表示
- 「Googleカレンダーに登録」ボタン
- 登録確認ダイアログ
- 同期ステータスの表示

### 6.3 Googleカレンダー連携仕様

#### 6.3.1 イベント作成フォーマット
- **種別**: 終日予定
- **タイトル**: `【AI予報】{天気}/{最高気温}℃`
  - 例: `【AI予報】晴れ/20℃`
- **説明文**:
  ```
  🌡 最高 {temp_max}℃ / 最低 {temp_min}℃
  💧 降水確率 {precipitation_prob}%
  📊 予測信頼度 {confidence}%
  🤖 モデル一致度 {model_agreement}%
  ─────────────
  GraphCast / FourCastNet / MetNet-3 アンサンブル予測
  ```
- **カラー**: 天気に応じた色ID
  - 晴れ: 5（バナナ/黄）
  - 曇り: 8（グラファイト/灰）
  - 雨: 9（ブルーベリー/青）
  - 雪: 1（ラベンダー/薄紫）

#### 6.3.2 更新ルール
- 同一日付・同一地点の予測が更新された場合、既存イベントを上書き更新
- `google_event_id` をキーとしてupsert

## 7. Google Colab 構成

### 7.1 構成方針: Pythonモジュール + オーケストレーターノートブック

複数の`.ipynb`をColab上で自動連続実行することは標準機能では困難なため、
**処理ロジックをPythonモジュール（`.py`）としてGoogle Driveに配置**し、
**1つのオーケストレーター用ノートブックから順次呼び出す**構成を採用する。

```
Google Drive
└── weather-forecast/
    ├── main.ipynb                  # オーケストレーター（唯一のノートブック）
    └── src/
        ├── __init__.py
        ├── data_fetcher.py         # ERA5 / GFS / ECMWFデータ取得・前処理
        ├── ensemble_inference.py   # 初期値摂動生成 + 各モデル推論
        ├── meta_learner.py         # メタ学習モデルの学習・推論・天気変換
        ├── upload.py               # 予測結果のSupabaseアップロード
        └── config.py               # 定数・設定値の一元管理
```

### 7.2 オーケストレーターノートブック (`main.ipynb`)

```python
# Cell 1: Google Driveマウント & パス設定
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/MyDrive/weather-forecast')

# Cell 2: 依存ライブラリのインストール
!pip install graphcast earth2mip lightgbm optuna supabase xarray cfgrib

# Cell 3: パイプライン実行
from src.data_fetcher import fetch_all_data
from src.ensemble_inference import run_ensemble
from src.meta_learner import run_meta_learning
from src.upload import upload_to_supabase

# Step 1: データ取得
raw_data = fetch_all_data(target_date="2026-04-15", region="japan")

# Step 2: アンサンブル推論（3モデル × 50メンバー）
ensemble_results = run_ensemble(raw_data, n_members=50)

# Step 3: メタ学習統合 → 天気概況変換
forecasts = run_meta_learning(ensemble_results)

# Step 4: Supabaseアップロード
upload_to_supabase(forecasts)
```

### 7.3 各モジュールの役割

| モジュール | 役割 | 主な関数 |
|---|---|---|
| `data_fetcher.py` | ERA5 / GFS / ECMWFデータ取得・前処理 | `fetch_all_data()` |
| `ensemble_inference.py` | 初期値摂動生成 + 各モデル推論 | `run_ensemble()` |
| `meta_learner.py` | メタ学習 + 天気概況変換 | `run_meta_learning()` |
| `upload.py` | Supabaseへの結果送信 | `upload_to_supabase()` |
| `config.py` | API Key、リージョン、モデルパス等の設定 | - |

### 7.4 実行フロー

```
main.ipynb（セル順次実行 or 「すべてのセルを実行」）
    │
    ├─ fetch_all_data()          ... ERA5 + GFS/ECMWF取得・前処理
    │
    ├─ run_ensemble()            ... 摂動生成 → 3モデル推論
    │    ├─ GraphCast × 50
    │    ├─ FourCastNet × 50
    │    └─ MetNet-3 × 50
    │
    ├─ run_meta_learning()       ... 特徴量生成 → LightGBM/MLP → 天気変換
    │
    └─ upload_to_supabase()      ... Supabase REST API経由でINSERT
```

### 7.5 この構成のメリット
- **ワンクリック実行**: `main.ipynb`で「すべてのセルを実行」するだけで全パイプラインが動作
- **モジュール単体テスト可能**: 各`.py`をインポートして個別にデバッグ可能
- **コード管理**: `.py`ファイルはGit管理しやすい（`.ipynb`のdiffは読みにくい）
- **段階的デバッグ**: `main.ipynb`のセルを1つずつ実行して中間結果を確認可能

### 7.6 Colab環境要件
- **GPU**: A100 または T4（GraphCast推論に必要）
- **RAM**: 最低16GB（High-RAM ランタイム推奨）
- **ストレージ**: Google Drive マウント（中間データ保存）
- **主要ライブラリ**:
  - `graphcast` (DeepMind)
  - `earth2mip` (NVIDIA / FourCastNet)
  - `xarray`, `cfgrib` (気象データ処理)
  - `lightgbm`, `optuna` (メタ学習)
  - `torch` (MLP / FourCastNet)
  - `jax` (GraphCast)
  - `supabase-py` (DB連携)

## 8. API設計（Next.js API Routes）

### 8.1 エンドポイント一覧

| メソッド | パス | 説明 |
|---|---|---|
| GET | `/api/forecasts` | 予測結果取得（クエリ: date_from, date_to, location） |
| GET | `/api/forecasts/:id` | 予測結果詳細取得 |
| POST | `/api/calendar/sync` | Googleカレンダーへの同期実行 |
| GET | `/api/calendar/status` | 同期ステータス確認 |
| GET | `/api/auth/[...nextauth]` | NextAuth認証 |

### 8.2 `POST /api/calendar/sync` リクエスト

```json
{
  "forecast_ids": ["uuid-1", "uuid-2"],
  "calendar_id": "primary"
}
```

### 8.3 `POST /api/calendar/sync` レスポンス

```json
{
  "synced": 7,
  "failed": 0,
  "events": [
    {
      "forecast_id": "uuid-1",
      "google_event_id": "abc123",
      "status": "created"
    }
  ]
}
```

## 9. 非機能要件

### 9.1 パフォーマンス
- Colab上での全モデル推論: 目標2時間以内（A100使用時）
- Next.js API応答: 500ms以内
- カレンダー同期: 1日あたり1件ずつ逐次登録（Google API Rate Limit考慮）

### 9.2 セキュリティ
- Google OAuthトークンはサーバーサイドで管理（クライアントに露出しない）
- Supabase Row Level Security (RLS) によるユーザー別データ分離
- API KeyはSupabase環境変数で管理

### 9.3 制約事項
- Colabの無料枠ではGPU使用時間に制限あり（Colab Pro推奨）
- ERA5データのダウンロードにCDS APIキーが必要（無料登録）
- Google Calendar APIの日次クォータ: 1,000,000リクエスト/日
- MetNet-3はGoogle公式の公開重みが限定的なため、代替としてオープンソース実装または簡易版で対応する可能性あり

### 9.4 将来の拡張候補
- Pangu-Weatherモデルの追加
- LINE通知連携
- 予測精度のダッシュボード（実測値との事後比較）
- 定期実行のスケジューラ（Colab → GitHub Actions等）
