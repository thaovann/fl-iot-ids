"""Sinh notebooks/colab_fl_ids.ipynb (chạy lại script này sau khi sửa nội dung cell)."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# FL IDS trên CICIoMT2024 (Colab)

Notebook này chỉ là "vỏ": code nằm trong repo GitHub `thaovann/fl-iot-ids`, dữ liệu, cache cửa sổ và kết quả
nằm trên Google Drive. Mỗi lần mở notebook, cell 2 sẽ clone hoặc `git pull` bản code mới nhất.

Thứ tự: **1** mount Drive → **2** lấy code → **3** đặt tham số → **4** kiểm tra file dữ liệu →
**5** dựng cửa sổ (cache lên Drive, chỉ chạy lần đầu) → **6** chạy thí nghiệm → **7** xem bảng tổng hợp.

Nếu Colab ngắt giữa chừng: chạy lại từ cell 1. Cửa sổ nạp lại từ cache, và các dòng kết quả đã có
trong `final_summary.csv` được bỏ qua, nên thí nghiệm chạy tiếp từ chỗ dừng.
""")

md("## 1. Mount Google Drive")
code("""
from google.colab import drive
drive.mount('/content/drive')
""")

md("""
## 2. Lấy code từ GitHub

Repo **private**: vào biểu tượng chìa khoá (Secrets) bên trái Colab, thêm secret `GITHUB_TOKEN`
(token GitHub có quyền đọc repo) và bật "Notebook access". Repo public thì không cần.
`BRANCH = None` nghĩa là lấy nhánh mặc định của repo.
""")
code("""
import os, subprocess, sys

REPO = 'thaovann/fl-iot-ids'
BRANCH = None            # ví dụ 'main' hoặc tên nhánh đang thử
CODE_DIR = '/content/fl-iot-ids'

token = ''
try:
    from google.colab import userdata
    token = userdata.get('GITHUB_TOKEN') or ''
except Exception:
    pass
url = f'https://{token}@github.com/{REPO}.git' if token else f'https://github.com/{REPO}.git'

def sh(cmd):
    print('$', cmd.replace(token, '***') if token else cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    print(out.replace(token, '***') if token else out)
    if r.returncode: raise RuntimeError('lệnh lỗi')

if not os.path.exists(CODE_DIR):
    sh(f'git clone {"-b " + BRANCH if BRANCH else ""} {url} {CODE_DIR}')
else:
    if BRANCH:
        sh(f'cd {CODE_DIR} && git fetch origin {BRANCH} && git checkout {BRANCH}')
    sh(f'cd {CODE_DIR} && git pull --ff-only')
sh(f'cd {CODE_DIR} && git log -1 --oneline')
sh(f'pip install -q -r {CODE_DIR}/requirements.txt')
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

import importlib, fl_ids
for m in list(sys.modules):                     # nạp lại module nếu vừa git pull
    if m.startswith('fl_ids'):
        importlib.reload(sys.modules[m])
""")

md("""
## 3. Tham số

- `CONFIG`: `configs/ciciomt2024.yaml` (thí nghiệm chính), `configs/poisoning.yaml` (Phần III),
  hoặc `configs/smoke.yaml` (dữ liệu giả, chạy thử vài phút).
- `DATA_ROOT`: thư mục trên Drive chứa CSV CICFlowMeter, có `train/` và `test/` (hoặc tên file có `_train`/`_test`).
- `OVERRIDES`: ghi đè bất kỳ khoá nào trong config, dạng `khoá.con=giá_trị`.
""")
code("""
from fl_ids.config import load_config

CONFIG = 'configs/ciciomt2024.yaml'
DATA_ROOT = '/content/drive/MyDrive/CICIoMT2024/WiFI_and_MQTT/attacks/CSV'
DRIVE_WORK = '/content/drive/MyDrive/fl-iot-ids'      # cache + kết quả

OVERRIDES = [
    f'data.root={DATA_ROOT}',
    f'paths.cache_dir={DRIVE_WORK}/cache',
    f'paths.out_dir={DRIVE_WORK}/results',
    # Chạy thử nhỏ trước khi chạy đủ lưới:
    'seeds=[42]',
    'partition.schemes=[iid]',
]

USE_SYNTHETIC = False      # True: tự sinh dữ liệu giả để kiểm tra pipeline, không cần Drive có dữ liệu
if USE_SYNTHETIC:
    sh('python scripts/make_synthetic.py --out /content/synthetic')
    CONFIG = 'configs/smoke.yaml'
    OVERRIDES = ['data.root=/content/synthetic', f'paths.cache_dir={DRIVE_WORK}/cache',
                 f'paths.out_dir={DRIVE_WORK}/results']

cfg = load_config(CONFIG, OVERRIDES)
print('Thí nghiệm:', cfg['name'])
print('Partition:', cfg['partition']['schemes'], '| seeds:', cfg['seeds'])
print('Chọn đặc trưng:', cfg['fs']['methods'], '| k:', cfg['fs']['k_list'])
print('Mô hình:', cfg['models']['methods'])
""")

md("## 4. Kiểm tra file dữ liệu (nhãn, nhóm, phần, train/test theo tên file)")
code("""
import pandas as pd
from fl_ids.data import discover_files, plan_splits

files = discover_files(cfg['data']['root'], cfg['data']['glob'])
plan = plan_splits(files, cfg['split'])
tab = pd.DataFrame([{**f.__dict__, 'decision': plan.decisions[f.path]} for f in files]).drop(columns='path')
print(len(files), 'file')
tab
""")

md("""
## 5. Dựng cửa sổ trượt (lần đầu lâu, sau đó nạp từ cache trên Drive)

Cache đặt tên theo băm tham số `data` + `window`, nên đổi FW/FS hay cap số dòng sẽ tự tạo cache mới.
""")
code("""
from fl_ids.windowing import get_windows, window_counts

ws = get_windows(cfg)          # get_windows(cfg, force=True) để dựng lại
print(ws.X.shape, 'cửa sổ × đặc trưng')
window_counts(ws)
""")

md("## 6. Chạy thí nghiệm (kết quả ghi dần vào Drive)")
code("""
from fl_ids.experiment import run

csv_path = run(cfg, ws)
print('Kết quả:', csv_path)
""")

md("## 7. Bảng tổng hợp (mean ± std theo seed)")
code("""
from fl_ids.experiment import summarize

summ = summarize(csv_path)
summ.to_csv(str(csv_path).replace('final_summary.csv', 'summary_mean_std.csv'))
pd.set_option('display.max_rows', 300)
summ.round(4)
""")

md("""
## 8. (Tuỳ chọn) Đẩy thay đổi code từ Colab lên GitHub

Chỉ dùng khi bạn sửa code trong `/content/fl-iot-ids` và muốn lưu lại. Kết quả và dữ liệu **không** đưa lên Git
(đã có trong `.gitignore`); chúng nằm trên Drive. Cần `GITHUB_TOKEN` có quyền ghi.
""")
code("""
# sh('cd /content/fl-iot-ids && git config user.name "Van" && git config user.email "you@example.com"')
# sh('cd /content/fl-iot-ids && git checkout -b colab-edit && git add -A && git commit -m "Sửa từ Colab"')
# sh(f'cd /content/fl-iot-ids && git push {url} colab-edit')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []},
                                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_fl_ids.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã ghi", out)
