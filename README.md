# fl-iot-ids

Cửa sổ trượt + chọn đặc trưng liên kết (Fed-MI, Fed-χ², Fed-RFE, gộp hạng) + IDS liên kết
(FedXGB, FedAvg/FedProx-MLP) trên CICIoMT2024, chạy trên Google Colab.

Code nằm trên GitHub; **dữ liệu, cache cửa sổ và kết quả nằm trên Google Drive** (không đưa lên Git).

## Chạy trên Colab

1. Mở `notebooks/colab_fl_ids.ipynb` trên Colab: *File → Open notebook → GitHub*, nhập `thaovann/fl-iot-ids`.
   Nếu repo private, tick "Include private repos" và thêm secret `GITHUB_TOKEN` trong Colab (biểu tượng chìa khoá).
2. Để CSV CICFlowMeter của CICIoMT2024 trên Drive, ví dụ
   `MyDrive/CICIoMT2024/WiFI_and_MQTT/attacks/CSV/{train,test}/*.csv`, rồi sửa `DATA_ROOT` ở cell 3.
3. Chạy lần lượt các cell. Lần đầu thử với `seeds=[42]`, `partition.schemes=[iid]` (đã đặt sẵn), hoặc
   `USE_SYNTHETIC = True` để kiểm tra pipeline bằng dữ liệu giả.

Kết quả ở `MyDrive/fl-iot-ids/results/<name>/`:

| File | Nội dung |
|---|---|
| `final_summary.csv` | Một dòng cho mỗi partition × seed × m_malicious × fs_method × k × model (macro-F1, MCC, FPR, recall Attack, AUC-PR, recall theo nhóm, τ, Comm_MB, thời gian) |
| `summary_mean_std.csv` | mean ± std theo seed |
| `rankings.jsonl` | Thứ hạng đặc trưng của từng phương pháp chọn, kèm tên top-100 và byte truyền |
| `window_counts.csv` | Số cửa sổ theo nhóm × kiểu tấn công × split |
| `clients_<partition>_s<seed>.csv` | Số cửa sổ mỗi lớp theo client |
| `config.json` | Cấu hình đã chạy |

Colab ngắt giữa chừng thì chạy lại notebook: cửa sổ nạp từ cache, dòng kết quả đã có được bỏ qua.

## Chạy từ dòng lệnh (máy riêng hoặc ô `!` trong Colab)

```bash
pip install -r requirements.txt          # thêm torch nếu dùng MLP
python scripts/make_synthetic.py --out data/synthetic
python scripts/run_experiment.py --config configs/smoke.yaml
python scripts/run_experiment.py --config configs/ciciomt2024.yaml \
    --set data.root=/path/to/CSV seeds=[42] "partition.schemes=[iid]"
python -m pytest -q tests
```

## Cấu trúc

```
fl_ids/
  config.py             tham số mặc định + nạp YAML + ghi đè "khoá.con=giá_trị"
  data.py               đọc CSV, nhãn theo tên file, sắp theo Timestamp, chia train/val/test theo file, chia khối
  windowing.py          cửa sổ trượt FW/FS + 5 thống kê, cache .npz trên Drive
  partition.py          chia client theo khối: iid, dirichlet:<α>, filepart
  preprocess.py         bỏ cột hằng + FedScaler (gộp Chan) + trọng số lớp, đều tính liên kết
  feature_selection.py  Fed-MI (bảng đếm), Fed-χ² (tổng theo lớp), Fed-MI→RFE, rank_<scorer>_<agg>,
                        scoresum_<scorer>, random, central_*; đầu độc client (promote_worst, score_dominate, inflate_n)
  models.py             central_xgb, fedxgb_bag, fedxgb_cyclic, fedavg_mlp, fedprox_mlp:<μ>, central_mlp, local_xgb
  metrics.py            độ đo nhị phân, chọn ngưỡng τ trên val liên kết
  experiment.py         vòng lặp thí nghiệm, ghi kết quả dần, resume
configs/                ciciomt2024.yaml (chính), poisoning.yaml (Phần III), smoke.yaml (dữ liệu giả)
notebooks/              colab_fl_ids.ipynb (sinh từ scripts/build_notebook.py)
scripts/                run_experiment.py, make_synthetic.py, build_notebook.py
tests/                  kiểm tra nhanh (pytest)
```

## Quy ước chính (theo `methodology-FL-CICIoMT2024.md`)

- Bài toán nhị phân; nhãn theo tên file (`Benign*` = 0). `group`, `attack_type` chỉ là metadata.
- Sắp theo `Timestamp` trong file, rồi mới bỏ Flow ID, IP, Port, Timestamp. Giữ `Protocol`
  (ablation: `--set "data.drop_features=[Protocol]"`).
- Test = file `*_test`. Val: kiểu tấn công nhiều phần lấy phần số lớn nhất; kiểu một file lấy 20% cuối theo
  thời gian, đệm FW dòng. Có thể giữ trọn một kiểu con làm val bằng `split.val_holdout`.
- Cửa sổ chỉ cắt trong một khối (`block_size` dòng liên tiếp của một file), nên train/val/test không chung dòng.
- `assert_no_metadata` dừng chương trình nếu cột định danh (Window_Index, *_id, timestamp…) lọt vào đặc trưng.
- Tiền xử lý không bao giờ dùng train gộp: min/max, (n, mean, M2) và số cửa sổ mỗi lớp đều gộp từ client.
- Vòng tốt nhất và ngưỡng τ chọn trên val liên kết (cộng ma trận nhầm lẫn từ các client); test chỉ để báo cáo.
