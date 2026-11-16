# Validation Error Analysis

This report analyzes the frozen validation configuration. The test split
remains untouched.

## Detector confusion matrix

- True positive: 118.
- False positive: 575.
- False negative: 17.
- True negative: 5426.

## Correction outcomes

- Correct corrections: 54.
- Wrong corrections on real error positions: 2.
- Detected errors kept unchanged: 62.
- Missed error positions: 17.
- False corrections on originally correct tokens: 35.
- Detector false positives safely kept: 540.

## Results by synthetic error type

- **delete** — total 35; detection recall 71.43%; Top-3 recall 42.86%; correction recall 25.71%.
- **diacritic** — total 13; detection recall 69.23%; Top-3 recall 69.23%; correction recall 30.77%.
- **insert** — total 40; detection recall 92.50%; Top-3 recall 72.50%; correction recall 42.50%.
- **substitute** — total 21; detection recall 100.00%; Top-3 recall 57.14%; correction recall 42.86%.
- **transpose** — total 26; detection recall 100.00%; Top-3 recall 84.62%; correction recall 57.69%.

## Representative examples

### Correct Correction

- `nạm` → `nam` (target `nam`, confidence 0.999, context: “Đội tuyển bóng chuyền nạm quốc gia Thổ Nhĩ”).
- `ội` → `đội` (target `đội`, confidence 1.000, context: “sách các thành viên ội tuyển nam quốc gia”).
- `gnữ` → `nữ` (target `nữ`, confidence 1.000, context: “Đội tuyển bóng chuyền gnữ quốc gia Thổ Nhĩ”).
- `chữc` → `chức` (target `chức`, confidence 1.000, context: “Kuala Lumpur sẽ tổ chữc từ ngày 23 tháng”).
- `chuyêền` → `chuyền` (target `chuyền`, confidence 1.000, context: “Giải bóng chuyêền vô địch châu Âu”).

### Wrong Correction

- `lt` → `là` (target `lụt`, confidence 0.999, context: “bởi lở đất và lt lội .”).
- `VB` → `VÀ` (target `DVB`, confidence 0.998, context: “88 và phủ sóng VB - T2”).

### Detected Kept

- `Jošako` → `Jošako` (target `Joško`, confidence 1.000, context: “luyện viên chính : Jošako Milenkoski”).
- `Basicd` → `Basicd` (target `Basic`, confidence 1.000, context: “2010 – 13 Veljko Basicd”).
- `nữh` → `nữh` (target `nữ`, confidence 1.000, context: “Giải đấu đồng đội nữh”).
- `chop` → `chop` (target `cho`, confidence 1.000, context: “mục chính tòa " chop Hàng Châu , được”).
- `Dxương` → `Dxương` (target `Dương`, confidence 1.000, context: “Giám mục Dxương sinh ngày 11 tháng”).

### Missed Detection

- `Yeong` → `Yeong` (target `Yong`, confidence n/a, context: “( , Joseph Yang Yeong - qiang ; sinh”).
- `hominids` → `hominids` (target `hominis`, confidence n/a, context: “giám mục là Filius hominids non venit ministrari sed”).
- `ánh` → `ánh` (target `anh`, confidence n/a, context: “giả ( sau khi ánh ta đã bị giết”).
- `bố` → `bố` (target `bộ`, confidence n/a, context: “Vào năm 1980 , bố phim Airplane !”).
- `cản` → `cản` (target `cảnh`, confidence n/a, context: “) cũng nhại lại cản hậu danh đề kinh”).

### False Correction

- `Thổ` → `Thi` (target `Thổ`, confidence 0.998, context: “chuyền nạm quốc gia Thổ Nhĩ Kỳ là đội”).
- `Thổ` → `Thi` (target `Thổ`, confidence 1.000, context: “tuyển nam quốc gia Thổ Nhĩ Kỳ tham dự”).
- `Thổ` → `Thi` (target `Thổ`, confidence 1.000, context: “chuyền gnữ quốc gia Thổ Nhĩ Kỳ”).
- `Andrej` → `Andrea` (target `Andrej`, confidence 0.991, context: “luyện viêtn chính : Andrej Kravarik”).
- `Ngay` → `Ngày` (target `Ngay`, confidence 1.000, context: “Ngay cả khi các cảnh”).

### False Positive Kept

- `Thổ` → `Thổ` (target `Thổ`, confidence 1.000, context: “bóng đại diện cho Thổ Nhĩ Kỳ tại các”).
- `Milenkoski` → `Milenkoski` (target `Milenkoski`, confidence 1.000, context: “viên chính : Jošako Milenkoski”).
- `Kosta` → `Kosta` (target `Kosta`, confidence 1.000, context: “1970 – 71 Kosta Shapov”).
- `Shapov` → `Shapov` (target `Shapov`, confidence 1.000, context: “1970 – 71 Kosta Shapov”).
- `08` → `08` (target `08`, confidence 0.998, context: “2006 – 08 Gennady Parshin”).

## Figures

- `reports/figures/viwiki-20261101-smoke-dedup-detector-confusion.png`.
- `reports/figures/viwiki-20261101-smoke-dedup-correction-outcomes.png`.
