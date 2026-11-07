# Đồ án sửa lỗi chính tả tiếng Việt — Bản bàn giao dự án

## 1. Tóm tắt

Dự án xây dựng một hệ thống **phát hiện và gợi ý sửa lỗi chính tả tiếng Việt**. Toàn bộ thành phần học máy và mô hình ngôn ngữ phải được nhóm tự xây dựng, tự huấn luyện từ dữ liệu được phép sử dụng và có thể giải trình đầy đủ.

Phạm vi được chốt cho MVP là sửa lỗi ở **cấp độ từ** trong câu tiếng Việt đã được tách từ, ưu tiên các lỗi đánh máy và lỗi chính tả phổ biến. Hệ thống không nhằm thay thế mô hình sinh văn bản lớn, không xử lý mọi lỗi ngữ pháp, và không cam kết hiểu ngữ nghĩa sâu.

Kiến trúc MVP:

1. **BiLSTM detector** tự huấn luyện để ước lượng từ/vị trí nào có khả năng sai trong ngữ cảnh.
2. **Edit Distance candidate generator** sinh các từ ứng viên từ từ điển của tập huấn luyện.
3. **Trigram language model** tự học bằng thống kê để chấm mức phù hợp ngữ cảnh của từng ứng viên.
4. Kết hợp điểm detector, khoảng cách chỉnh sửa, tần suất từ và xác suất trigram để xếp hạng gợi ý.

## 2. Mục tiêu

### 2.1. Mục tiêu chính

- Tự xây dựng một pipeline NLP hoàn chỉnh: thu thập/chuẩn hóa dữ liệu, sinh lỗi, huấn luyện, đánh giá, suy luận và demo.
- Phát hiện được vị trí từ có khả năng sai chính tả trong câu tiếng Việt.
- Sinh và xếp hạng được các phương án sửa hợp lý, trả về ít nhất Top-1 và Top-3.
- Chứng minh kết quả bằng tập test tách biệt, metric rõ ràng và thí nghiệm có thể tái lập.
- Có giao diện Streamlit để nhập câu, xem từ bị nghi sai, các ứng viên và câu sau khi sửa.

### 2.2. Mục tiêu học thuật

- Hiểu và triển khai BiLSTM bằng PyTorch thay vì gọi mô hình có sẵn.
- Hiểu vai trò của ngữ cảnh thông qua mô hình ngôn ngữ trigram thống kê.
- So sánh với baseline đơn giản để chứng minh giá trị của từng thành phần.
- Thực hiện ablation tối thiểu để đo đóng góp của detector và language model.

### 2.3. Ngoài phạm vi MVP

- Sửa ngữ pháp tổng quát, diễn đạt, văn phong hoặc dấu câu phức tạp.
- Sửa lỗi khiến từ sai vẫn là một từ hợp lệ nhưng cần hiểu ngữ nghĩa sâu trong mọi trường hợp.
- Mô hình Transformer, LLM, mô hình sinh, mô hình dịch máy hoặc mô hình pretrained.
- Hỗ trợ thời gian thực ở quy mô sản phẩm lớn.
- Cam kết xử lý hoàn hảo tên riêng, tiếng lóng, mã trộn ngôn ngữ và phương ngữ.

## 3. Ràng buộc tuyệt đối

Đây là các điều kiện không được vi phạm:

- **Train from scratch**: mọi trọng số của BiLSTM phải khởi tạo ngẫu nhiên và được huấn luyện bởi nhóm.
- **Không dùng pretrained** dưới mọi hình thức: không dùng BERT, PhoBERT, viBERT, mBERT, fastText pretrained, word2vec pretrained, GloVe pretrained, pretrained tokenizer/model checkpoint hoặc embedding tải sẵn.
- **Không gọi API AI** để tạo kết quả trong pipeline: không dùng API ChatGPT/OpenAI, Gemini, Claude, Copilot, dịch máy hay dịch vụ sửa chính tả/ngữ pháp bên ngoài.
- Không dùng mô hình đóng hoặc thư viện cung cấp sẵn một bộ sửa lỗi hoàn chỉnh như “hộp đen”.
- Có thể dùng PyTorch và các thư viện xử lý dữ liệu thông thường, nhưng phải tự định nghĩa kiến trúc, loss, vòng lặp train/evaluate và logic suy luận.
- Trigram LM phải được đếm và ước lượng từ chính tập train; từ điển và tần suất từ cũng chỉ được xây từ train.
- Validation và test không được tham gia xây từ điển, sinh thống kê, chọn embedding, fit tokenizer/vocabulary hay điều chỉnh trọng số.
- Mọi nguồn dữ liệu phải được ghi lại nguồn gốc, giấy phép/quyền sử dụng và bước tiền xử lý.

AI có thể được dùng như công cụ hỗ trợ lập trình trong quá trình phát triển nếu quy định môn học cho phép, nhưng **không được là một thành phần chạy trong sản phẩm hoặc tạo nhãn/kết quả không thể giải trình**. Khi chưa rõ, phải hỏi giảng viên và ghi nhận câu trả lời.

## 4. Định nghĩa bài toán và MVP

### 4.1. Đầu vào và đầu ra

Đầu vào: một câu tiếng Việt dạng chuỗi.

Đầu ra:

- Danh sách vị trí/từ bị nghi là sai cùng xác suất hoặc confidence.
- Tối đa `k` ứng viên sửa cho mỗi vị trí, mặc định `k = 3`.
- Câu được sửa tự động theo phương án Top-1, đồng thời cho phép người dùng chọn ứng viên khác.
- Thông tin giải thích cơ bản: khoảng cách edit, tần suất ứng viên và điểm ngữ cảnh.

### 4.2. Các loại lỗi ưu tiên

- Thiếu/thừa/thay một ký tự.
- Hoán đổi hai ký tự kề nhau.
- Lỗi dấu thanh hoặc dấu chữ cái tiếng Việt.
- Nhầm phụ âm/vần phổ biến nếu có bảng nhầm lẫn được nhóm định nghĩa và kiểm chứng.
- Lỗi dính/tách từ chỉ là mục tiêu mở rộng; không đưa vào tiêu chí bắt buộc của MVP nếu tokenizer cấp từ làm bài toán quá phức tạp.

### 4.3. Đơn vị xử lý

MVP xử lý chuỗi thành **token từ**. Cần giữ ánh xạ từ token về khoảng ký tự trong câu gốc để giao diện tô sáng đúng vị trí. Dấu câu được tách thành token riêng hoặc xử lý nhất quán trong toàn bộ pipeline.

## 5. Kiến trúc đã chốt

### 5.1. BiLSTM detector

Nhiệm vụ: gán nhãn nhị phân cho từng token: `CORRECT` hoặc `ERROR`.

Kiến trúc tối thiểu:

```text
token ids
  -> Embedding khởi tạo ngẫu nhiên
  -> BiLSTM (1–2 tầng)
  -> Dropout
  -> Linear theo từng time step
  -> logits [batch, seq_len, 2]
```

Gợi ý cấu hình khởi đầu, phải được chọn bằng validation:

- Vocabulary: chỉ fit trên tập train, có `<PAD>` và `<UNK>`.
- Embedding dimension: 128.
- Hidden size mỗi chiều: 128.
- Số tầng BiLSTM: 1.
- Dropout: 0.2–0.4.
- Loss: weighted cross-entropy hoặc focal loss nếu nhãn lỗi quá mất cân bằng.
- Optimizer: Adam, learning rate khởi đầu `1e-3`.
- Gradient clipping: `1.0`.
- Early stopping theo validation F1 của lớp `ERROR`.

Không nạp embedding bên ngoài. Embedding được học cùng detector từ dữ liệu train.

### 5.2. Edit Distance candidate generator

Nhiệm vụ: với token bị detector đánh dấu, tìm các từ hợp lệ gần nhất trong từ điển train.

Yêu cầu:

- Từ điển lấy **chỉ từ tập train sạch** và lưu kèm tần suất.
- Dùng Levenshtein distance; có thể mở rộng Damerau–Levenshtein để tính hoán đổi ký tự.
- Giới hạn khoảng cách theo độ dài token, ví dụ `1` với từ ngắn và tối đa `2` với từ dài.
- Lọc theo chữ cái đầu, độ dài, âm tiết hoặc bảng nhầm lẫn để giảm không gian tìm kiếm.
- Không lấy từ/cụm từ từ API hoặc từ điển online tại thời điểm suy luận.
- Nếu không có ứng viên đủ tốt, giữ nguyên từ và đánh dấu “không chắc chắn”.

Điểm ứng viên ban đầu có thể gồm:

```text
edit_score(c, x) = - normalized_edit_distance(c, x)
frequency_score(c) = log(1 + count_train(c))
```

### 5.3. Trigram language model tự học thống kê

Nhiệm vụ: đo ứng viên nào phù hợp nhất với ngữ cảnh câu.

Mô hình tính xác suất:

```text
P(w_i | w_{i-2}, w_{i-1})
```

Yêu cầu triển khai:

- Đếm unigram, bigram và trigram chỉ trên corpus train sạch.
- Thêm token biên câu `<s>` và `</s>`.
- Có smoothing để tránh xác suất bằng 0. MVP có thể dùng add-k; phiên bản tốt hơn dùng linear interpolation giữa trigram, bigram và unigram.
- Tham số smoothing/interpolation được chọn trên validation, không chọn trên test.
- Khi thay ứng viên ở vị trí `i`, tính lại log-probability của cửa sổ ngữ cảnh bị ảnh hưởng.

Ví dụ điểm nội suy:

```text
P_interp = λ3 * P_trigram + λ2 * P_bigram + λ1 * P_unigram
λ1 + λ2 + λ3 = 1
```

### 5.4. Xếp hạng cuối

Một công thức MVP dễ giải trình:

```text
score(c) = α * context_log_prob(c)
         + β * edit_score(c, x)
         + γ * frequency_score(c)
```

Trong đó `α, β, γ` được tune bằng validation. Detector quyết định vị trí cần xét; không nên dùng detector confidence như điểm phân biệt ứng viên tại cùng một vị trí trừ khi có thiết kế rõ ràng.

Nếu nhiều lỗi trong một câu, MVP có thể sửa tuần tự từ trái sang phải, mỗi bước cập nhật câu. Hướng mở rộng là beam search qua nhiều vị trí lỗi.

## 6. Pipeline dữ liệu

### 6.1. Dữ liệu sạch

Ưu tiên corpus tiếng Việt có nguồn và quyền sử dụng rõ ràng. Mỗi tài liệu cần metadata tối thiểu: `document_id`, nguồn, loại văn bản và nội dung. Không đưa nguyên văn dữ liệu có hạn chế bản quyền vào repository.

Tiền xử lý đề xuất:

1. Chuẩn hóa Unicode về NFC.
2. Loại hoặc đánh dấu URL, email, emoji, ký tự điều khiển theo quy tắc cố định.
3. Tách câu.
4. Tokenize nhất quán.
5. Lọc câu quá ngắn/quá dài và câu có tỷ lệ ký tự bất thường cao.
6. Giữ bản sạch làm nhãn đích.

Không nên xóa dấu tiếng Việt hoặc lowercase mù quáng. Quyết định về viết hoa phải nhất quán giữa train và inference.

### 6.2. Sinh lỗi tổng hợp

Vì dữ liệu cặp sai–đúng thường ít, MVP có thể tạo lỗi từ câu sạch:

- Chọn token hợp lệ có độ dài phù hợp.
- Áp dụng một phép biến đổi: xóa, thêm, thay, hoán đổi ký tự hoặc biến đổi dấu.
- Không biến đổi token thành chính nó.
- Ghi lại `clean_token`, `noisy_token`, loại lỗi, vị trí và seed.
- Kiểm soát tỷ lệ câu không lỗi và số lỗi/câu để gần tình huống thực tế.

Khởi đầu hợp lý: 50–70% câu có ít nhất một lỗi, chủ yếu một lỗi/câu. Sau đó đánh giá thêm trên tập có phân phối gần thực tế hơn.

### 6.3. Dữ liệu lỗi thật

Nếu thu thập được, tạo một tập lỗi thật nhỏ do người viết hoặc người gán nhãn cung cấp. Dùng tập này làm test bổ sung để đo khoảng cách giữa lỗi tổng hợp và thực tế. Không được dùng chính test lỗi thật để điều chỉnh mô hình.

## 7. Chia train/validation/test và chống leakage

Nguyên tắc quan trọng nhất: **chia dữ liệu sạch trước, sau đó mới sinh lỗi riêng cho từng split**.

Quy trình:

1. Chuẩn hóa và gán `document_id` cho dữ liệu sạch.
2. Loại trùng lặp chính xác và gần trùng trước khi chia.
3. Chia theo tài liệu/nguồn hoặc tác giả, không chia ngẫu nhiên theo từng câu nếu các câu cùng tài liệu có thể rơi vào nhiều split.
4. Tỷ lệ đề xuất: train 80%, validation 10%, test 10%.
5. Sinh lỗi độc lập trên từng split bằng seed cố định khác nhau.
6. Fit vocabulary, từ điển, tần suất và mọi n-gram **chỉ trên train**.
7. Dùng validation để chọn hyperparameter, threshold, smoothing và trọng số ranker.
8. Chỉ chạy test khi thiết kế đã khóa; không sửa mô hình dựa trên test.

Kiểm tra leakage bắt buộc:

- Không có `document_id` xuất hiện ở nhiều split.
- Không có câu chuẩn hóa giống nhau ở nhiều split.
- Báo cáo tỷ lệ near-duplicate giữa các split nếu có.
- Không cache tokenizer/vocabulary từ toàn bộ corpus.
- Tập lỗi sinh từ một câu sạch không được rơi sang split khác với câu đó.

## 8. Chức năng MVP

- Nhập hoặc dán một câu tiếng Việt.
- Tiền xử lý và tokenize đúng như lúc train.
- Tô sáng token bị nghi sai.
- Hiển thị detector confidence.
- Sinh tối đa Top-3 ứng viên cho mỗi lỗi.
- Hiển thị điểm tổng và các thành phần điểm cơ bản.
- Tạo câu sửa Top-1.
- Cho phép người dùng chọn ứng viên khác hoặc giữ nguyên.
- Có chế độ ví dụ mẫu.
- Có trang/thẻ hiển thị metric và giới hạn của mô hình.
- Nạp model artifact cục bộ; không cần Internet khi chạy demo.

## 9. Metric và thiết kế đánh giá

### 9.1. Detector

Báo cáo ở cấp token, tập trung vào lớp `ERROR`:

- Precision.
- Recall.
- F1-score — metric chính.
- Confusion matrix.
- Có thể thêm PR-AUC nếu confidence có ý nghĩa.

Không dùng accuracy làm metric chính vì số token đúng thường áp đảo.

### 9.2. Candidate generator

- Candidate Recall@1, Recall@3, Recall@5: từ đúng có nằm trong danh sách hay không.
- Số ứng viên trung bình mỗi token.
- Thời gian sinh ứng viên trung bình.
- Tỷ lệ không sinh được ứng viên.

### 9.3. Ranker và hệ thống đầu-cuối

- Correction Accuracy@1 trên các vị trí lỗi đã biết.
- Correction Accuracy@3.
- End-to-end token correction precision/recall/F1 khi detector tự chọn vị trí.
- Sentence correction accuracy: tỷ lệ câu sau sửa khớp hoàn toàn câu sạch.
- False correction rate: tỷ lệ từ vốn đúng bị sửa sai.
- Latency trung vị hoặc p95 trên CPU/Colab.

### 9.4. Baseline và ablation

Baseline tối thiểu:

1. Chọn ứng viên có edit distance nhỏ nhất, phá hòa bằng tần suất.
2. Candidate generator + trigram, không dùng BiLSTM khi vị trí lỗi đã biết.

Ablation đề xuất:

- Bỏ trigram để đo vai trò ngữ cảnh.
- Bỏ frequency score.
- So sánh unigram/bigram/trigram.
- So sánh weighted cross-entropy với cross-entropy thường cho detector.

### 9.5. Báo cáo theo loại lỗi

Tách metric theo: xóa, thêm, thay, hoán đổi, lỗi dấu, từ ngắn/dài, từ OOV và câu có một/nhiều lỗi. Điều này giúp giải thích điểm yếu thay vì chỉ đưa một con số tổng.

## 10. Stack công nghệ

- **Google Colab**: môi trường thí nghiệm và huấn luyện có GPU khi cần.
- **Python 3.10+**.
- **PyTorch**: tự định nghĩa dataset, model, loss, optimizer và training loop.
- **pandas / NumPy**: xử lý và phân tích dữ liệu.
- **scikit-learn**: metric, confusion matrix và tiện ích đánh giá; không dùng mô hình pretrained.
- **Streamlit**: giao diện demo.
- **Matplotlib/Seaborn**: biểu đồ báo cáo.
- **pytest**: test cho preprocessing, edit distance, n-gram và inference.
- **Git/GitHub** nếu được phép: quản lý phiên bản; tuyệt đối không commit dữ liệu nhạy cảm hoặc checkpoint quá lớn.

Mọi dependency phải được khóa trong `requirements.txt` hoặc `pyproject.toml`. Demo phải chạy được bằng checkpoint cục bộ đã train bởi nhóm.

## 11. Cấu trúc repository đề xuất

```text
vietnamese-spell-correction/
├── README.md
├── AGENT_HANDOFF.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── detector.yaml
│   └── ranking.yaml
├── data/
│   ├── README.md
│   ├── raw/                 # không commit nếu bị hạn chế
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_train_detector.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── normalize.py
│   │   ├── split.py
│   │   ├── noise.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── bilstm_detector.py
│   │   └── trigram_lm.py
│   ├── correction/
│   │   ├── edit_distance.py
│   │   ├── candidates.py
│   │   └── ranker.py
│   ├── train_detector.py
│   ├── evaluate.py
│   └── inference.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_normalize.py
│   ├── test_split.py
│   ├── test_noise.py
│   ├── test_candidates.py
│   ├── test_trigram_lm.py
│   └── test_inference.py
├── artifacts/
│   ├── README.md
│   ├── vocabulary.json
│   ├── trigram_counts.*
│   └── detector.pt
├── reports/
│   ├── figures/
│   └── final_report.md
└── scripts/
    ├── prepare_data.py
    ├── train.py
    └── run_demo.py
```

Notebook dùng để khám phá và trình bày; logic cốt lõi phải nằm trong `src/` để test và tái sử dụng được.

## 12. Roadmap 5 tuần

### Tuần 1 — Chốt dữ liệu và nền móng

- Xác minh quy định môn học và nguồn dữ liệu được phép.
- Viết data card ngắn: nguồn, giấy phép, quy mô, rủi ro.
- Xây normalization, sentence splitting, tokenization và kiểm tra thống kê.
- Chia dữ liệu theo document trước khi sinh lỗi.
- Cài seed, logging, config và cấu trúc repository.
- Hoàn thành baseline edit distance + tần suất đầu tiên.

**Mốc bàn giao:** pipeline dữ liệu nhỏ chạy end-to-end, test chống leakage qua.

### Tuần 2 — Noise generator và trigram LM

- Cài các phép sinh lỗi và lưu metadata loại lỗi.
- Kiểm tra thủ công mẫu lỗi, tránh lỗi không hợp lệ/quá phi thực tế.
- Xây unigram/bigram/trigram counts và smoothing.
- Xây candidate generator có cache/index đơn giản.
- Đánh giá candidate Recall@k và baseline ranking.

**Mốc bàn giao:** khi biết trước vị trí lỗi, từ đúng xuất hiện trong Top-k với tỷ lệ đủ dùng.

### Tuần 3 — BiLSTM detector

- Xây vocabulary chỉ từ train và PyTorch Dataset/DataLoader.
- Cài BiLSTM detector, masking padding, weighted loss và training loop.
- Theo dõi train/validation loss, precision, recall, F1.
- Lưu best checkpoint theo validation F1.
- Error analysis theo loại lỗi và độ dài từ.

**Mốc bàn giao:** detector tốt hơn baseline từ điển/OOV trên validation và tái lập được bằng seed.

### Tuần 4 — Tích hợp và đánh giá

- Tích hợp detector → candidates → trigram ranker.
- Tune threshold và `α, β, γ` trên validation.
- Chạy baseline, ablation và đánh giá end-to-end.
- Kiểm tra latency, false correction và trường hợp không chắc chắn.
- Khóa thiết kế, sau đó chạy test chính thức một lần.

**Mốc bàn giao:** bảng metric chính, error analysis và artifact mô hình đã khóa.

### Tuần 5 — Demo, báo cáo và diễn tập

- Xây Streamlit UI và các câu ví dụ có/không có lỗi.
- Hoàn thiện README chạy lại thí nghiệm và demo.
- Viết báo cáo: bài toán, dữ liệu, mô hình, metric, kết quả, giới hạn.
- Kiểm tra trên máy/môi trường sạch, không Internet.
- Chuẩn bị slide, phân chia phần trình bày và câu hỏi phản biện.

**Mốc bàn giao:** demo chạy ổn định, báo cáo đầy đủ và toàn bộ kết quả có thể truy vết.

## 13. Rủi ro và cách giảm thiểu

### Dữ liệu lỗi tổng hợp khác lỗi thật

- Giảm thiểu: thiết kế noise dựa trên thống kê/quan sát thực tế, gán nhãn một tập lỗi thật nhỏ, báo cáo riêng kết quả synthetic và real-world.

### Mất cân bằng nhãn detector

- Giảm thiểu: weighted loss/focal loss, sampling hợp lý, threshold theo validation và dùng F1 lớp lỗi thay cho accuracy.

### Từ đúng không có trong từ điển train

- Giảm thiểu: báo cáo OOV, không ép sửa khi confidence thấp, mở rộng corpus hợp lệ thay vì lấy dữ liệu từ test.

### Candidate generation chậm

- Giảm thiểu: chia index theo độ dài/chữ đầu, cache kết quả, giới hạn edit distance và benchmark latency.

### Trigram thưa và không hiểu ngữ nghĩa xa

- Giảm thiểu: smoothing/interpolation, phân tích giới hạn rõ ràng, không tuyên bố khả năng ngoài phạm vi.

### Detector sửa nhầm từ đúng

- Giảm thiểu: tune threshold theo false correction rate, yêu cầu margin điểm đủ lớn trước khi tự sửa, cho phép “giữ nguyên”.

### Leakage do sinh lỗi trước khi chia

- Giảm thiểu: khóa pipeline chia theo document trước; có test tự động kiểm tra document/câu trùng giữa split.

### Vi phạm yêu cầu “tự xây dựng”

- Giảm thiểu: lưu log dependency và provenance, không dùng checkpoint/embedding pretrained, mô tả rõ mọi thuật toán, xin xác nhận giảng viên nếu có điểm mơ hồ.

### Thiếu thời gian/GPU

- Giảm thiểu: giữ BiLSTM nhỏ, bắt đầu từ subset, early stopping, cache preprocessing và ưu tiên MVP trước các phần mở rộng.

## 14. Tiêu chí nghiệm thu

Đồ án được xem là đạt MVP khi thỏa toàn bộ điều kiện sau:

### Tuân thủ

- Có bằng chứng BiLSTM được khởi tạo và train from scratch.
- Không có pretrained model/embedding/tokenizer và không có API AI trong runtime.
- Có danh sách nguồn dữ liệu, giấy phép/quyền sử dụng và dependency.

### Dữ liệu

- Pipeline chuẩn hóa, chia split và sinh lỗi chạy lại được bằng config/seed.
- Không có document hoặc câu trùng chính xác giữa train/validation/test.
- Vocabulary, từ điển và n-gram chỉ fit từ train.

### Mô hình

- Detector BiLSTM train/evaluate được, có best checkpoint và log metric.
- Candidate generator dùng edit distance và đo được Recall@k.
- Trigram LM có smoothing, có unit test xác suất/score cơ bản.
- Ranker kết hợp ngữ cảnh, edit distance và tần suất với trọng số chốt từ validation.

### Đánh giá

- Báo cáo precision/recall/F1 của detector, Candidate Recall@k, correction accuracy và false correction rate.
- Có ít nhất một baseline và một ablation.
- Có error analysis theo loại lỗi, ví dụ thành công/thất bại và giới hạn.
- Test chỉ được dùng sau khi khóa thiết kế.

### Sản phẩm

- Streamlit demo chạy cục bộ từ lệnh được ghi trong README dự án.
- Demo nhận câu, tô lỗi, đưa Top-3 và sinh câu sửa Top-1.
- Có cơ chế giữ nguyên khi không chắc chắn.
- Demo chạy không cần Internet sau khi đã có artifact.

### Tái lập

- Một thành viên khác có thể cài dependency, chuẩn bị dữ liệu, train/evaluate và chạy demo theo tài liệu.
- Config, seed, phiên bản code và artifact dùng trong báo cáo được ghi lại.

Ngưỡng số cụ thể (ví dụ F1 tối thiểu) chỉ nên chốt sau khi có baseline tuần 1–2. Một mục tiêu nội bộ hợp lý là hệ thống cuối phải cải thiện rõ ràng so với baseline edit-distance-only trên validation và test; không đặt ngưỡng tùy ý rồi vô tình tối ưu vào test.

## 15. Thứ tự ưu tiên khi phải cắt phạm vi

1. Giữ pipeline dữ liệu sạch và chống leakage.
2. Giữ detector BiLSTM, edit-distance candidates và trigram ranker.
3. Giữ metric, baseline, test và khả năng tái lập.
4. Giữ Streamlit demo tối giản.
5. Cắt beam search, lỗi dính/tách từ, dashboard đẹp và tối ưu nâng cao trước.

## 16. Quy ước làm việc đề xuất

- Mỗi thí nghiệm có config, seed, mã commit và thư mục kết quả riêng.
- Không chỉnh tay file test hoặc metric sau khi chạy.
- Viết unit test ngay cho normalization, split, noise, edit distance và n-gram.
- Mỗi quyết định kỹ thuật ghi ngắn vào `reports/decision_log.md`.
- Không commit dữ liệu gốc/checkpoint nếu giấy phép hoặc dung lượng không phù hợp.
- Trước khi thêm thư viện hoặc tài nguyên, kiểm tra nó có chứa mô hình/embedding pretrained hay không.

## 17. Trạng thái triển khai

### Mốc 1 — Nền tảng pipeline dữ liệu

Đã triển khai:

- Cấu hình split, seed và sinh lỗi trong `configs/data.yaml`.
- Chuẩn hóa Unicode NFC và tokenization có ánh xạ character span.
- Chia train/validation/test theo `document_id` với kiểm tra leakage.
- Sinh lỗi token có seed, metadata và các phép xóa, thêm, thay, đảo ký tự, biến đổi dấu.
- Unit test và smoke test cho pipeline dữ liệu giả.

Chạy kiểm thử từ thư mục gốc của dự án:

```bash
python -m pytest -q
```

### Mốc 2 — Schema và chuẩn bị dữ liệu

Đã triển khai:

- Schema JSONL có kiểm tra cho dữ liệu sạch và dữ liệu noisy đã căn chỉnh token.
- Loại câu trùng chính xác sau khi chuẩn hóa Unicode và khoảng trắng.
- Kiểm tra trùng câu và `document_id` giữa các split.
- Xây vocabulary cùng tần suất token chỉ từ train, có `<PAD>` và `<UNK>`.
- Script chuẩn bị dữ liệu xuất các split, vocabulary và báo cáo thống kê.

```bash
python -m scripts.prepare_data --input data/raw/corpus.jsonl
```

### Mốc 3 — Baseline Edit Distance

Đã triển khai:

- Levenshtein và unrestricted Damerau–Levenshtein từ đầu.
- Từ điển tần suất chỉ lấy token chữ cái từ tập train sạch.
- Candidate generator có index theo độ dài và giới hạn khoảng cách theo độ dài token.
- Baseline xếp hạng theo edit distance, sau đó phá hòa bằng tần suất train.
- Đánh giá Recall@1/3/5 toàn bộ và riêng target in-vocabulary, OOV rate, tỷ lệ không có ứng viên, số ứng viên và latency.

```bash
python -m scripts.evaluate_candidates \
  --train data/processed/train.jsonl \
  --evaluation data/processed/validation.jsonl
```

