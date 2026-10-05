# Báo Cáo Thực Hành Lab 18 — 2D Perception: Detection · Segmentation · Keypoints

## 1. Thông Tin Chung
- **Học viên:** Bùi Văn Quang
- **Mã học viên / Lớp:** 2A202602688
- **Môn học:** AICB · Computer Vision (Track 4)
- **Chủ đề:** 2D Perception: Detection · Segmentation · Keypoints
- **Link notebook đã chạy:** https://github.com/Quang20040/Track04-Day3-BuiVanQuang-2A202602688-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

---

## 2. Tóm Tắt Kết Quả Triển Khai

### 2.1. Phần 1 — Object Detection
- **Tự cài đặt:**
  - `box_iou`: Tính IoU chuẩn xác giữa hai tập bounding box dạng `xyxy`, sử dụng kỹ thuật broadcasting và clamp phần giao không âm. Kết quả khớp hoàn toàn với `torchvision.ops.box_iou`.
  - `nms`: Greedy Non-Maximum Suppression tuần tự lọc các bounding box trùng lặp theo điểm confidence giảm dần, chỉ giữ box có `IoU <= iou_thr`.
  - `batched_nms`: Khử trùng lặp class-aware bằng phương pháp dịch chuyển toạ độ theo class (`offset = cls * (max_coord + 1)`), đảm bảo các box thuộc các class khác nhau không triệt tiêu lẫn nhau.
  - ⭐ `average_precision` (1D): Tính toán đường Precision-Recall tích lũy, xây dựng đường bao envelope (`np.maximum.accumulate` từ phải sang trái) và nội suy tại 101 điểm chuẩn COCO. Đạt $AP \approx 0.535$ trên dữ liệu mẫu slide.
- **Đánh giá Latency & So sánh Head:**
  - So sánh giữa Head One-to-Many (+ NMS) và Head One-to-One (NMS-free) ở hai mức ngưỡng `conf = 0.25` và `conf = 0.001`.
  - Ở `conf = 0.001`, head one-to-one thể hiện ưu thế vượt trội về thời gian hậu xử lý (`postprocess`), giảm thiểu hoàn toàn nút thắt cổ chai NMS khi số lượng candidate boxes lên tới hàng nghìn.

### 2.2. Phần 2 — Segmentation
- **Tự cài đặt:**
  - `mask_iou`: Tính ma trận IoU giữa hai tập mask boolean thông qua phép nhân ma trận tích vô hướng (`inter = a @ b.T`) và broadcast diện tích union, đạt độ chính xác tương đương so sánh vòng lặp.
  - `polygon_to_mask`: Chuyển đổi chuỗi toạ độ đỉnh polygon pixel $(x, y)$ sang mặt nạ nhị phân boolean $(H, W)$ bằng `cv2.fillPoly`.
  - `mask_to_yolo_seg`: Trích xuất đường viền ngoài lớn nhất (`max(contours, key=cv2.contourArea)`), làm mịn bằng `cv2.approxPolyDP` và chuẩn hoá toạ độ về khoảng $[0, 1]$ tương ứng với chiều rộng $W$ và chiều cao $H$.
- **Gán nhãn tự động với SAM 2.1 (Auto-labeling):**
  - Trích xuất bounding box từ detector YOLO26 để làm box prompt cho SAM 2.1-tiny.
  - Tạo nhãn segmentation chính xác cho toàn bộ các cá thể người và xe buýt, xuất ra file `autolabel/bus.txt`.
  - Kiểm tra round-trip polygon $\to$ mask đạt IoU $> 0.95$, khớp trọn vẹn với ground-truth SAM.

### 2.3. Phần 3 — Keypoints & Pose Estimation
- **Tự cài đặt:**
  - `oks` (Object Keypoint Similarity): Tính độ tương đồng keypoint chuẩn COCO có tính đến tỷ lệ diện tích $s^2 = \text{area}$, sai số từng điểm $d_i^2$, trọng số dung sai $k_i = 2\sigma_i$ và chỉ tính trung bình trên các keypoint có nhãn ($vis = \text{True}$).
  - `joint_angle`: Tính góc giữa ba điểm khớp $a - b - c$ tại đỉnh $b$ thông qua tích vô hướng hai vector $\vec{ba}$ và $\vec{bc}$, chuẩn hoá và chặn sai số số học trước khi chuyển đổi sang độ.
- **Phát hiện người ngã & Phân tích góc thân:**
  - Xây dựng luật đo độ nghiêng thân `torso_tilt` nối trung điểm hai vai với trung điểm hai hông so với phương thẳng đứng.
  - Trên ảnh gốc: thân nghiêng $\approx 1^\circ$ (trạng thái "đứng").
  - Trên ảnh xoay $90^\circ$: thân nghiêng $83^\circ - 94^\circ$ (kích hoạt cảnh báo "🚨 NGÃ?"). Đồng thời ghi nhận hiện tượng số lượng người phát hiện bị sụt giảm do mô hình bị phụ thuộc vào tư thế đứng trong tập dữ liệu COCO.

### 2.4. Phần 4 — Fine-tune YOLO26n-pose trên dữ liệu custom (tiger-pose)
- **Hiệu chỉnh `FLIP_IDX`:**
  - Sửa `FLIP_IDX = [0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]` theo đúng quy ước giải phẫu học: hoán đổi đối xứng các cặp chân trái/phải (`right_hind_hock` $\leftrightarrow$ `left_hind_hock`, `right_hind_paw` $\leftrightarrow$ `left_hind_paw`, `right_front_wrist` $\leftrightarrow$ `left_front_wrist`, `right_front_paw` $\leftrightarrow$ `left_front_paw`) và giữ nguyên 4 điểm trục giữa (`nose`, `head`, `withers`, `tail_base`).
- **Huấn luyện mô hình:**
  - Fine-tune trên T4 GPU với 40 epochs, `imgsz=640`, `batch=16`.
  - Pose mAP50 đạt $> 0.99$.
- **Phân tích lỗi (Error Analysis):**
  - Chấm điểm OKS từng ảnh tập validation để tìm 6 trường hợp tệ nhất.
  - Chỉ ra hai nhóm lỗi cốt lõi: Che khuất chéo giữa các chân (Inter-limb occlusion) và Cắt cụt biên khung hình (Boundary truncation), kèm giải pháp khắc phục cụ thể về data augmentation và ngưỡng lọc confidence.

---

## 3. Trả Lời Chi Tiết Các Câu Hỏi Lý Thuyết & Thực Nghiệm (Q1 – Q12)

### Câu hỏi Q1 (Phần 1A)
- **(a)** `bus.jpg` sau khi letterbox về kích thước $640 \times 480$ có đúng **6 300** dự đoán thô (shape `(1, 84, 6300)`). Số dự đoán này ít hơn con số 8 400 của ảnh vuông $640 \times 640$ vì kích thước lưới feature map ở cả 3 tầng (strides 8, 16, 32) giảm theo chiều cao 480 px:
  $$\left(\frac{640}{8} \times \frac{480}{8}\right) + \left(\frac{640}{16} \times \frac{480}{16}\right) + \left(\frac{640}{32} \times \frac{480}{32}\right) = (80 \times 60) + (40 \times 30) + (20 \times 15) = 4800 + 1200 + 300 = 6300 \text{ vị trí}.$$
- **(b)** Mọi bounding box đều bị lệch sang **phải và xuống dưới** đúng nửa kích thước $(+w/2, +h/2)$ chứng tỏ nhãn gốc vốn được lưu theo quy ước **`cxcywh`** (toạ độ tâm $c_x, c_y$ và kích thước $w, h$), nhưng khi đọc lại đã bị hiểu nhầm thành quy ước **`xywh`** (coi toạ độ tâm $c_x, c_y$ là toạ độ góc trên-trái $x_{\text{topleft}}, y_{\text{topleft}}$). Khi lấy tâm làm góc trên-trái, toàn bộ bounding box sẽ bị dịch sang phải đúng $w/2$ và dịch xuống dưới đúng $h/2$.

### Câu hỏi Q2 (Phần 1C)
- Cột thay đổi nhiều nhất giữa hai head là **`postprocess (ms)`** (thời gian hậu xử lý), thể hiện rõ rệt nhất ở mức `conf = 0.001`.
- Ở `conf = 0.25`, số lượng box vượt ngưỡng tin cậy rất ít (chỉ vài box) nên NMS chạy rất nhanh (< 2 ms). Nhưng ở `conf = 0.001`, hàng nghìn box thô được giữ lại trước bước lọc:
  - Head one-to-many + NMS phải chạy thuật toán greedy NMS tuần tự (so sánh IoU từng cặp, sắp xếp, lọc lặp) khiến thời gian postprocess tăng vọt.
  - Head one-to-one (NMS-free) không cần tính ma trận IoU và lọc lặp, trực tiếp xuất kết quả nên thời gian postprocess giữ ở mức gần như không đổi (< 1 ms).
- Lợi ích của NMS-free rõ rệt hơn khi:
  1. **Conf thấp:** Số lượng box ứng viên bùng nổ, độ phức tạp $O(N^2)$ của việc tính IoU và lọc lặp làm NMS quá tải.
  2. **Cảnh đông (dense objects):** Mật độ đối tượng cao làm tăng số vòng lặp loại bỏ box duplicate.
  3. **Chạy trên CPU/NPU:** NMS là thuật toán tuần tự, chứa nhiều câu lệnh rẽ nhánh điều kiện và truy xuất bộ nhớ rời rạc, rất khó vector hóa song song trên phần cứng AI chuyên biệt (NPU/DSP) hoặc CPU hạn chế luồng, trở thành điểm nghẽn (bottleneck) lớn nhất của toàn bộ pipeline triển khai edge.

### Câu hỏi Q3 (Phần 1C)
- **(a)** Với NMS ngưỡng 0.7: Hai công nhân có box IoU = 0.75 > 0.7, nên box của người có điểm confidence thấp hơn sẽ bị coi là duplicate của người kia và bị NMS loại bỏ hoàn toàn, dẫn đến việc camera bị **đếm thiếu người** (bỏ sót 1 công nhân).
  - Tăng ngưỡng NMS (ví dụ lên 0.85): Cho phép giữ lại các box chồng lấn nhau nhiều hơn, giảm nguy cơ bỏ sót trong đám đông, nhưng đánh đổi lại là dễ giữ lại duplicate (một người nhận 2-3 box trùng lặp khi model dự đoán dày).
  - Giảm ngưỡng NMS (ví dụ xuống 0.5): Lọc sạch duplicate ở cảnh thưa, nhưng sẽ loại bỏ các trường hợp người đi cạnh nhau hoặc che khuất nhau (occlusion).
- **(b)** YOLO26 vẫn train thêm head one-to-many vì: Cơ chế one-to-one matching chỉ gán đúng 1 positive anchor cho mỗi ground-truth, dẫn đến tín hiệu giám sát (supervision signal) trong quá trình huấn luyện rất thưa thớt, làm backbone học biểu diễn đặc trưng chậm và kém phong phú. Head one-to-many cung cấp gradient dày đặc, đa tỷ lệ để tối ưu hóa backbone tốt hơn. Khi inference, ta có thể bỏ head one-to-many và dùng trực tiếp head one-to-one để đạt tốc độ cao mà vẫn thừa hưởng đặc trưng mạnh mẽ từ backbone.

### Câu hỏi Q4 (Phần 2A)
- **(a)** Đếm người bằng vùng liên thông trên bản đồ semantic bị sai vì semantic segmentation chỉ phân loại class ở cấp độ pixel mà không phân biệt các cá thể (instance):
  - **Đếm thiếu:** Khi hai hay nhiều người đứng gần nhau, chạm vào nhau hoặc che khuất nhau, các pixel class `person` dính liền thành 1 mảng liên thông duy nhất, hàm đếm chỉ ghi nhận 1 vùng.
  - **Đếm thừa:** Khi một người bị che ngang bởi vật cản (như vali, lan can, cột hoặc phụ kiện), mảng pixel của người đó bị đứt đoạn thành nhiều phần rời rạc (như đầu, thân, chân tách biệt), hàm đếm sẽ coi mỗi phần là một "người" riêng biệt (trong ảnh `bus.jpg` có 4 người nhưng ra tới 7 vùng liên thông).
  - **Panoptic segmentation** giải quyết việc này bằng cách thống nhất cả semantic ("stuff" - các vùng vô định hình như đường, trời) và instance ("things" - các đối tượng đếm được như người, xe với ID riêng biệt), giúp vừa hiểu toàn diện ngữ cảnh vừa đếm chính xác từng đối tượng.
- **(b)** Bản đồ semantic gán phần lớn pixel của xe buýt vào class **`train`**. Model nhầm như vậy vì model YOLO26n-sem được huấn luyện trên dataset Cityscapes (châu Âu) - nơi các phương tiện giao thông công cộng đường phố lớn như tàu điện mặt đất (tram/train) và xe buýt có kích thước hộp chữ nhật dài, bề mặt phẳng và các dải cửa kính liên tục rất giống nhau. Khi gặp xe buýt đỏ trong ảnh `bus.jpg` chụp cận cảnh với các mảng kính lớn, model kích hoạt mạnh đặc trưng của tàu điện/tàu hỏa trong tập Cityscapes.

### Câu hỏi Q5 (Phần 2B)
- **(a)**
  - Mask R-CNN dự đoán mask có độ phân giải cố định 28×28 trên mỗi RoI, sau đó phóng to (bilinear interpolation) về kích thước bounding box gốc. Đối với object nhỏ, 28×28 là đủ chi tiết; nhưng đối với object **LỚN** (chiếm hàng trăm pixel), việc phóng đại từ lưới 28×28 khiến biên mask bị mờ nhòe, mất chi tiết góc cạnh và trở nên rất thô.
  - YOLO26-seg tổ hợp tuyến tính từ 32 prototype masks có kích thước cố định (ví dụ 160×160) trên toàn ảnh. Đối với object lớn, mask tận dụng độ phân giải của prototype map nên biên khá mịn; nhưng đối với object **NHỎ**, phần crop trên prototype map chỉ có kích thước vài pixel, khiến mask của object nhỏ bị răng cưa, rời rạc và rất thô.
- **(b)** Nếu thay Hungarian matching bằng thuật toán tham lam ("mỗi mask YOLO lấy mask Mask R-CNN có IoU cao nhất"):
  - Có thể xảy ra xung đột khi hai mask YOLO khác nhau cùng có IoU lớn nhất với **DUY NHẤT MỘT** mask của Mask R-CNN, dẫn đến một mask Mask R-CNN bị gán trùng lặp cho nhiều mask YOLO, đồng thời bỏ sót mask Mask R-CNN khác.
  - Ghép cặp tham lam chỉ tối ưu cục bộ từng cặp, không đảm bảo tính song ánh (bipartite matching 1-1) và không tối đa hóa tổng IoU trên toàn bộ tập đối tượng như Hungarian algorithm (Linear Sum Assignment).

### Câu hỏi Q6 (Phần 2C)
- **(a)** Điểm A (ở giữa tâm box người) cho ra mask của toàn bộ cơ thể người (IoU $\approx 0.99$ với mask trọn vẹn). Điểm B (ở 80% chiều cao box, rơi vào ống quần) khiến SAM chỉ tách riêng chiếc quần hoặc ống chân (IoU $\approx 0.00$ so với cả người). Prompt bằng box an toàn hơn nhiều vì prompt bằng điểm có tính mơ hồ ngữ nghĩa cao (ambiguity - mô hình không biết người dùng muốn chọn cả đối tượng hay chỉ một chi tiết/bộ phận chứa điểm đó). Bounding box xác định rõ phạm vi không gian toàn thể của đối tượng, giúp SAM tách đúng toàn bộ instance.
- **(b)**
  - **Dùng SAM trực tiếp lúc triển khai:** Khi cần tính linh hoạt cao (zero-shot, interactive segmentation), người dùng thao tác tương tác linh hoạt trên các class chưa từng thấy trước đó, và bài toán không yêu cầu thời gian thực khắt khe (vì SAM có backbone rất nặng, latency hàng trăm ms đến vài giây).
  - **Dùng SAM để auto-label rồi train YOLO26-seg:** Khi triển khai thực tế trên camera giám sát/thiết bị biên (edge devices) yêu cầu tốc độ cao thời gian thực (FPS cao > 30-60, latency vài ms), tài nguyên phần cứng giới hạn (CPU/NPU nhúng), và tập class mục tiêu đã cố định xác định trước (như người, xe, mũ bảo hộ). YOLO-seg nhỏ kế thừa nhãn chất lượng cao từ SAM nhưng chạy nhanh hơn hàng chục đến hàng trăm lần.

### Câu hỏi Q7 (Phần 3A)
- Khi đổi `KP_THR = -100` (bỏ lọc ngưỡng logit), Keypoint R-CNN vẽ các điểm đầu gối và mắt cá bị dồn cụm ở sát mép dưới của bounding box hoặc rải rác ngẫu nhiên trong RoI, dù thực tế hai chân của người nằm hoàn toàn ngoài khung ảnh.
- Điều này nói lên rằng thao tác giải mã heatmap bằng phép toán `argmax` thuần túy luôn luôn tìm ra một vị trí có giá trị lớn nhất trong ma trận heatmap (ngay cả khi toàn bộ ma trận heatmap có giá trị rất thấp hoặc âm do không có tín hiệu của khớp đó). Phép `argmax` không có khả năng tự biết điểm đó có thực sự tồn tại trong ảnh hay không.
- Vì vậy, mỗi keypoint bắt buộc phải đi kèm một điểm tin cậy (confidence score hoặc heatmap logit). Điểm tin cậy đóng vai trò điều kiện lọc (gating): chỉ khi giá trị vượt qua một ngưỡng tin cậy xác định thì toạ độ argmax mới được công nhận; nếu điểm tin cậy thấp, điểm đó phải được coi là bị che khuất hoặc nằm ngoài ảnh để tránh đưa ra toạ độ giả gây sai lệch nghiêm trọng cho các thuật toán nghiệp vụ.

### Câu hỏi Q8 (Phần 3B)
- Từ đồ thị bên trái:
  - Đối với người nhỏ 40×80 px (diện tích 3 200 px²): Chỉ cần sai số khoảng **2 đến 3 pixel** là OKS trung bình đã rơi dốc đứng xuống dưới ngưỡng 0.5.
  - Đối với người lớn 250×500 px (diện tích 125 000 px²): Cần sai số lên tới khoảng **15 đến 18 pixel** thì OKS mới giảm xuống dưới 0.5.
- Hệ số $\sigma$ (độ lệch chuẩn gán nhãn) của mắt (0.025) nhỏ hơn hông (0.107) nhiều lần vì mắt là một điểm mốc thị giác rõ ràng, sắc nét, có ranh giới giải phẫu chính xác nên độ phân tán giữa những người gán nhãn rất nhỏ. Ngược lại, khớp hông nằm ẩn bên trong cơ thể dưới lớp cơ và trang phục, không lộ mốc xương rõ rệt nên annotator chỉ có thể ước lượng xấp xỉ, dẫn đến độ lệch gán nhãn lớn hơn nhiều.
- **Ý nghĩa với camera cổng gắn trên cao (mỗi người chỉ cao khoảng 80 pixel):** Ở khoảng cách này, một người chỉ chiếm vài chục pixel, sai số định vị dù chỉ 1-2 pixel cũng làm OKS tụt thảm hại. Ngoài ra, các chi tiết có $\sigma$ nhỏ như mắt/mũi gần như không thể phát hiện chính xác vì mắt chỉ chiếm chưa đầy 1 pixel mờ nhòe. Hệ thống pose khi đó phải dựa chủ yếu vào các khớp lớn, có dung sai $\sigma$ cao như vai, hông, đầu gối, hoặc cần bố trí camera ở góc thấp/tăng độ phân giải quang học.

### Câu hỏi Q9 (Phần 3C)
- **(a)**
  - **Báo nhầm (False Positive):** Công nhân đang cúi người xuống sàn để nhặt bu lông/dụng cụ hoặc cúi lưng làm việc; hoặc công nhân chui gầm máy móc để kiểm tra kỹ thuật. Trong các tình huống này, góc nghiêng thân so với phương thẳng đứng dễ dàng vượt quá 60°, luật sẽ lập tức cảnh báo "🚨 NGÃ?" dù người đó hoàn toàn bình thường.
  - **Bỏ sót (False Negative):** Người bị ngã ngửa hoặc ngã sấp theo phương trực diện hướng thẳng về phía camera (trục ngã vuông góc với mặt phẳng ảnh 2D). Trên ảnh 2D, hai vai vẫn nằm phía trên hai hông nên góc nghiêng thân tính ra chỉ khoảng 0°–20°, luật kết luận "đứng" và bỏ sót hoàn toàn. Hoặc trường hợp người ngã bị che khuất khớp hông/vai dẫn đến model không đủ confidence và rơi vào trạng thái "không kết luận".
- **(b)** Ảnh xoay 90° model tìm được ít người hơn vì các mô hình pose (như YOLO-pose) được huấn luyện chủ yếu trên ảnh chụp người ở tư thế thẳng đứng trong COCO. Mạng tích chập đã học các đặc trưng có tính định hướng không gian (đầu ở trên, thân ở giữa, chân ở dưới). Khi ảnh bị xoay 90°, hình thái trực quan bị lệch hoàn toàn khỏi phân phối huấn luyện, detector ở giai đoạn đầu không kích hoạt hoặc score tụt xuống dưới ngưỡng phát hiện (conf < 0.25).
- **(c)** Một hệ thống cảnh báo ngã thực tế cần bổ sung:
  1. **Chiều thời gian (Temporal tracking):** Theo dõi chuyển động qua chuỗi khung hình video liên tục — nhận diện pha rơi nhanh đột ngột (vận tốc biến thiên lớn của trọng tâm) kèm theo trạng thái nằm bất động kéo dài dưới sàn trong T giây (ví dụ > 5–10s) để tránh báo nhầm hành động cúi người nhặt đồ.
  2. **Bản đồ sàn và không gian 3D (3D floor plane / depth):** Ước lượng mặt phẳng sàn nhà máy và chiều cao của trọng tâm cơ thể so với mặt đất (khi ngã khoảng cách đến mặt sàn tiến về 0).
  3. **Kết hợp đa camera (Multi-view fusion) hoặc cảm biến bổ trợ:** Triệt tiêu điểm mù do che khuất và giải quyết trường hợp ngã theo phương trực diện với ống kính.

### Câu hỏi Q10 (Phần 4A)
- Nếu giữ `flip_idx` đồng nhất [0, 1, ..., 11], mAP trên tập val **NÀY KHÔNG HỀ GIẢM** (mAP vẫn có thể rất cao > 0.9).
  - **Lý do:** 100% con hổ trong tập val đều quay sang phải (facing: 53 quay phải, 0 quay trái). Vì tập val chỉ chứa hổ quay phải, mô hình chỉ được đánh giá ở hướng mà chân nhìn thấy phía camera luôn luôn là chân phải (`right_*`). Tập val bị thiên lệch phân phối (distribution bias) giống hệt tập train, nên lỗi tráo đổi nhãn khi lật gương trong quá trình huấn luyện hoàn toàn bị che giấu bởi tập val này.
- **Tình huống triển khai trở thành bug thật sự:** Khi triển khai mô hình vào thực tế (in production / in the wild), hổ có thể quay sang **TRÁI** (hoặc di chuyển từ phải qua trái). Khi con hổ quay sang trái, chân nhìn thấy phía trước camera thực tế là chân trái (`left_*`), nhưng mô hình đã bị học nhầm do augmentation lật gương với flip_idx đồng nhất, dẫn đến việc mô hình dự đoán toàn bộ chân trái thành chân phải và ngược lại (đảo ngược hoàn toàn nhãn giải phẫu trái/phải).

### Câu hỏi Q11 (Phần 4B)
- **1. Kiểu lỗi 1: Nhầm lẫn và che khuất giữa các chân (Inter-limb Occlusion & Left/Right Swapping)**
  - *Chi tiết quan sát:* Trong các ảnh có OKS thấp nhất (như các ảnh hổ đang sải bước di chuyển), hai chân sau hoặc hai cẳng chân bắt chéo, che lấp nhau. Dự đoán của mô hình bị nhầm lẫn giữa chân bên trái và chân bên phải, hoặc hai điểm keypoint cùng bị kéo dồn về một bên chân đang nhìn thấy rõ (vạch vàng nối giữa GT và Pred kéo dài chéo giữa hai chân). Biểu đồ sai số cũng cho thấy các điểm chân (`paw`, `hock`) có sai số chuẩn hóa cao nhất (lên tới 0.15–0.25).
  - *Cách khắc phục:* Bổ sung augmentation che khuất cục bộ ngẫu nhiên (Cutout, GridMask, Random Erasing) để mô hình học cách suy luận vị trí khớp bị che dựa trên cấu trúc xương toàn thân; đồng thời bổ sung cờ visibility ($v = 0, 1, 2$) vào nhãn keypoint để không phạt nặng sai số định vị khi khớp bị che khuất hoàn toàn.
- **2. Kiểu lỗi 2: Cắt cụt biên ảnh và góc chụp bất thường (Boundary Truncation & Extreme Poses)**
  - *Chi tiết quan sát:* Ở các ảnh hổ bị chụp sát mép khung hình hoặc chuyển động nhanh ra khỏi ảnh, các keypoint ở phần đầu (`nose`, `head`) hoặc phần đuôi (`tail_base`) nằm sát mép hoặc bị cắt mất một phần ra ngoài ảnh. Mô hình cố định vị điểm vào bên trong ảnh dẫn đến sai số lớn.
  - *Cách khắc phục:* Về dữ liệu, áp dụng Random Crop/Mosaic có kiểm tra để giữ cấu trúc cơ thể tự nhiên; về hậu xử lý, thiết lập ngưỡng tin cậy (keypoint confidence threshold, ví dụ 0.4-0.5) để loại bỏ các điểm có độ tin cậy thấp ở mép ảnh thay vì cố gắng lấy toạ độ argmax thô.

### Câu hỏi Q12 (Phần 4B)
- Việc dùng chung `SIGMA_TIGER = 1/12` cho mọi keypoint là **KHÔNG HỢP LÝ** về mặt giải phẫu và độ chính xác đo lường:
  - Mũi hổ (`nose`) là một mốc thị giác rất sắc nét, cố định và có ranh giới rõ ràng, sai số gán nhãn giữa những người đánh dấu rất nhỏ. Do đó mũi cần có $\sigma$ nhỏ để đòi hỏi độ chính xác định vị cao.
  - Ngược lại, gốc đuôi (`tail_base`) hay các khớp cẳng chân/khớp hông nằm dưới lớp da và lông dày di động, không có mốc xương lộ rõ trên bề mặt, độ bất định khi gán nhãn lớn hơn nhiều. Các điểm này cần có $\sigma$ lớn hơn để có dung sai sai số phù hợp.
- **Cách ước lượng $\sigma$ cho một dataset custom (theo phương pháp chuẩn của MS COCO):**
  1. Chọn một tập mẫu ngẫu nhiên (khoảng 50-100 ảnh) và cho ít nhất 2-3 annotator độc lập cùng gán nhãn keypoint trên cùng tập ảnh đó.
  2. Với mỗi con hổ thứ $k$ có diện tích bounding box là $s_k^2$, tính khoảng cách Euclidean $d_{i,k}$ từ điểm gán nhãn thứ $i$ đến vị trí trung bình (ground-truth consensus) của điểm đó.
  3. Tính sai số chuẩn hóa theo kích thước đối tượng: $e_{i,k} = \frac{d_{i,k}}{s_k}$.
  4. Ước lượng $\sigma_i$ cho từng keypoint $i$ bằng độ lệch chuẩn của các sai số chuẩn hóa này trên toàn bộ tập mẫu: $\sigma_i = \sqrt{\frac{1}{N} \sum_{k} e_{i,k}^2}$. Vector $\sigma$ này sau đó được điền vào tham số `kpt_oks_sigmas` trong file cấu hình dataset YAML.

---

## 4. Phần Bonus (⭐) & Thí Nghiệm Mở Rộng

### 4.1. Thí nghiệm 4C — Val gốc so với Val lật gương
- Khi đặt `RUN_4C = True` và `TRAIN_IDENTITY = True`:
  - **Tập val gốc (quay phải 100%):** Cả hai mô hình (`flip_idx` giải phẫu và `flip_idx` đồng nhất) đều đạt mAP rất cao ($\approx 0.42 - 0.46$).
  - **Tập val lật gương (quay trái 100%):**
    - Mô hình với `flip_idx` giải phẫu duy trì mAP cao tương đương ($\approx 0.44$).
    - Mô hình với `flip_idx` đồng nhất tụt dốc thảm hại (Pose mAP50-95 tụt sâu xuống $0.297$, giảm hơn $28\%$).
  - **Metric nào đã che lỗi?** Chính metric Pose mAP trên tập val gốc ban đầu đã hoàn toàn che giấu lỗi `flip_idx`, do tập val này bị bias 100% hướng quay phải. Để tránh lỗi ngụy biện này trong thực tế, tập validation bắt buộc phải được thiết kế cân bằng phân phối (balanced distribution) đại diện cho mọi góc nhìn, hướng chuyển động và biến thể đối xứng thực tế.

### 4.2. Bài tập về nhà — Đo Latency Pipeline khi Export ONNX
- Thực hiện export model sang ONNX: `model.export(format="onnx")`.
- Đo lường độ trễ suy luận trên CPU cho cả hai head:
  - Head one-to-one (NMS-free) cho tốc độ ổn định xuyên suốt các mức confidence từ 0.25 đến 0.001.
  - Head one-to-many xuất hiện độ trễ lớn khi conf thấp do phụ thuộc vào thuật toán NMS trên CPU.
