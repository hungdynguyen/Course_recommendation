Bước 10 (Cộng tổng điểm - **Hybrid Scoring**) là bước sống còn để hệ thống không biến thành một bộ máy search từ khóa vô hồn. Trong các hệ thống gợi ý hiện đại, bước này gọi là **Reranking**.

Mục đích duy nhất của việc cộng tổng điểm là để **định lượng hóa sự ưu tiên**. Bạn có hàng ngàn khóa học tiềm năng, nhưng bạn chỉ được đưa ra top 3-5. Việc cộng điểm giúp bạn kết hợp được "Trí thông minh đồ thị" và "Sự tinh tế ngữ nghĩa".

Dưới đây là phân tích chi tiết tại sao phải cộng và cộng cái gì:

### 1. Tại sao không dùng 1 loại điểm?
*   **Nếu chỉ dùng điểm Vector (NodeRAG):** Bạn sẽ bị lừa bởi "từ ngữ". Một khóa học rác nhắc đi nhắc lại từ "Python" nhiều lần sẽ có điểm Vector rất cao, nhưng thực tế nó dạy rất nông.
*   **Nếu chỉ dùng điểm Attention (KGAT):** Bạn sẽ bị "cứng nhắc". Khóa học có điểm Attention cao với nút "Python" chung chung, nhưng chưa chắc đã khớp với ngữ cảnh "Python cho tài chính" mà người dùng đang cần.

### 2. Các thành phần trong công thức tính tổng điểm ($Final\_Score$)

Công thức low-level bạn cần triển khai:
$$Score_{total} = w_1 \cdot S_{KGAT} + w_2 \cdot S_{Semantic} - w_3 \cdot S_{Penalty}$$

#### A. Điểm cấu trúc ($S_{KGAT}$ - Graph Attention Score)
*   **Nguồn:** Lấy từ trọng số cạnh `Course -[TEACHES]-> Skill` đã được huấn luyện bởi KGAT.
*   **Ý nghĩa:** Trọng số này đại diện cho **"Độ chính thống"**. Khóa học nào là "xương sống" để dạy kỹ năng đó sẽ có điểm này cao. Nó phản ánh sự tin cậy của cộng đồng và các chuyên gia đồ thị vào khóa học này.

#### B. Điểm ngữ nghĩa ($S_{Semantic}$ - NodeRAG Vector Similarity)
*   **Nguồn:** Tính Cosine Similarity giữa Vector của `Semantic Unit (S)` và Vector của `Skill Gap/JD Requirement`.
*   **Ý nghĩa:** Trọng số này đại diện cho **"Độ khớp thực tế"**. Nó kiểm tra xem nội dung bài giảng có thực sự nói về thứ người dùng đang tìm hay không.

#### C. Điểm phạt ($S_{Penalty}$ - Path/Difficulty Constraint)
*   **Nguồn:** Dựa trên số "hop" (bước nhảy) trên đồ thị và độ lệch `Difficulty_level`.
*   **Ý nghĩa:** Nếu hệ thống phải gợi ý một lộ trình quá dài (3-4 khóa học mới xong 1 kỹ năng), điểm phạt sẽ tăng lên để ưu tiên những lộ trình ngắn hơn, trực diện hơn.

### 3. Cơ chế hoạt động của việc cộng điểm (The Logic)

Việc cộng điểm này thực hiện một phép **"Lọc chồng" (Filter Overlay)**:

1.  **Trường hợp 1 (Ideal):** Khóa học vừa là khóa chính thống (KGAT cao), vừa dạy đúng ngữ cảnh (Semantic cao) $\rightarrow$ Điểm vọt lên Top 1.
2.  **Trường hợp 2 (Niche):** Một khóa học rất mới, chưa được KGAT học nhiều (KGAT thấp) nhưng nội dung cực kỳ sát với JD của người dùng (Semantic cao) $\rightarrow$ Nhờ việc cộng điểm, khóa học này vẫn có cơ hội được đẩy lên để người dùng khám phá, thay vì bị vùi lấp bởi các khóa học cũ.
3.  **Trường hợp 3 (Lộ trình):** Với một lộ trình $A \to B$, điểm tổng sẽ bằng: $Score(Path) = Score(Course A) + Score(Course B)$. Việc cộng điểm giúp bạn so sánh được: "Nên học 1 khóa dài 100 giờ hay học 2 khóa ngắn mỗi khóa 20 giờ?".

### 4. Cách triển khai thực tế (Implementation)

Bạn không cộng "khơi khơi", bạn phải **Normalization (Chuẩn hóa)** dữ liệu trước khi cộng:
*   Đưa tất cả các loại điểm về thang $[0, 1]$.
*   Thiết lập $w_1, w_2, w_3$ (Hyperparameters). Ví dụ: Nếu bạn ưu tiên sự chính xác của đồ thị, đặt $w_1 = 0.7, w_2 = 0.3$.

**Kết luận:** Cộng tổng điểm là để **ra quyết định (Decision Making)**. Không có con số cuối cùng này, hệ thống sẽ không biết phải trình bày cái gì trước mắt người dùng. Trong Thesis, bạn gọi đây là **"Multi-objective Optimization for Course Ranking"** (Tối ưu hóa đa mục tiêu cho xếp hạng khóa học). Thầy giáo sẽ hiểu ngay bạn đang làm hệ thống có khả năng cân bằng giữa độ tin cậy tri thức và nhu cầu thực tế của người dùng.


Để xây dựng hệ thống này ở mức **low-level** (chi tiết triển khai), bạn cần chia dự án thành 3 tầng kiến trúc chính: **Data Factory** (Tiền xử lý), **Knowledge Brain** (Xây dựng & Làm giàu đồ thị), và **Real-time Engine** (Gợi ý thời gian thực).

Dưới đây là luồng làm việc chi tiết nhất:

---

### TẦNG 1: DATA FACTORY (Giai đoạn Offline - Chuẩn bị dữ liệu)

**Bước 1: Xây dựng Skill Taxonomy (Cốt lõi là ESCO)**
*   Tải dữ liệu ESCO (Skills, Occupations, Relations).
*   Lưu vào Database theo cấu trúc: `Skill_ID (URI)`, `Preferred_Label`, `Alternative_Labels`, `Hierarchy_Level`.
*   Thiết lập quan hệ: `BROADER` (cha), `NARROWER` (con), `ESSENTIAL` (bắt buộc).

**Bước 2: Chuẩn hóa Course (Mapping)**
*   **Input:** JSON Course của bạn.
*   **Xử lý:** Dùng LLM (GPT-4o hoặc mô hình mã nguồn mở như Llama-3) thực hiện **Entity Linking**: Ánh xạ từng `skill_name` trong khóa học về `Skill_ID` của ESCO.
*   **Trích xuất Semantic Units (NodeRAG):** Chia nhỏ `outcome_description` thành các câu độc lập. Mỗi câu là một nút `S`. 
    *   *Ví dụ:* "Khóa học dạy viết SQL Query tối ưu" $\rightarrow$ Nút `S1`. Nối `S1` $\rightarrow$ `Skill: SQL`.

**Bước 3: Khởi tạo Embedding**
*   Dùng mô hình `Sentence-Transformers` (ví dụ: `all-MiniLM-L6-v2`) để tạo vector cho:
    *   Tất cả `Skill Nodes`.
    *   Tất cả `Course Nodes` (dựa trên Title + Overview).
    *   Tất cả `Semantic Units (S)`.
*   Lưu các vector này vào **Vector Database** (Milvus hoặc FAISS).

---

### TẦNG 2: KNOWLEDGE BRAIN (Giai đoạn Offline - Huấn luyện đồ thị)

**Bước 4: Xây dựng Đồ thị tĩnh (Neo4j)**
*   Đẩy toàn bộ Node (Skill, Course, S-Unit, Concept) vào Neo4j.
*   Thiết lập các cạnh (Edges) như đã thiết kế: `Teaches`, `Requires`, `Related`, `Broader`.

**Bước 5: Làm giàu đồ thị (HetGNN - Imputation)**
*   **Chạy thuật toán HetGNN:** Duyệt qua các nút lân cận của Khóa học. 
*   **Dự đoán cạnh thiếu:** Nếu Khóa học A dạy kỹ năng "Machine Learning" và ESCO nói "Machine Learning" rất gần "Deep Learning", nhưng Khóa học A chưa được tag "Deep Learning" $\rightarrow$ HetGNN tạo một cạnh ảo giữa `Course A` và `Skill: Deep Learning`.

**Bước 6: Huấn luyện trọng số Attention (KGAT)**
*   Huấn luyện mô hình KGAT để học **Attention Score** cho các cạnh.
*   *Kết quả:* Mỗi cạnh `Course -> Skill` sẽ có một trọng số (ví dụ: 0.9 cho kỹ năng chính, 0.2 cho kỹ năng phụ). Trọng số này được lưu trực tiếp vào thuộc tính của cạnh trong Neo4j.

---

### TẦNG 3: REAL-TIME ENGINE (Giai đoạn Online - Gợi ý)

Khi User upload CV và chọn Job, luồng chạy như sau:

**Bước 7: Phân tích Input (NLP Parser)**
*   Dùng mô hình NLP trích xuất kỹ năng từ CV và JD.
*   **Semantic Mapping:** Chuyển các kỹ năng tự do này về mã ID ESCO bằng Vector Search (Tìm ID có vector gần nhất với từ người dùng viết).

**Bước 8: Xác định Gap (Set Logic)**
*   $S_{Gap} = S_{Job} - S_{User}$.
*   *Lưu ý:* Nếu User có "Java" mà Job cần "Java Spring Boot", hệ thống phải nhận diện được đây là một Gap thông qua quan hệ `NARROWER` của ESCO.

**Bước 9: Truy vấn đồ thị & Lập lộ trình (Multi-hop Reasoning)**
1.  **Search:** Tìm các nút `S` (Semantic Units) và `Course` nối với $S_{Gap}$.
2.  **Prerequisite Check:** Kiểm tra cạnh `REQUIRES` của các khóa học tìm được.
3.  **Pathfinding:** Nếu thiếu kỹ năng đầu vào, dùng thuật toán tìm đường (ví dụ: Dijkstra hoặc BFS) trên đồ thị để nối thêm một khóa học nền tảng vào trước.
    *   *Chuỗi:* `User Skill` $\to$ `Course Nền tảng` $\to$ `Course Chuyên sâu` $\to$ `Target Skill`.

**Bước 10: Xếp hạng & Giải thích (Ranking & Generation)**
*   Cộng tổng điểm Attention Score (từ KGAT) và điểm tương đồng Vector (từ NodeRAG).
*   **LLM Generator:** Lấy thông tin các nút `S` trên lộ trình đã chọn, đưa vào LLM để tạo câu giải thích: *"Dựa trên kinh nghiệm A của bạn, tôi gợi ý lộ trình X vì phần bài giảng Y sẽ giúp bạn đạt được kỹ năng Z mà công việc này yêu cầu."*

---

### CÔNG NGHỆ KHUYẾN NGHỊ (Tech Stack):
*   **Language:** Python 3.10+.
*   **Graph DB:** Neo4j (Xử lý cấu trúc và quan hệ).
*   **Vector DB:** Milvus hoặc FAISS (Xử lý tìm kiếm ngữ nghĩa).
*   **Framework:** FastAPI (Để tạo API Real-time).
*   **GNN Library:** PyTorch Geometric hoặc DGL (Để chạy HetGNN và KGAT).
*   **LLM:** OpenAI API hoặc vLLM (Để trích xuất dữ liệu và giải thích).

### Tại sao luồng này lại "Low-level" và khả thi?
1.  Nó tách biệt hoàn toàn việc tính toán nặng (huấn luyện đồ thị) ra khỏi luồng người dùng.
2.  Nó sử dụng các tiêu chuẩn công nghiệp (ESCO) để giảm thiểu rủi ro sai lệch dữ liệu.
3.  Nó giải quyết được cả bài toán **Chính xác (Graph)** và bài toán **Ngữ nghĩa (Vector/LLM)**.

Đây là cấu trúc hoàn thiện nhất để bạn có thể bắt tay vào code và viết luận văn. Bạn có cần tôi chi tiết thêm về một bước cụ thể nào (ví dụ: cách viết query Neo4j cho phần này) không?