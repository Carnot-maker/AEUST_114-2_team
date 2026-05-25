建議放置位置：
E:\Project_Collection\L3_YOLO_media\workspace\app\

模組結構：
app\
├─ grab_app\
│  ├─ __init__.py
│  ├─ config.py          # 路徑、門檻、節點設定
│  ├─ models.py          # DetectionBox / PersonState / GrabDecision
│  ├─ geometry.py        # 幾何運算
│  ├─ yolo_service.py    # YOLO 載入與解析
│  ├─ pose_service.py    # MediaPipe Pose 載入與少量節點繪製
│  ├─ grab_logic.py      # 抓取判定邏輯
│  ├─ renderer.py        # 框線、文字、狀態繪製
│  └─ video_pipeline.py  # 主流程整合
└─ run_grab_pipeline.py  # 執行入口

執行方式：
1. 先切到你的 yolo_mp 虛擬環境
2. 到 app 資料夾後執行：
   python run_grab_pipeline.py

你最常改的地方：
1. config.py 裡的 target_object_names
2. config.py 裡的 stable_frames_for_grab
3. config.py 裡的 yolo_every_n_frames
4. config.py 裡的路徑

如果你之後要再拆更細：
- 可再加一個 tracker.py 處理多人 ID 對應
- 可再加一個 shelf_zone.py 處理貨架 ROI 區域
- 可再加一個 hand_state.py 處理手掌開合
