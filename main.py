import time
import threading
import serial
import numpy as np
import joblib
from collections import deque
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler

# ===================================================
# ⚙️ 系統設定
# ===================================================
TARGET_COM_PORT = "COM4"  # ⚠️ 請再次確認你的 COM Port
TARGET_BAUD_RATE = 57600
SAMPLING_RATE = 512
RAW_DATA_BUFFER = deque(maxlen=8192) # 容納 10s 校準資料綽綽有餘
IS_RUNNING = True

# ===================================================
# 🚀 1. 推論引擎 (完美對齊 2.0s 視窗)
# ===================================================
class RealTimeBCIEngine:
    def __init__(self, model, global_scaler, subject_scaler=None, threshold_blink=1200):
        self.model = model
        self.global_scaler = global_scaler
        self.subject_scaler = subject_scaler 
        self.threshold_blink = threshold_blink
        self.fs = SAMPLING_RATE
        self.last_blink_time = -999.0  
        self.blink_cooldown = 1      
        nyquist = 0.5 * self.fs
        self.b, self.a = butter(4, [1.0 / nyquist, 40.0 / nyquist], btype='band')

    def extract_features(self, segment):
        freqs, psd = welch(segment, fs=self.fs, nperseg=self.fs)
        p = [np.sum(psd[(freqs>=1)&(freqs<4)]), np.sum(psd[(freqs>=4)&(freqs<8)]),
             np.sum(psd[(freqs>=8)&(freqs<13)]), np.sum(psd[(freqs>=13)&(freqs<30)]),
             np.sum(psd[(freqs>=30)&(freqs<=40)])]
        total = sum(p) + 1e-6
        return np.array(p + [p[2]/total, p[3]/total, p[1]/(p[3]+1e-6), np.var(segment), np.ptp(segment), kurtosis(segment)]).reshape(1, -1)

    def analyze_buffer(self, raw_buffer, current_time_sec):
        # 💡 修正點 1：精準抓取 2.0 秒，對齊訓練集
        required_samples = int(2.0 * self.fs)
        if len(raw_buffer) < required_samples: return 0.5, 0
            
        recent_data = list(raw_buffer)[-required_samples:]
        recent_data = np.array(recent_data)
        
        # 物理眨眼觸發 (看最後 0.5 秒的振幅)
        ptp_value = np.ptp(recent_data[-int(0.5 * self.fs):])
        is_blink = 1 if (current_time_sec - self.last_blink_time >= self.blink_cooldown and ptp_value > self.threshold_blink) else 0
        if is_blink: self.last_blink_time = current_time_sec
        
        # 專注度預測
        filtered = filtfilt(self.b, self.a, recent_data)
        feat = self.extract_features(filtered)
        if self.subject_scaler: feat = self.subject_scaler.transform(feat)
        feat = self.global_scaler.transform(feat)
        
        # Sklearn 推論
        probs = self.model.predict_proba(feat)[0]
        
        # 權重補償：[Relax, Focus]
        weights = np.array([0.9, 1.1])
        weighted_probs = probs * weights
        focus_ratio = weighted_probs[1] / (weighted_probs[0] + weighted_probs[1])
        return focus_ratio, is_blink

# ===================================================
# 🔌 2. 藍牙通訊執行緒
# ===================================================
def brainlink_reader_thread():
    global RAW_DATA_BUFFER, IS_RUNNING
    print(f"🔌 嘗試連接 BrainLink ({TARGET_COM_PORT})...")
    try:
        with serial.Serial(TARGET_COM_PORT, TARGET_BAUD_RATE, timeout=1) as ser:
            ser.reset_input_buffer()
            print("✅ 成功連接 BrainLink！正在解碼二進位腦波封包...")
            
            while IS_RUNNING:
                if ser.read(1) == b'\xaa':
                    if ser.read(1) == b'\xaa':
                        plength_byte = ser.read(1)
                        if not plength_byte: continue
                        plength = plength_byte[0]
                        
                        payload = ser.read(plength)
                        checksum_byte = ser.read(1)
                        
                        if not checksum_byte or len(payload) < plength: 
                            continue
                            
                        if plength == 4: 
                            checksum = checksum_byte[0]
                            calc_checksum = (~sum(payload)) & 0xFF
                            
                            if checksum == calc_checksum:
                                if payload[0] == 0x80 and payload[1] == 0x02:
                                    high = payload[2]
                                    low = payload[3]
                                    
                                    raw_value = (high << 8) | low
                                    if raw_value > 32767:
                                        raw_value -= 65536
                                        
                                    RAW_DATA_BUFFER.append(float(raw_value))
                                    
    except Exception as e:
        print(f"❌ 藍牙連線錯誤: {e}")
        IS_RUNNING = False

# ===================================================
# 🎮 3. 遊戲主迴圈執行緒
# ===================================================
def game_loop_thread(model, global_scaler):
    global RAW_DATA_BUFFER, IS_RUNNING
    
    fs = SAMPLING_RATE
    # 💡 修正點 2：引擎啟動需求降為 2.0 秒
    required_samples = int(2.0 * fs)
    calib_samples = int(10.0 * fs)  # 10 秒校準長度維持不變，確保基準線穩定
    
    # ------------------------------------------------
    # 階段 A：自動校準 (Calibration)
    # ------------------------------------------------
    print("\n" + "="*50)
    print("⏳ [遊戲準備] 系統初始化校準中...")
    print("👉 請戴好設備，保持平常心，等待 10 秒鐘...")
    print("="*50)
    
    while len(RAW_DATA_BUFFER) < calib_samples and IS_RUNNING:
        current_len = len(RAW_DATA_BUFFER)
        progress = (current_len / calib_samples) * 100
        print(f"📡 收集腦波資料中... {current_len}/{calib_samples} ({progress:.1f}%)", end="\r")
        time.sleep(0.2)
        
    if not IS_RUNNING: return
        
    print("\n\n⚙️ 資料收集完畢！正在建立玩家專屬阻抗模型...")
    calibration_data = np.array(RAW_DATA_BUFFER)
    
    temp_engine = RealTimeBCIEngine(model, global_scaler)
    calib_features = []
    
    start = 0
    step = fs # 步進 1 秒
    while start + required_samples <= len(calibration_data):
        seg = calibration_data[start : start + required_samples]
        feat = temp_engine.extract_features(seg)
        calib_features.append(feat[0])
        start += step
        
    subject_scaler = StandardScaler()
    subject_scaler.fit(np.array(calib_features))
    
    print("✅ 玩家基準線校準完成！")
    
    # ------------------------------------------------
    # 階段 B：正式進入遊戲控制迴圈
    # ------------------------------------------------
    engine = RealTimeBCIEngine(model, global_scaler, subject_scaler, threshold_blink=700)

    current_focus_level = 0.5  
    GAIN = 0.025               
    DECAY = 0.03               
    
    print("\n🏹 [射箭遊戲開始] 專注縮小準星，眨眼發射！\n")
    
    while IS_RUNNING:
        start_time = time.time()
        
        if len(RAW_DATA_BUFFER) >= required_samples:
            current_data = np.array(RAW_DATA_BUFFER)[-required_samples:]
            
            raw_focus_ratio, is_blink = engine.analyze_buffer(current_data, current_time_sec=start_time)
            
            prediction = 1 if 1*raw_focus_ratio > 0.53 else 0 
            
            if prediction == 1:
                current_focus_level += GAIN  
            else:
                current_focus_level -= DECAY 
            
            current_focus_level = np.clip(current_focus_level, 0, 1)
            
            if is_blink == 1:
                print("\n💥 [開火] 偵測到眨眼！箭矢射出！\n")
            else:
                bar = "█" * int(current_focus_level * 20)
                print(f"🎯 準星穩定度: [{bar:<20}] {current_focus_level:.3f}", end="\r")
        
        elapsed = time.time() - start_time
        time.sleep(max(0, 0.1 - elapsed))

# ===================================================
# 🚀 4. 程式進入點
# ===================================================
if __name__ == "__main__":
    print("🧠 正在載入 AI 模型與特徵縮放器...")
    try:
        model = joblib.load('bci_model.pkl')
        global_scaler = joblib.load('global_scaler.pkl')
        print("✅ 模型與 Scaler 載入成功！\n")
        
    except FileNotFoundError:
        print("❌ 找不到模型檔案！請確認 'bci_model.pkl' 與 'global_scaler.pkl' 放在同一資料夾。")
        exit()
    except Exception as e:
        print(f"❌ 檔案載入失敗: {e}")
        exit()

    hw_thread = threading.Thread(target=brainlink_reader_thread)
    hw_thread.daemon = True 
    hw_thread.start()
    
    try:
        game_loop_thread(model, global_scaler)
    except KeyboardInterrupt:
        print("\n🛑 收到終止指令，關閉系統...")
        IS_RUNNING = False
        hw_thread.join()