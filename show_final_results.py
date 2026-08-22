# LƯU Ý: Chạy lệnh này từ thư mục gốc project (ví dụ: python show_final_results.py)
import pandas as pd
import sys
import os

def main():
    try:
        # Cấu hình in pandas để hiển thị đẹp trên console
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'left')
        
        # Đọc file CSV đã chốt (dùng absolute path dựa trên vị trí của script để an toàn tuyệt đối)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'results', 'comparison_summary.csv')
        
        df = pd.read_csv(csv_path)
        
        print("\n==========================================================================================================")
        print("FINAL EVALUATION TABLE (Mean ± Std over 15 Seeds)")
        print("==========================================================================================================")
        print(df.to_string(index=False))
        print("==========================================================================================================\n")
        
    except FileNotFoundError:
        print(f"[Error] File {csv_path} không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Không thể đọc hoặc in kết quả: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
