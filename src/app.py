import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import requests
import json
from agent import DQNAgent

# Page layout and aesthetics
st.set_page_config(
    page_title="AI Personalized Spaced Repetition Flashcards",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium styling (Glassmorphism & Dark/Modern Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 30px;
    }
    
    /* Glassmorphism Card styling */
    .flashcard {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .flashcard:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }
    
    .flashcard-word {
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #e2e8f0;
        margin-bottom: 20px;
    }
    
    .flashcard-type {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #718096;
        margin-bottom: 10px;
    }
    
    .flashcard-meaning {
        font-size: 1.8rem;
        font-weight: 400;
        color: #38bdf8;
        border-top: 1.5px dashed rgba(255, 255, 255, 0.1);
        padding-top: 20px;
        margin-top: 20px;
    }
    
    /* Stat cards */
    .stat-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .stat-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #818cf8;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Predefined high-quality vocabulary dataset (CSE terms)
VOCABULARY = [
    {"word": "Polymorphism", "type": "Object-Oriented Programming", "meaning": "Đa hình - Khả năng các đối tượng thuộc các lớp khác nhau phản hồi cùng một thông điệp theo các cách khác nhau."},
    {"word": "Recursion", "type": "Algorithm Design", "meaning": "Đệ quy - Phương pháp giải quyết bài toán bằng cách chia nhỏ bài toán và gọi lại chính nó với các tham số nhỏ hơn."},
    {"word": "Asymptotic Complexity", "type": "Theory of Computation", "meaning": "Độ phức tạp tiệm cận - Cách biểu diễn sự tăng trưởng về thời gian chạy hoặc không gian bộ nhớ của thuật toán (ví dụ: Big-O)."},
    {"word": "Concurrency", "type": "Operating Systems", "meaning": "Đồng thời - Khả năng thực hiện nhiều tác vụ trong các khoảng thời gian xen kẽ nhau mà không cần chờ tác vụ trước kết thúc."},
    {"word": "Deadlock", "type": "Operating Systems", "meaning": "Khóa chết - Tình trạng hai hay nhiều tiến trình bị treo vĩnh viễn vì mỗi tiến trình đều chờ tài nguyên do tiến trình kia nắm giữ."},
    {"word": "Idempotency", "type": "Software Engineering", "meaning": "Tính lũy đẳng - Thuộc tính của một thao tác mà khi thực hiện nhiều lần vẫn cho ra cùng một kết quả như khi thực hiện một lần."},
    {"word": "Normalization", "type": "Database Systems", "meaning": "Chuẩn hóa dữ liệu - Quy trình tổ chức dữ liệu trong cơ sở dữ liệu quan hệ để giảm thiểu dư thừa và tránh dị thường dữ liệu."},
    {"word": "Abstraction", "type": "Object-Oriented Programming", "meaning": "Trừu tượng hóa - Quy trình ẩn đi các chi tiết triển khai phức tạp và chỉ hiển thị các tính năng thiết yếu của đối tượng."},
    {"word": "Garbage Collection", "type": "Runtime Systems", "meaning": "Thu gom rác - Cơ chế quản lý bộ nhớ tự động giúp giải phóng vùng nhớ của các đối tượng không còn được sử dụng."},
    {"word": "Serialization", "type": "Data Engineering", "meaning": "Tuần tự hóa - Quy trình chuyển đổi cấu trúc dữ liệu hoặc trạng thái đối tượng thành định dạng byte để lưu trữ hoặc truyền qua mạng."}
]

INTERVALS = [1, 2, 4, 7, 15, 30, 60]

# Load DQN models
@st.cache_resource
def load_rl_agents():
    online_agent = DQNAgent(state_dim=3, action_dim=7)
    offline_agent = DQNAgent(state_dim=3, action_dim=7)
    
    online_loaded = False
    offline_loaded = False
    
    if os.path.exists("models/dqn_agent_online.pt"):
        online_agent.load("models/dqn_agent_online.pt")
        online_loaded = True
        
    if os.path.exists("models/dqn_agent_offline.pt"):
        offline_agent.load("models/dqn_agent_offline.pt")
        offline_loaded = True
        
    return online_agent, online_loaded, offline_agent, offline_loaded

online_agent, online_loaded, offline_agent, offline_loaded = load_rl_agents()

# Initialize session state variables
if "cards" not in st.session_state:
    # Initialize vocabulary card memory properties
    cards = []
    for i, item in enumerate(VOCABULARY):
        cards.append({
            "id": i,
            "word": item["word"],
            "type": item["type"],
            "meaning": item["meaning"],
            "difficulty": 0.5,           # Default moderate difficulty
            "half_life": 5.0,            # Default starting stability (5.0 days)
            "last_reviewed": 0.0,        # Initial time
            "consecutive_corrects": 0,   # Leitner box tracker
            "ef": 2.5,                   # SM-2 Easiness Factor
            "n": 0,                      # SM-2 consecutive repetitions
            "sm2_interval": 1.0,         # SM-2 continuous interval
            "history_seen": 0,           # Số lần đã gặp (cho RL)
            "history_correct": 0         # Số lần trả lời đúng (cho RL)
        })
    st.session_state.cards = cards
    st.session_state.current_day = 0.0
    st.session_state.history = []
    st.session_state.current_card_index = 0
    st.session_state.show_answer = False

# Sidebar config
st.sidebar.markdown("<h2 style='color: #818cf8;'>⚙️ Project Settings</h2>", unsafe_allow_html=True)
algorithm = st.sidebar.selectbox(
    "Choose Scheduler Algorithm:",
    ["DQN (Online RL - Simulator)", "DQN (Offline RL - Anki)", "SuperMemo-2 (SM-2)", "Leitner System", "Random Scheduler"]
)

# Show model status in sidebar
if "DQN" in algorithm:
    is_online = "Online" in algorithm
    loaded = online_loaded if is_online else offline_loaded
    agent_name = "Online DQN" if is_online else "Offline DQN"
    
    if loaded:
        st.sidebar.success(f"🤖 {agent_name} Model loaded!")
    else:
        st.sidebar.warning(f"⚠️ {agent_name} model not found. Training recommended.")
        if st.sidebar.button(f"Train {agent_name} Now"):
            with st.spinner(f"Training {agent_name}..."):
                from train import train_dqn, train_dqn_offline
                from evaluate import compare_schedulers
                if is_online:
                    train_dqn(num_episodes=800)
                else:
                    train_dqn_offline()
                compare_schedulers()
                st.sidebar.success("🎉 Trained and Evaluated!")
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📅 Virtual Current Day:** `{st.session_state.current_day:.1f}`")

# Main Page Title
st.markdown("<div class='main-title'>Personalized Flashcard Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Applying Reinforcement Learning to optimize Spaced Repetition review cycles.</div>", unsafe_allow_html=True)

# Select the card that is most overdue for review
def get_due_card():
    # A card is overdue if current_day >= last_reviewed + current_interval
    # For simulation, we compute: t_overdue = current_day - last_reviewed - current_interval
    # Or simply finding the card with lowest recall probability
    cards = st.session_state.cards
    day = st.session_state.current_day
    
    probabilities = []
    for c in cards:
        t = day - c["last_reviewed"]
        p_recall = 2.0 ** (-t / c["half_life"])
        probabilities.append((p_recall, c["id"]))
        
    # Return card with lowest recall probability (most urgent)
    probabilities.sort()
    return probabilities[0][1]

# Set current card
if "current_card_index" not in st.session_state or st.session_state.current_card_index is None:
    st.session_state.current_card_index = get_due_card()

card_idx = st.session_state.current_card_index
card = st.session_state.cards[card_idx]

# Helper function for Gemini API flashcard generation
def generate_flashcards_api(text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    prompt = f"""
    You are an expert computer science educator. Analyze the text below and extract key technical terms, concepts, or algorithms.
    Return a JSON array of objects. Each object MUST have exactly these keys:
    - "word": the name of the concept or term (in English).
    - "type": the category or classification of this term (e.g. Object-Oriented Programming, Database Systems, Network Protocol, etc.).
    - "meaning": a comprehensive, high-quality definition and explanation of the concept in Vietnamese. It should explain what the concept is, how it works, or its core purpose in detail (similar to: "Là một phương pháp lập trình (programming paradigm) tổ chức thiết kế phần mềm xoay quanh dữ liệu hoặc đối tượng (objects) thay vì hàm và logic...").
    
    Text to analyze:
    {text}
    """
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        res_json = response.json()
        content = res_json['candidates'][0]['content']['parts'][0]['text']
        cards_list = json.loads(content)
        return cards_list, None
    except Exception as e:
        return None, str(e)

# Navigation Tabs
tab1, tab2 = st.tabs(["📖 Study Session & Analytics", "⚡ AI Card Creator"])

with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📝 Study Session")
        
        # Calculate recall probability at current moment
        t_elapsed = st.session_state.current_day - card["last_reviewed"]
        current_p_recall = 2.0 ** (-t_elapsed / card["half_life"])
        
        # Display Card
        st.markdown(f"""
        <div class='flashcard'>
            <div class='flashcard-type'>{card['type']}</div>
            <div class='flashcard-word'>{card['word']}</div>
            {f"<div class='flashcard-meaning'>{card['meaning']}</div>" if st.session_state.show_answer else ""}
        </div>
        """, unsafe_allow_html=True)
        
        # Buttons layout
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        
        if not st.session_state.show_answer:
            if btn_col2.button("👁️ Show Answer", use_container_width=True):
                st.session_state.show_answer = True
                st.rerun()
        else:
            # Handle card review updates
            def review_callback(recalled):
                c = st.session_state.cards[card_idx]
                day = st.session_state.current_day
                t = day - c["last_reviewed"]
                
                p_recall = 2.0 ** (-t / c["half_life"])
                old_h = c["half_life"]
                
                # --- Algorithm Calculations ---
                if "DQN" in algorithm:
                    state = np.array([c["history_seen"], c["history_correct"], t], dtype=np.float32)
                    current_agent = online_agent if "Online" in algorithm else offline_agent
                    action = current_agent.select_action(state, evaluate=True)
                    interval = INTERVALS[action]
                    
                elif algorithm == "SuperMemo-2 (SM-2)":
                    if recalled:
                        grade = 5 if p_recall > 0.85 else 4
                    else:
                        grade = 1 if p_recall > 0.20 else 0
                    
                    if grade >= 3:
                        if c["n"] == 0:
                            c["sm2_interval"] = 1.0
                        elif c["n"] == 1:
                            c["sm2_interval"] = 4.0
                        else:
                            c["sm2_interval"] = c["sm2_interval"] * c["ef"]
                        c["n"] += 1
                    else:
                        c["n"] = 0
                        c["sm2_interval"] = 1.0
                    
                    c["ef"] = max(1.3, c["ef"] + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)))
                    interval = int(np.round(c["sm2_interval"]))
                    
                elif algorithm == "Leitner System":
                    if recalled:
                        c["consecutive_corrects"] = min(c["consecutive_corrects"] + 1, len(INTERVALS) - 1)
                    else:
                        c["consecutive_corrects"] = 0
                    interval = INTERVALS[c["consecutive_corrects"]]
                    
                else:  # Random
                    interval = np.random.choice(INTERVALS)
                    
                # Update memory stability
                if recalled:
                    diff_factor = 1.3 - 0.8 * c["difficulty"]
                    factor = (1.5 + 4.0 * (1.0 - p_recall)) * diff_factor
                    factor = max(1.1, min(factor, 8.0))
                    new_h = old_h * factor
                else:
                    new_h = max(0.5, old_h * 0.3)
                    
                new_h = min(new_h, 365.0)
                
                # Save updates
                c["history_seen"] += 1
                if recalled:
                    c["history_correct"] += 1
                c["half_life"] = new_h
                c["last_reviewed"] = day
                
                # Log history
                st.session_state.history.append({
                    "word": c["word"],
                    "day": day,
                    "interval": interval,
                    "recalled": recalled,
                    "algorithm": algorithm,
                    "old_h": old_h,
                    "new_h": new_h
                })
                
                # Advance time
                st.session_state.current_day += max(0.1, float(interval) * 0.1)
                
                # Reset UI state
                st.session_state.show_answer = False
                st.session_state.current_card_index = None
                st.rerun()

            if btn_col1.button("❌ Incorrect (Forgot)", type="secondary", use_container_width=True):
                review_callback(recalled=False)
                
            if btn_col3.button("✅ Correct (Remembered)", type="primary", use_container_width=True):
                review_callback(recalled=True)

    with col2:
        st.markdown("### 📊 Learning Analytics")
        
        # Compute Statistics
        history_df = pd.DataFrame(st.session_state.history)
        total_revs = len(history_df)
        
        if total_revs > 0:
            corrects = history_df["recalled"].sum()
            accuracy = (corrects / total_revs) * 100
        else:
            accuracy = 100.0
            
        avg_stability = np.mean([c["half_life"] for c in st.session_state.cards])
        
        # Statistics cards layout
        st.markdown(f"""
        <div class='stat-container'>
            <div class='stat-card'>
                <div class='stat-value'>{total_revs}</div>
                <div class='stat-label'>Total Reviews</div>
            </div>
            <div class='stat-card'>
                <div class='stat-value'>{accuracy:.1f}%</div>
                <div class='stat-label'>Recall Accuracy</div>
            </div>
            <div class='stat-card'>
                <div class='stat-value'>{avg_stability:.1f}d</div>
                <div class='stat-label'>Avg Memory Stability</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Plot stability chart
        st.markdown("**Card Memory Half-lives (Days)**")
        words = [c["word"] for c in st.session_state.cards]
        half_lives = [c["half_life"] for c in st.session_state.cards]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(words, half_lives, color="#6366f1", height=0.6)
        ax.set_xlabel("Memory Stability (days)")
        ax.set_title("Current Memory Half-Life per Word")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Deck Details table
    st.markdown("### 🗂️ Memory Deck Overview")
    deck_data = []
    for c in st.session_state.cards:
        t = st.session_state.current_day - c["last_reviewed"]
        p_recall = 2.0 ** (-t / c["half_life"])
        deck_data.append({
            "Word": c["word"],
            "Difficulty": f"{c['difficulty']:.2f}",
            "Memory Stability (Half-life)": f"{c['half_life']:.2f} days",
            "Last Reviewed": f"Day {c['last_reviewed']:.1f}",
            "Recall Prob. Now": f"{p_recall * 100:.1f}%"
        })
    st.table(pd.DataFrame(deck_data))

    # History log
    if len(st.session_state.history) > 0:
        st.markdown("### 📜 Session History Logs")
        st.dataframe(history_df.tail(10)[["day", "word", "interval", "recalled", "old_h", "new_h", "algorithm"]].iloc[::-1])

with tab2:
    st.markdown("### ⚡ AI Flashcard Generator (Powered by Google Gemini)")
    st.markdown("Dán tài liệu học tập hoặc ghi chú của bạn vào đây để AI tự động trích xuất các thuật ngữ chuyên ngành CSE và dịch nghĩa tiếng Việt tương ứng.")
    
    # API Key Input
    api_key = st.text_input("Google Gemini API Key:", type="password", help="Lấy API Key miễn phí từ Google AI Studio")
    
    study_text = st.text_area("Tài liệu học tập:", height=200, placeholder="Ví dụ: Abstraction is the concept of OOP that shows only essential attributes and hides unnecessary information. Polymorphism is the ability of an object to take on many forms...")
    
    if st.button("Tự động trích xuất thẻ"):
        if not api_key:
            st.error("Vui lòng nhập Gemini API Key để sử dụng tính năng này.")
        elif not study_text.strip():
            st.warning("Vui lòng nhập nội dung tài liệu.")
        else:
            with st.spinner("AI đang phân tích tài liệu và trích xuất các thuật ngữ..."):
                cards_list, err = generate_flashcards_api(study_text, api_key)
                if err:
                    st.error(f"Lỗi khi gọi API: {err}")
                elif not cards_list:
                    st.warning("Không tìm thấy thuật ngữ mới nào từ văn bản.")
                else:
                    # Add newly generated cards
                    start_id = len(st.session_state.cards)
                    new_cards_added = 0
                    added_details = []
                    
                    for c_new in cards_list:
                        # Validate structure
                        if not all(k in c_new for k in ["word", "type", "meaning"]):
                            continue
                        
                        # Avoid duplicates
                        if any(c["word"].lower() == c_new["word"].lower() for c in st.session_state.cards):
                            continue
                        
                        st.session_state.cards.append({
                            "id": start_id + new_cards_added,
                            "word": c_new["word"],
                            "type": c_new["type"],
                            "meaning": c_new["meaning"],
                            "difficulty": 0.5,
                            "half_life": 5.0,
                            "last_reviewed": st.session_state.current_day,
                            "consecutive_corrects": 0,
                            "ef": 2.5,
                            "n": 0,
                            "sm2_interval": 1.0,
                            "history_seen": 0,
                            "history_correct": 0
                        })
                        new_cards_added += 1
                        added_details.append(c_new)
                        
                    if new_cards_added > 0:
                        st.success(f"🎉 Đã thêm thành công {new_cards_added} thẻ mới vào bộ bài!")
                        st.table(pd.DataFrame(added_details))
                        # Rerun to update
                        st.rerun()
                    else:
                        st.info("Các từ vựng được trích xuất đã tồn tại trong bộ bài.")
