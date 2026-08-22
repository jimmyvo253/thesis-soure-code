import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_mdp_loop():
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    
    # 1. Draw Agent Box (DQN Scheduler)
    agent_box = patches.FancyBboxPatch(
        (1.5, 1.0), 2.2, 2.2, 
        boxstyle="round,pad=0.1", 
        fc="#E8F0FE", ec="#1A73E8", lw=2
    )
    ax.add_patch(agent_box)
    ax.text(2.6, 2.1, "Agent\n(DQN Scheduler)", ha='center', va='center', fontsize=12, fontweight='bold', color='#1A73E8')
    
    # 2. Draw Environment Box (User Simulator)
    env_box = patches.FancyBboxPatch(
        (6.3, 1.0), 2.2, 2.2, 
        boxstyle="round,pad=0.1", 
        fc="#FCE8E6", ec="#D93025", lw=2
    )
    ax.add_patch(env_box)
    ax.text(7.4, 2.1, "Environment\n(FlashcardEnv)", ha='center', va='center', fontsize=12, fontweight='bold', color='#D93025')
    
    # 3. Draw Initial State Arrow (Entering Agent from the left)
    arrow_initial = patches.FancyArrowPatch(
        (0.2, 2.1), (1.5, 2.1), 
        arrowstyle="Simple,tail_width=1.5,head_width=8,head_length=8", 
        color="#1A73E8", lw=1.5
    )
    ax.add_patch(arrow_initial)
    ax.text(0.85, 2.3, "Initial\nState $s$", ha='center', va='center', fontsize=10, color='#1A73E8', fontweight='semibold')
    
    # 4. Draw Top Action Arrow (Straight)
    arrow_action = patches.FancyArrowPatch(
        (3.7, 2.7), (6.3, 2.7), 
        arrowstyle="Simple,tail_width=1.5,head_width=8,head_length=8", 
        color="#34A853", lw=1.5
    )
    ax.add_patch(arrow_action)
    ax.text(5.0, 2.9, r"Action $a \in A$ (Interval)", ha='center', va='center', fontsize=11, color='#137333', fontweight='semibold')
    
    # 5. Draw Next State Arrow (Straight)
    arrow_state = patches.FancyArrowPatch(
        (6.3, 2.1), (3.7, 2.1), 
        arrowstyle="Simple,tail_width=1.5,head_width=8,head_length=8", 
        color="#1A73E8", lw=1.5
    )
    ax.add_patch(arrow_state)
    ax.text(5.0, 2.3, r"Next State $s'$ ($[H'_s, H'_c, t']$)", ha='center', va='center', fontsize=10, color='#1A73E8', fontweight='semibold')
    
    # 6. Draw Reward Arrow (Straight)
    arrow_reward = patches.FancyArrowPatch(
        (6.3, 1.5), (3.7, 1.5), 
        arrowstyle="Simple,tail_width=1.5,head_width=8,head_length=8", 
        color="#F9AB00", lw=1.5
    )
    ax.add_patch(arrow_reward)
    ax.text(5.0, 1.7, r"Reward $r$", ha='center', va='center', fontsize=11, color='#B06000', fontweight='semibold')
    
    plt.tight_layout()
    plt.savefig("results/mdp_loop.png", bbox_inches='tight')
    plt.close()
    print("Successfully generated results/mdp_loop.png")

def draw_debug_workflow():
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    steps = [
        ("1. DATA LEAKAGE\nResolved\n(Composite Grouping)", 1.5, "#E8F0FE", "#1A73E8"),
        ("2. BUFFER TRUNCATION\nResolved\n(Expanded Capacity\n+ Shuffling)", 4.5, "#E6F4EA", "#137333"),
        ("3. NON-DETERMINISM\nResolved\n(Global Seeding)", 7.5, "#FEF7E0", "#B06000"),
        ("4. Q-VALUE EXTRAPOLATION\nMitigated\n(CQL-inspired\nRegularization)", 10.5, "#FCE8E6", "#C5221F")
    ]
    
    # Draw Boxes
    for text, x, bg_color, border_color in steps:
        box = patches.FancyBboxPatch(
            (x - 1.2, 1.2), 2.4, 1.6, 
            boxstyle="round,pad=0.1", 
            fc=bg_color, ec=border_color, lw=2
        )
        ax.add_patch(box)
        ax.text(x, 2.0, text, ha='center', va='center', fontsize=9, fontweight='bold', color=border_color)
        
    # Draw Arrows
    for i in range(3):
        arrow = patches.FancyArrowPatch(
            (steps[i][1] + 1.3, 2.0), (steps[i+1][1] - 1.3, 2.0), 
            arrowstyle="Simple,tail_width=2,head_width=8,head_length=8", 
            color="#5F6368"
        )
        ax.add_patch(arrow)
        
    plt.tight_layout()
    plt.savefig("results/debug_workflow.png", bbox_inches='tight')
    plt.close()
    print("Successfully generated results/debug_workflow.png")

if __name__ == "__main__":
    draw_mdp_loop()
    draw_debug_workflow()
