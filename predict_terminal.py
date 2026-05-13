import os
import sys
import time

# Add parent directory to path to allow importing from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.explain import load_model_and_scaler, predict_single
except ImportError:
    print("\n[!] Error: Could not import prediction logic. Ensure you are running from the project root.")
    sys.exit(1)

# ANSI Color Codes for "Dark Tech" Terminal Theme
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
MAGENTA = "\033[95m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_cli():
    clear_screen()
    
    # 1. Setup paths and load model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    print(f"{CYAN}{BOLD}--- DIABETES PREDICTION SYSTEM [DPS] ---{RESET}")
    print(f"{BLUE}Initializing system components...{RESET}")
    
    if not os.path.exists(os.path.join(models_dir, "best_model.pkl")):
        print(f"\n{RED}[ERROR] Model files not found!{RESET}")
        print(f"{YELLOW}Please run 'python src/train.py' first to train the models.{RESET}")
        return

    try:
        model, scaler, meta = load_model_and_scaler(models_dir)
        features = meta["feature_names"]
    except Exception as e:
        print(f"\n{RED}[ERROR] Failed to load model: {e}{RESET}")
        return

    time.sleep(0.5)
    print(f"{GREEN}[OK] Model Loaded: {meta['best_model']}{RESET}")
    print("\n" + "="*60)
    print(f"{CYAN}{BOLD}              PATIENT DATA ENTRY{RESET}")
    print("="*60)
    print(f"{BLUE}Please enter clinical measurements for analysis:{RESET}\n")

    # 2. Collect input
    user_input = []
    for f in features:
        while True:
            try:
                # Add a little prompt arrow
                val_str = input(f"  {CYAN}»{RESET} {f:<28}: {BOLD}")
                print(f"{RESET}", end="")
                val = float(val_str)
                user_input.append(val)
                break
            except ValueError:
                print(f"    {RED}Invalid input. Please enter a numerical value.{RESET}")

    # 3. Predict with a "loading" effect
    print("\n" + "="*60)
    print(f"{MAGENTA}Processing data through neural weights...{RESET}", end="\r")
    time.sleep(0.8)
    print(f"{MAGENTA}Calculating feature importance and risk...   {RESET}")
    time.sleep(0.4)
    
    result = predict_single(model, scaler, user_input, features)
    
    # 4. Show Results
    risk_color = GREEN
    if result['risk_level'] == "Moderate Risk":
        risk_color = YELLOW
    elif result['risk_level'] == "High Risk":
        risk_color = RED

    print("\n" + "╔" + "═"*58 + "╗")
    print(f"║ {BOLD}FINAL ASSESSMENT REPORT{RESET}" + " "*(59 - 24) + "║")
    print("╠" + "═"*58 + "╣")
    
    label_str = result['label'].upper()
    print(f"║ {BOLD}PREDICTION{RESET}   : {risk_color}{BOLD}{label_str:<43}{RESET} ║")
    
    prob_str = f"{result['probability']}%"
    print(f"║ {BOLD}CONFIDENCE{RESET}   : {CYAN}{prob_str:<43}{RESET} ║")
    
    risk_str = result['risk_level']
    print(f"║ {BOLD}RISK STATUS{RESET}  : {risk_color}{risk_str:<43}{RESET} ║")
    
    print("╚" + "═"*58 + "╝")
    
    print(f"\n{BOLD}Top Contributing Factors:{RESET}")
    for exp in result["explanations"][:3]:
        impact_color = RED if exp['impact'] == "increases risk" else GREEN
        print(f"  {CYAN}•{RESET} {exp['feature']:<25} {impact_color}{exp['impact']}{RESET}")

    print("\n" + "="*60)
    print(f"{YELLOW}{BOLD}DISCLAIMER:{RESET} This tool is for educational purposes only.")
    print("Consult a medical professional for clinical diagnosis.")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Enable ANSI escape sequences for Windows 10+
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        
    run_cli()
