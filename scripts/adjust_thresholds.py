# -*- coding: utf-8 -*-
"""
Adjust Thresholds
Điều chỉnh thresholds để có signals hợp lý
"""
import argparse

RECOMMENDED_THRESHOLDS = {
    "conservative": {
        "description": "An toàn, ít signals nhưng chất lượng cao",
        "BULL": {
            "min_confidence": 60,
            "min_risk_reward": 1.8,
            "require_trend_alignment": True,
            "require_volume_confirmation": False,
        },
        "SIDEWAYS": {
            "min_confidence": 65,
            "min_risk_reward": 2.0,
            "require_trend_alignment": True,
            "require_volume_confirmation": False,
        },
        "BEAR": {
            "min_confidence": 75,
            "min_risk_reward": 2.5,
            "require_trend_alignment": True,
            "require_volume_confirmation": False,
        },
    },
    "balanced": {
        "description": "Cân bằng giữa số lượng và chất lượng (RECOMMENDED)",
        "BULL": {
            "min_confidence": 50,
            "min_risk_reward": 1.5,
            "require_trend_alignment": True,
            "require_volume_confirmation": False,
        },
        "SIDEWAYS": {
            "min_confidence": 55,
            "min_risk_reward": 1.8,
            "require_trend_alignment": True,
            "require_volume_confirmation": False,
        },
        "BEAR": {
            "min_confidence": 65,
            "min_risk_reward": 2.0,
            "require_trend_alignment": True,
            "require_volume_confirmation": False,
        },
    },
    "aggressive": {
        "description": "Nhiều signals, cần quản lý risk tốt",
        "BULL": {
            "min_confidence": 40,
            "min_risk_reward": 1.3,
            "require_trend_alignment": False,
            "require_volume_confirmation": False,
        },
        "SIDEWAYS": {
            "min_confidence": 45,
            "min_risk_reward": 1.5,
            "require_trend_alignment": False,
            "require_volume_confirmation": False,
        },
        "BEAR": {
            "min_confidence": 55,
            "min_risk_reward": 1.8,
            "require_trend_alignment": True,
            "require_volume_confirmation": False,
        },
    },
}


def show_current_thresholds():
    """Show current thresholds from bot_runner"""
    print("📊 CURRENT THRESHOLDS")
    print("=" * 60)
    print("\nFrom bot_runner_improved.py:")
    print("\nDefault (line 74):")
    print("  min_confidence: 65")
    print("  min_risk_reward: 2.0")
    print("  require_trend_alignment: True")
    print("\nDynamic adjustments:")
    print("  BULL: min_confidence=60, min_risk_reward=1.8")
    print("  SIDEWAYS: min_confidence=70, min_risk_reward=2.0")
    print("  BEAR: min_confidence=80, min_risk_reward=2.5")
    print("\n⚠️ These are VERY STRICT thresholds!")
    print()


def show_recommendations():
    """Show recommended thresholds"""
    print("💡 RECOMMENDED THRESHOLDS")
    print("=" * 60)

    for profile, config in RECOMMENDED_THRESHOLDS.items():
        print("\n{profile.upper()}: {config['description']}")
        print("-" * 60)
        for regime, settings in config.items():
            if regime == "description":
                continue
            print("\n  {regime}:")
            for key, value in settings.items():
                print("    {key}: {value}")
    print()


def generate_code(profile: str):
    """Generate code to update bot_runner"""
    if profile not in RECOMMENDED_THRESHOLDS:
        print("❌ Unknown profile: {profile}")
        return

    config = RECOMMENDED_THRESHOLDS[profile]

    print("\n📝 CODE TO UPDATE bot_runner_improved.py")
    print("=" * 60)
    print("\nProfile: {profile.upper()} - {config['description']}")
    print("\n1. Update default (around line 74):")
    print("-" * 60)

    default = config["SIDEWAYS"]  # Use SIDEWAYS as default
    print(
        f"""
entry_logic = ImprovedEntryLogic(
    min_confidence={default['min_confidence']},
    min_risk_reward={default['min_risk_reward']},
    require_trend_alignment={default['require_trend_alignment']},
    require_volume_confirmation={default['require_volume_confirmation']}
)
"""
    )

    print("\n2. Update dynamic adjustments (around line 294):")
    print("-" * 60)

    for regime in ["BULL", "SIDEWAYS", "BEAR"]:
        settings = config[regime]
        if regime == "BULL":
            print(
                f"""
if regime == 'BULL':
    entry_logic.min_confidence = {settings['min_confidence']}
    entry_logic.min_risk_reward = {settings['min_risk_reward']}
    entry_logic.require_trend_alignment = {settings['require_trend_alignment']}
    position_sizer.max_total_exposure = 0.70
    position_sizer.min_positions = 6
"""
            )
        elif regime == "BEAR":
            print(
                f"""elif regime == 'BEAR':
    entry_logic.min_confidence = {settings['min_confidence']}
    entry_logic.min_risk_reward = {settings['min_risk_reward']}
    entry_logic.require_trend_alignment = {settings['require_trend_alignment']}
    position_sizer.max_total_exposure = 0.30
    position_sizer.min_positions = 2
"""
            )
        else:  # SIDEWAYS
            print(
                f"""else:  # SIDEWAYS / UNKNOWN
    entry_logic.min_confidence = {settings['min_confidence']}
    entry_logic.min_risk_reward = {settings['min_risk_reward']}
    entry_logic.require_trend_alignment = {settings['require_trend_alignment']}
    position_sizer.max_total_exposure = 0.50
    position_sizer.min_positions = 4
"""
            )

    print("\n✅ Copy and paste the code above into bot_runner_improved.py")
    print()


def compare_profiles():
    """Compare all profiles"""
    print("\n📊 PROFILE COMPARISON")
    print("=" * 60)

    print(
        "\n{:<15} {:<12} {:<12} {:<12}".format("Regime", "Conservative", "Balanced", "Aggressive")
    )
    print("-" * 60)

    for regime in ["BULL", "SIDEWAYS", "BEAR"]:
        conf_values = []
        for profile in ["conservative", "balanced", "aggressive"]:
            conf = RECOMMENDED_THRESHOLDS[profile][regime]["min_confidence"]
            conf_values.append(f"{conf}%")

        print("{:<15} {:<12} {:<12} {:<12}".format(regime, *conf_values))

    print("\n💡 Recommendations:")
    print("  • Start with BALANCED profile")
    print("  • Monitor win rate for 1-2 weeks")
    print("  • If win rate >60%: Consider CONSERVATIVE")
    print("  • If win rate <45%: Already using CONSERVATIVE, review strategy")
    print("  • AGGRESSIVE: Only for experienced traders with good risk management")
    print()


def main():
    parser = argparse.ArgumentParser(description="Adjust trading thresholds")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["show", "recommend", "generate", "compare"],
        default="show",
        help="Command to run",
    )
    parser.add_argument(
        "--profile",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Threshold profile",
    )

    args = parser.parse_args()

    if args.command == "show":
        show_current_thresholds()
        print("\n💡 Run 'python adjust_thresholds.py recommend' to see recommendations")

    elif args.command == "recommend":
        show_current_thresholds()
        show_recommendations()
        compare_profiles()

    elif args.command == "generate":
        generate_code(args.profile)

    elif args.command == "compare":
        compare_profiles()


if __name__ == "__main__":
    print("🎯 THRESHOLD ADJUSTMENT TOOL")
    print("=" * 60)
    main()
