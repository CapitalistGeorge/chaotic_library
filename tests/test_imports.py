try:
    from chaotic_library.chaotic_measures import hurst_trajectory
    from chaotic_library.enhanced_esn_fan import EnhancedESN_FAN

    print("✅ Все импорты работают корректно!")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
