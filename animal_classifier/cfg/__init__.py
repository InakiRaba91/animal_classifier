from animal_classifier.cfg.config import FactoryConfig, GlobalConfig

cfg = FactoryConfig(GlobalConfig().ENV_STATE)()
