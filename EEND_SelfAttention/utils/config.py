import yaml

class Config:
    """
    Loads configuration from the YAML file and provides access to constants.
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        """
        Retrieve a value from the YAML config file.

        Args:
            key (str): Configuration key in the format "section.key".
            default: Default value if the key is not found.

        Returns:
            Value from the config file or the default value.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, {})
            if not isinstance(value, dict):
                return value
        return default

# Create a global config instance
config = Config()