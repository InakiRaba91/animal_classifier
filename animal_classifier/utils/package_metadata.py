import tomlkit


class PackageMetadata(object):
    """Package Metadata read from pyproject.toml to have a single source of info

    Attributes: the attributes will be read and set from pyproject.toml fields
    """

    def __init__(self, path_to_toml="./pyproject.toml"):
        """
        Initialize by setting the path to the pyproject.toml file to read from

        Args:
            path_to_toml: path to pyproject.toml file

        Returns: None
        """
        with open(path_to_toml) as pyproject:
            file_contents = pyproject.read()
        package_info = tomlkit.parse(file_contents)["tool"]["poetry"]
        self.__dict__.update(package_info)

        # ensure package contains at least 'name', 'description' and 'version' fields
        assert hasattr(self, "name")
        assert hasattr(self, "description")
        assert hasattr(self, "version")
