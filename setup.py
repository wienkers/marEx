from setuptools import setup


setup(
    name='spot_the_blOb',
    use_scm_version={
        "write_to": "spot_the_blOb/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    }
)
