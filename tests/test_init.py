import pathlib
import re
import subprocess
import sys
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "smdp_agents"


class PackageImportTests(unittest.TestCase):
    def test_package_exports_every_agent_in_a_clean_interpreter(self):
        # A subprocess rather than a plain import: this has to fail on an
        # import cycle or ordering bug that the already-populated test
        # interpreter would hide.
        result = subprocess.run(
            [sys.executable, "-c", f"import {PACKAGE}; print({PACKAGE}.__all__)"],
            cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("DeepQWrapper", "HarmonicSMAPO", "SMART", "RelaxedSMART", "UCB"):
            self.assertIn(name, result.stdout)


class PackagingTests(unittest.TestCase):
    """The import name is ``smdp_agents``, deliberately not the bare ``agents``.

    ``agents`` is a common top-level module name -- ``openai-agents`` ships one --
    so shipping under it would shadow, or be shadowed by, another package in a
    shared environment. These tests keep the three places that encode the name in
    agreement, and keep a stale import from creeping back in.
    """

    def pyproject(self):
        with open(ROOT / "pyproject.toml", "rb") as handle:
            return tomllib.load(handle)

    def test_the_declared_package_is_the_one_on_disk(self):
        declared = self.pyproject()["tool"]["setuptools"]["packages"]
        self.assertEqual(declared, [PACKAGE])
        self.assertTrue((ROOT / PACKAGE / "__init__.py").is_file())
        self.assertFalse((ROOT / "agents").exists(),
                         "the old top-level `agents` package is back")

    def test_the_distribution_name_matches_the_import_name(self):
        self.assertEqual(self.pyproject()["project"]["name"],
                         PACKAGE.replace("_", "-"))

    def test_the_examples_only_dependencies_are_declared_as_an_extra(self):
        # requirements.txt installs them unconditionally; the package must not,
        # since nothing under smdp_agents/ imports them.
        extra = self.pyproject()["project"]["optional-dependencies"]["examples"]
        self.assertEqual(sorted(extra), ["gymnasium", "matplotlib", "pandas"])
        self.assertEqual(sorted(self.pyproject()["project"]["dependencies"]),
                         ["numpy", "torch"])

    def test_nothing_still_imports_the_old_package_name(self):
        stale = re.compile(r"^\s*(?:from|import)\s+agents\b", re.MULTILINE)
        offenders = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*.py")
            if ".git" not in path.parts and "__pycache__" not in path.parts
            and stale.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
