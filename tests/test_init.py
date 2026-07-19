import importlib.util
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackageImportTests(unittest.TestCase):
    def test_package_import_matches_optional_dependency_state(self):
        result = subprocess.run(
            [sys.executable, "-c", "import agents; print(agents.__all__)"],
            cwd=ROOT, text=True, capture_output=True
        )
        if importlib.util.find_spec("torch") is None:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No module named 'torch'", result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DeepQWrapper", result.stdout)
            self.assertIn("HarmonicPPO", result.stdout)


if __name__ == "__main__":
    unittest.main()
