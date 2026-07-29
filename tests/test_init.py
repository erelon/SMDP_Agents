import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackageImportTests(unittest.TestCase):
    def test_package_exports_every_agent_in_a_clean_interpreter(self):
        # A subprocess rather than a plain import: this has to fail on an
        # import cycle or ordering bug that the already-populated test
        # interpreter would hide.
        result = subprocess.run(
            [sys.executable, "-c", "import agents; print(agents.__all__)"],
            cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("DeepQWrapper", "HarmonicPPO", "SMART", "RelaxedSMART", "UCB"):
            self.assertIn(name, result.stdout)


if __name__ == "__main__":
    unittest.main()
