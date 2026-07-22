import importlib
import io
import unittest


runner = importlib.import_module("tests.__main__")


class SubtestOutcomeTests(unittest.TestCase):
    def test_failed_subtests_are_counted_in_the_parent_group(self):
        class ExampleTests(unittest.TestCase):
            def runTest(self):
                with self.subTest(kind="failure"):
                    self.fail("failed subtest")
                with self.subTest(kind="error"):
                    raise RuntimeError("errored subtest")

        stream = io.StringIO()
        result = runner.ColoredTextTestResult(
            stream=unittest.runner._WritelnDecorator(stream),
            descriptions=True,
            verbosity=0,
        )

        ExampleTests().run(result)

        self.assertEqual(
            result.group_stats["Example"],
            {"OK": 0, "skipped": 0, "warning": 0, "failed": 2},
        )


if __name__ == "__main__":
    unittest.main()
