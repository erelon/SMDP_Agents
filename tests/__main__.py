"""Run baseline tests in algorithm groups with color-coded statuses."""

from collections import OrderedDict
import sys
import unittest
import warnings

GREEN = "\033[32m"
YELLOW = "\033[33m"
BRIGHT_RED = "\033[91m"
RESET = "\033[0m"

GROUP_NAMES = {
    "HarmonicTests": "Harmonic / WeightedHarmonic",
    "RelaxedSmartTests": "RelaxedSMART",
    "SmartTests": "SMART",
    "EpsilonGreedyBanditTests": "MAB / ContinuesMAB",
    "UcbTests": "UCB / ContinuosUCB",
    "AgentCoreTests": "Agent",
    "OracleTests": "Oracle",
    "RandomAgentTests": "RandomAgent",
    "DeepQWrapperTests": "DeepQWrapper",
    "GaussianMLPTests": "GaussianMLP",
    "PPOTests": "PPO",
    "JustHmaTests": "Just HMA",
    "PackageImportTests": "Package import",
    "QLearningTests": "QLearning",
    "RLearningTests": "RLearning",
}


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def _group_name(test):
    class_name = type(test).__name__
    return GROUP_NAMES.get(class_name, class_name.removesuffix("Tests"))


def _grouped_suite(suite):
    groups = OrderedDict()
    for test in _flatten(suite):
        groups.setdefault(_group_name(test), []).append(test)
    return unittest.TestSuite(
        unittest.TestSuite(tests) for tests in groups.values()
    )


class ColoredTextTestResult(unittest.TextTestResult):
    """Track and color outcomes for each contiguous algorithm group."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_stats = OrderedDict()
        self.current_group = None
        self._warning_context = None
        self._captured_warnings = None

    @staticmethod
    def _color(text, color):
        return f"{color}{text}{RESET}"

    def _stats(self, test):
        name = _group_name(test)
        return self.group_stats.setdefault(
            name, {"OK": 0, "skipped": 0, "warning": 0, "failed": 0}
        )

    def startTest(self, test):
        group = _group_name(test)
        if group != self.current_group:
            self.stream.writeln(f"\n{group}")
            self.current_group = group
        self._stats(test)
        self._warning_context = warnings.catch_warnings(record=True)
        self._captured_warnings = self._warning_context.__enter__()
        warnings.simplefilter("always")
        super().startTest(test)

    def stopTest(self, test):
        captured = list(self._captured_warnings or ())
        self._warning_context.__exit__(None, None, None)
        self._warning_context = None
        self._captured_warnings = None
        if captured:
            self._stats(test)["warning"] += len(captured)
            for warning in captured:
                rendered = warnings.formatwarning(
                    warning.message,
                    warning.category,
                    warning.filename,
                    warning.lineno,
                    warning.line,
                ).rstrip()
                self.stream.writeln(self._color(rendered, YELLOW))
        super().stopTest(test)

    def addSuccess(self, test):
        unittest.TestResult.addSuccess(self, test)
        self._stats(test)["OK"] += 1
        if self.showAll:
            self._write_status(test, self._color("ok", GREEN))

    def addSkip(self, test, reason):
        unittest.TestResult.addSkip(self, test, reason)
        self._stats(test)["skipped"] += 1
        if self.showAll:
            self._write_status(test, self._color(f"skipped {reason!r}", YELLOW))

    def addExpectedFailure(self, test, err):
        unittest.TestResult.addExpectedFailure(self, test, err)
        self._stats(test)["skipped"] += 1
        if self.showAll:
            self._write_status(test, self._color("expected failure", YELLOW))

    def addFailure(self, test, err):
        unittest.TestResult.addFailure(self, test, err)
        self._stats(test)["failed"] += 1
        if self.showAll:
            self._write_status(test, self._color("FAIL", BRIGHT_RED))

    def addError(self, test, err):
        unittest.TestResult.addError(self, test, err)
        self._stats(test)["failed"] += 1
        if self.showAll:
            self._write_status(test, self._color("ERROR", BRIGHT_RED))

    def addUnexpectedSuccess(self, test):
        unittest.TestResult.addUnexpectedSuccess(self, test)
        self._stats(test)["failed"] += 1
        if self.showAll:
            self._write_status(
                test, self._color("unexpected success", BRIGHT_RED)
            )

    def print_group_summary(self):
        self.stream.writeln("\nResults by group:")
        for group, counts in self.group_stats.items():
            categories = (
                ("OK", "ok", GREEN),
                ("warning", "warnings", YELLOW),
                ("skipped", "skips", YELLOW),
                ("failed", "failures", BRIGHT_RED),
            )
            summary = [
                self._color(f"{counts[key]} {label}", color)
                for key, label, color in categories
                if counts[key]
            ]
            self.stream.writeln(f"{group}: {', '.join(summary)}")


def main(test_names=None):
    loader = unittest.defaultTestLoader
    names = list(sys.argv[1:] if test_names is None else test_names)
    suite = loader.loadTestsFromNames(names) if names else loader.discover("tests")
    result = unittest.TextTestRunner(
        verbosity=2,
        resultclass=ColoredTextTestResult,
    ).run(_grouped_suite(suite))
    result.print_group_summary()
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
