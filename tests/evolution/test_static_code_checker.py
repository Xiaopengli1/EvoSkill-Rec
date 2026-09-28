from recskill.evolution import StaticCodeChecker

from .helpers import SAFE_GENERATED_CODE


def test_static_checker_rejects_unsafe_import_and_eval():
    checker = StaticCodeChecker()

    unsafe_import = checker.check("import subprocess\nsubprocess.run(['ls'])\n")
    unsafe_eval = checker.check("x = eval('1 + 1')\n")

    assert not unsafe_import["passed"]
    assert "Import rejected: subprocess" in unsafe_import["issues"]
    assert not unsafe_eval["passed"]
    assert "Dangerous call rejected: eval" in unsafe_eval["issues"]


def test_static_checker_accepts_safe_torch_module():
    result = StaticCodeChecker().check(SAFE_GENERATED_CODE)

    assert result["passed"], result
