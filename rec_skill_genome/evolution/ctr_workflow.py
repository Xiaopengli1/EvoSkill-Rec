import sys


sys.dont_write_bytecode = True

from recskill.evolution.ctr_workflow import main


if __name__ == "__main__":
    raise SystemExit(main())
