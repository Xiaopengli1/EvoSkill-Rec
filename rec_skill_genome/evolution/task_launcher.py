import sys


sys.dont_write_bytecode = True

from recskill.evolution.task_launcher import main


if __name__ == "__main__":
    raise SystemExit(main())
