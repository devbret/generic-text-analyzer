import sys

from analyzer.startup import configure_logging, verify_dependencies

configure_logging()
verify_dependencies()

from analyzer.batch import run

if __name__ == "__main__":
    sys.exit(run())
