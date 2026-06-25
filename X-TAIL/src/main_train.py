from . import utils
from .args import parse_arguments
from .models.finetune import finetune


def main(args) -> None:
    utils.seed_all(args.seed)
    finetune(args)


if __name__ == "__main__":
    main(parse_arguments())
