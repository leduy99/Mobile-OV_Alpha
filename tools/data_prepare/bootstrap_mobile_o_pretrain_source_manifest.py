#!/usr/bin/env python3
"""Bootstrap Mobile-O-Pre-Train into the repo's image source-manifest format."""

from mobile_o_bootstrap_common import run_bootstrap


def main() -> None:
    run_bootstrap(
        description=__doc__ or "Bootstrap Mobile-O-Pre-Train into the repo manifest format.",
        default_repo_id="Amshaker/Mobile-O-Pre-Train",
        default_output_root="data/mobile_o_pretrain",
        default_dataset_name="mobile_o_pretrain",
        default_filenames="00000.tar",
    )


if __name__ == "__main__":
    main()
