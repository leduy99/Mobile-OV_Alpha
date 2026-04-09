#!/usr/bin/env python3
"""Bootstrap a local image/text WebDataset tar directory into the repo manifest format."""

from mobile_o_bootstrap_common import run_bootstrap


def main() -> None:
    run_bootstrap(
        description=__doc__ or "Bootstrap a local image/text WebDataset tar directory into the repo manifest format.",
        default_repo_id="",
        default_output_root="data/local_wds_image",
        default_dataset_name="local_wds_image",
        default_filenames="all",
    )


if __name__ == "__main__":
    main()
