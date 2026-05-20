import logging
import argparse
import nibabel as nib
import numpy as np
import scipy.ndimage as nd

from . import fake_lesion_mask, OPTIONS, logger
from .utils import setup_loggers, timestampify


def entrypoint(mask: str, out_path: str, params, logging_kwargs: dict = None) -> None:
    """
    passing logging_kwargs as an empty dict turns on default logging kwargs.
    """

    logging_kwargs = logging_kwargs or {
        'verbosity': logging.DEBUG,
        'log_file': timestampify('fake_lesion_mask') + '.txt',
        'console_verbosity': logging.INFO
    }

    setup_loggers(logger, **logging_kwargs)

    logger.info('Gonna run fake_lesion_mask on input "%s"!', mask)

    loaded_mask = nib.load(mask)
    mask_array = np.asanyarray(loaded_mask.dataobj)

    out = fake_lesion_mask(mask_array, params=params, structure=nd.generate_binary_structure(mask_array.ndim, mask_array.ndim))
    out = out.astype(mask_array.dtype, copy=False)

    logger.info('Done! Saving result at "%s"', out_path)
    nib.save(nib.Nifti1Image(out, affine=loaded_mask.affine, header=loaded_mask.header, dtype=np.uint8), out_path)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mask', type=str)
    parser.add_argument('-o', '--output_file', type=str)
    args = parser.parse_args()
    args.output_file = args.output_file or args.mask.replace('.nii.gz', '_flm.nii.gz')

    params = OPTIONS
    entrypoint(args.mask, args.output_file, params)

if __name__ == "__main__":
    cli_main()