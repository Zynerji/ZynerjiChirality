"""Expand fp_precomputed.pkl with additional SMILES not yet fingerprinted."""
import argparse
import logging
import multiprocessing
import pickle
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _fp_one(smiles: str) -> tuple[str, object]:
    """Compute fingerprint for one SMILES in a worker process."""
    try:
        from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
        fp = chirality_fingerprint(smiles, nbits=128)
        return smiles, fp
    except Exception as e:
        return smiles, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='targets_work/fp_precomputed.pkl')
    parser.add_argument('--smiles-file', default='targets_work/missing_smiles.txt')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    # Load existing pkl
    pkl_path = Path(args.pkl)
    with open(pkl_path, 'rb') as f:
        cache = pickle.load(f)
    logger.info('Loaded %d existing FPs from %s', len(cache), pkl_path)

    # Load missing SMILES
    smiles_path = Path(args.smiles_file)
    smiles_list = [s.strip() for s in smiles_path.read_text().splitlines() if s.strip()]
    # Filter out any that got added since the file was written
    to_compute = [s for s in smiles_list if s not in cache]
    logger.info('Need to compute %d FPs', len(to_compute))

    if not to_compute:
        logger.info('Nothing to compute, exiting')
        return

    start = time.time()
    done = 0
    failed = 0

    with multiprocessing.Pool(processes=args.workers) as pool:
        for smiles, fp in pool.imap_unordered(_fp_one, to_compute, chunksize=20):
            if fp is not None:
                cache[smiles] = fp
                done += 1
            else:
                failed += 1
            total = done + failed
            if total % 500 == 0 or total == len(to_compute):
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(to_compute) - total) / rate if rate > 0 else 0
                logger.info('[%d/%d] done=%d fail=%d rate=%.1f/s ETA=%.0fmin',
                            total, len(to_compute), done, failed, rate, eta / 60)

    elapsed = time.time() - start
    logger.info('Done! %d FPs computed in %.0fs (%.1f/s) | %d failures',
                done, elapsed, done / elapsed if elapsed > 0 else 0, failed)

    # Save expanded pkl
    with open(pkl_path, 'wb') as f:
        pickle.dump(cache, f)
    size_mb = pkl_path.stat().st_size / 1e6
    logger.info('Saved expanded pkl to %s (%.1f MB, %d total entries)', pkl_path, size_mb, len(cache))


if __name__ == '__main__':
    main()
