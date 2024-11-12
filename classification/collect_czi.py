from cryoet_data_portal import Client, Tomogram
import sys
import os
import mrcfile
import numpy as np
from glob import glob


def get_data(path: str):
    """
    Parameters
    ----------
    path: path where to store the data once downloaded.

    Returns None
    -------
    It will create the data structure loaders will look for and download all the available data
    in the czi portal for the selected species.
    """
    client = Client()
    tomos = Tomogram.find(client,
                          [Tomogram.tomogram_voxel_spacing.run.dataset.organism_name._in(["Chlamydomonas reinhardtii",
                                                                                          "Homo sapiens",
                                                                                          "Saccharomyces cerevisiae",
                                                                                          "Schizosaccharomyces pombe"])])
    ids = [t.id for t in tomos]

    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists(f'{path}/scratch'):
        os.mkdir(f'{path}/scratch')

    if not os.path.exists(f'{path}/czi'):
        os.mkdir(f'{path}/czi')

    if not os.path.exists(f'{path}/czi_membrain'):
        os.mkdir(f'{path}/czi_membrain')

    if not os.path.exists(f'{path}/labeled_annotations'):
        os.mkdir(f'{path}/labeled_annotations')

    pixel_size = {}
    counts = -1
    for i in ids:
        counts += 1

        t = Tomogram.find(client, [Tomogram.id == i])

        voxel_spacing = t[0].tomogram_voxel_spacing

        for annotation in voxel_spacing.annotations:
            if annotation.annotation_software is None:
                continue
            if 'membrain' in annotation.annotation_software:
                if not os.path.exists(f'{path}/czi/{t[0].name}.mrc'):
                    t[0].download_mrcfile(f'{path}/czi/')

                annotation.download(f'{path}/scratch', format='mrc')
                os.rename(glob(f'{path}/scratch/*membrain*.mrc')[0], f"{path}/czi_membrain/{t[0].name}.mrc")
                pixel_size[t[0].name] = float(t[0].tomogram_voxel_spacing.voxel_spacing)
                break

        for file in glob(f'{path}/scratch/*'):
            os.remove(file)

    np.save(f'{path}/pixel_size.npy', pixel_size)
    os.rmdir(f'{path}/scratch')


if __name__ == "__main__":
    get_data(sys.argv[1])
