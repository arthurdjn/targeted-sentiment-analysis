r"""
``NoNLPL`` is a dataset instance used to load pre-trained embeddings.
"""

import os
from torchtext.vocab import Vectors

from ._utils import download_from_url, extract_to_dir


class NoNLPL(Vectors):
    r"""The Norwegian Bokmal NLPL dataset contains more than 1,000,000 pre-trained word embeddings from
    the norwegian language.

    Examples::

        >>> vectors = NoNLPL.load()

    """

    urls = ['http://vectors.nlpl.eu/repository/20/58.zip']
    name = '58'
    dirname = 'nlpl-vectors'

    def __init__(self, filepath):
        super().__init__(filepath)

    @classmethod
    def load(cls, data='model.txt', root='.vector_cache'):
        r"""Load pre-trained word embeddings.

        Args:
            data (sting): string of the data containing the pre-trained word embeddings.
            root (string): root folder where vectors are saved.

        Returns:
            NoNLPL: loaded dataset.

        """
        path = os.path.join(root, cls.dirname, cls.name)
        # Maybe download
        if not os.path.isdir(path):
            path = cls.download(root)

        filepath = os.path.join(path, data)
        return NoNLPL(filepath)

    @classmethod
    def download(cls, root):
        r"""Download and unzip a web archive (.zip, .gz, or .tgz).

        Args:
            root (str): Folder to download data to.

        Returns:
            string: Path to extracted dataset.

        """
        path_dirname = os.path.join(root, cls.dirname)
        path_name = os.path.join(path_dirname, cls.name)
        if not os.path.isdir(path_dirname):
            for url in cls.urls:
                filename = os.path.basename(url)
                zpath = os.path.join(path_dirname, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print(f'Download {filename} from {url} to {zpath}')
                    download_from_url(url, zpath)
                extract_to_dir(zpath, path_name)

        return path_name
